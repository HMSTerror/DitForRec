from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def _load_feature(feature_root: Path, feature: str) -> np.ndarray:
    text_path = feature_root / "item_text.npy"
    image_path = feature_root / "item_image.npy"
    if feature == "text":
        return np.load(text_path).astype(np.float32)
    if feature == "image":
        return np.load(image_path).astype(np.float32)
    if feature == "combined":
        text = normalize(np.load(text_path).astype(np.float32), axis=1)
        image = normalize(np.load(image_path).astype(np.float32), axis=1)
        return np.concatenate([text, image], axis=1).astype(np.float32)
    raise ValueError(f"Unsupported feature: {feature}")


def build_semantic_ids(
    processed_root: str | Path,
    feature: str = "combined",
    code_len: int = 4,
    codebook_size: int = 256,
    pca_dim: int = 128,
    batch_size: int = 4096,
    seed: int = 2026,
) -> np.ndarray:
    root = Path(processed_root)
    feature_root = root / "features"
    features = _load_feature(feature_root, feature)
    if features.shape[0] <= 1:
        raise ValueError("Need at least one non-padding item to build semantic IDs.")

    item_features = normalize(features[1:], axis=1)
    max_pca_dim = min(pca_dim, item_features.shape[0] - 1, item_features.shape[1])
    if max_pca_dim >= 2 and max_pca_dim < item_features.shape[1]:
        item_features = PCA(n_components=max_pca_dim, random_state=seed).fit_transform(item_features)
        item_features = normalize(item_features.astype(np.float32), axis=1)

    residual = item_features.astype(np.float32)
    codes = np.zeros((features.shape[0], code_len), dtype=np.int64)
    metadata: dict[str, object] = {
        "feature": feature,
        "code_len": code_len,
        "codebook_size": codebook_size,
        "pca_dim": int(max_pca_dim),
        "seed": seed,
        "num_items": int(features.shape[0]),
        "codebooks": [],
    }

    for code_index in range(code_len):
        n_clusters = min(codebook_size, residual.shape[0])
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed + code_index,
            batch_size=min(batch_size, max(n_clusters * 4, 256)),
            n_init="auto",
            max_iter=200,
            reassignment_ratio=0.01,
        )
        labels = kmeans.fit_predict(residual)
        codes[1:, code_index] = labels.astype(np.int64) + 1
        residual = residual - kmeans.cluster_centers_[labels].astype(np.float32)
        metadata["codebooks"].append(
            {
                "index": code_index,
                "clusters": int(n_clusters),
                "inertia": float(kmeans.inertia_),
                "residual_norm": float(np.linalg.norm(residual, axis=1).mean()),
            }
        )

    output_path = feature_root / "item_semantic_ids.npy"
    np.save(output_path, codes)
    with (feature_root / "item_semantic_ids.meta.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return codes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build residual-KMeans semantic IDs for DitForRec items.")
    parser.add_argument("--dataset", type=str, default="beauty")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--processed-root", type=str, default=None)
    parser.add_argument("--feature", type=str, choices=["text", "image", "combined"], default="combined")
    parser.add_argument("--code-len", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    processed_root = Path(args.processed_root) if args.processed_root else Path(args.data_root) / "processed" / args.dataset
    codes = build_semantic_ids(
        processed_root=processed_root,
        feature=args.feature,
        code_len=args.code_len,
        codebook_size=args.codebook_size,
        pca_dim=args.pca_dim,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"Saved semantic IDs to {processed_root / 'features' / 'item_semantic_ids.npy'} with shape {codes.shape}")


if __name__ == "__main__":
    main()
