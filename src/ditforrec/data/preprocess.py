from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ditforrec.data.amazon2014 import download_dataset, load_metadata, load_reviews
from ditforrec.utils import ensure_dir, write_json, write_jsonl


def _load_toy_raw(raw_root: Path) -> tuple[list[dict], dict[str, dict]]:
    reviews_path = raw_root / "reviews_toy_5.jsonl"
    meta_path = raw_root / "meta_toy.jsonl"
    reviews = [json.loads(line) for line in reviews_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            metadata[row["asin"]] = row
    return reviews, metadata


def _iterative_k_core(events: list[dict], min_count: int = 5) -> list[dict]:
    current = events
    while True:
        user_counts = Counter(event["reviewerID"] for event in current)
        item_counts = Counter(event["asin"] for event in current)
        filtered = [
            event
            for event in current
            if user_counts[event["reviewerID"]] >= min_count and item_counts[event["asin"]] >= min_count
        ]
        if len(filtered) == len(current):
            return filtered
        current = filtered


def _build_metadata_text(meta: dict) -> str:
    title = meta.get("title", "")
    brand = meta.get("brand", "")
    categories = meta.get("categories", [])
    flattened_categories = []
    for group in categories:
        if isinstance(group, list):
            flattened_categories.extend(group)
        else:
            flattened_categories.append(str(group))
    parts = [title, brand, " ".join(flattened_categories)]
    return " [SEP] ".join(part.strip() for part in parts if part and part.strip())


def _first_image_path(meta: dict) -> str | None:
    candidates = []
    for key in ("image_path", "imageURLHighRes", "imageURL", "imUrl"):
        value = meta.get(key)
        if isinstance(value, list):
            candidates.extend(value)
        elif isinstance(value, str):
            candidates.append(value)
    for candidate in candidates:
        if candidate:
            return candidate
    return None


def _prepare_sequences(events: list[dict], max_history: int) -> dict[str, list[tuple[int, str]]]:
    per_user: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for event in events:
        timestamp = int(event.get("unixReviewTime") or event.get("timestamp") or 0)
        per_user[event["reviewerID"]].append((timestamp, event["asin"]))

    prepared = {}
    for user, rows in per_user.items():
        deduped = sorted({(ts, asin) for ts, asin in rows}, key=lambda x: (x[0], x[1]))
        prepared[user] = deduped[-max_history:]
    return prepared


def _build_mappings(sequences: dict[str, list[tuple[int, str]]]) -> tuple[dict[str, int], dict[str, int]]:
    users = sorted(sequences.keys())
    items = sorted({asin for rows in sequences.values() for _, asin in rows})
    user_to_id = {user: idx + 1 for idx, user in enumerate(users)}
    item_to_id = {item: idx + 1 for idx, item in enumerate(items)}
    return user_to_id, item_to_id


def _build_instances(
    sequences: dict[str, list[tuple[int, str]]],
    user_to_id: dict[str, int],
    item_to_id: dict[str, int],
    max_history: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    train_records: list[dict] = []
    val_records: list[dict] = []
    test_records: list[dict] = []

    for user, rows in sequences.items():
        if len(rows) < 5:
            continue
        item_ids = [item_to_id[asin] for _, asin in rows]
        train_items = item_ids[:-2]
        val_item = item_ids[-2]
        test_item = item_ids[-1]
        user_id = user_to_id[user]

        for idx in range(1, len(train_items)):
            history = train_items[max(0, idx - max_history) : idx]
            train_records.append(
                {
                    "user_id": user_id,
                    "history": history,
                    "target": train_items[idx],
                    "split": "train",
                }
            )

        val_records.append(
            {
                "user_id": user_id,
                "history": train_items[-max_history:],
                "target": val_item,
                "split": "val",
            }
        )
        test_records.append(
            {
                "user_id": user_id,
                "history": (train_items + [val_item])[-max_history:],
                "target": test_item,
                "split": "test",
            }
        )
    return train_records, val_records, test_records


def _download_image_if_needed(image_url: str, output_dir: Path) -> str | None:
    if not image_url.startswith(("http://", "https://")):
        return image_url if Path(image_url).exists() else None
    try:
        import requests

        output_dir.mkdir(parents=True, exist_ok=True)
        local_name = output_dir / Path(image_url).name
        if local_name.exists():
            return str(local_name.resolve())
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()
        local_name.write_bytes(response.content)
        return str(local_name.resolve())
    except Exception:
        return None


def preprocess_dataset(
    dataset: str,
    data_root: str | Path = "data",
    download: bool = False,
    max_history: int = 50,
    extract_text: bool = False,
    extract_image: bool = False,
    text_backbone: str = "google-t5/t5-small",
    image_backbone: str = "google/siglip-base-patch16-224",
) -> Path:
    dataset = dataset.lower()
    data_root = Path(data_root)
    raw_root = ensure_dir(data_root / "raw")
    processed_root = ensure_dir(data_root / "processed" / dataset)
    feature_root = ensure_dir(processed_root / "features")

    if dataset in {"beauty", "toys"}:
        if download:
            files = download_dataset(dataset, raw_root)
            review_path = files["reviews"]
            meta_path = files["meta"]
        else:
            dataset_root = raw_root / dataset
            review_path = next(dataset_root.glob("reviews*.gz"))
            meta_path = next(dataset_root.glob("meta*.gz"))
        reviews = load_reviews(review_path)
        metadata = load_metadata(meta_path)
    elif dataset == "toy":
        reviews, metadata = _load_toy_raw(raw_root / "toy")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    filtered = _iterative_k_core(reviews, min_count=5)
    sequences = _prepare_sequences(filtered, max_history=max_history)
    user_to_id, item_to_id = _build_mappings(sequences)
    train_records, val_records, test_records = _build_instances(sequences, user_to_id, item_to_id, max_history)

    write_jsonl(train_records, processed_root / "train.jsonl")
    write_jsonl(val_records, processed_root / "val.jsonl")
    write_jsonl(test_records, processed_root / "test.jsonl")
    write_json({"user_to_id": user_to_id, "item_to_id": item_to_id}, processed_root / "mappings.json")

    image_cache_root = ensure_dir(processed_root / "downloaded_images")
    item_metadata_records = []
    text_inputs = [""]
    image_inputs = [None]
    image_candidate_count = 0
    image_downloaded_count = 0
    image_missing_count = 0
    for asin, item_id in sorted(item_to_id.items(), key=lambda x: x[1]):
        meta = metadata.get(asin, {"asin": asin})
        text = _build_metadata_text(meta)
        image_path = _first_image_path(meta)
        if image_path:
            image_candidate_count += 1
        if image_path and image_path.startswith(("http://", "https://")):
            image_path = _download_image_if_needed(image_path, image_cache_root)
        if image_path and Path(image_path).exists():
            image_downloaded_count += 1
        else:
            image_missing_count += 1
        item_metadata_records.append(
            {
                "item_id": item_id,
                "asin": asin,
                "title": meta.get("title", ""),
                "brand": meta.get("brand", ""),
                "text": text,
                "image_path": image_path,
                "categories": meta.get("categories", []),
            }
        )
        text_inputs.append(text)
        image_inputs.append(image_path)

    write_jsonl(item_metadata_records, processed_root / "item_metadata.jsonl")

    manifest = {
        "dataset": dataset,
        "text_backbone": text_backbone,
        "image_backbone": image_backbone,
        "num_users": len(user_to_id),
        "num_items": len(item_to_id),
        "num_train": len(train_records),
        "num_val": len(val_records),
        "num_test": len(test_records),
        "image_candidate_count": image_candidate_count,
        "image_downloaded_count": image_downloaded_count,
        "image_missing_count": image_missing_count,
    }

    if extract_text:
        from ditforrec.data.features import encode_text_features

        text_features = encode_text_features(text_inputs, text_backbone)
        np.save(feature_root / "item_text.npy", text_features.astype(np.float32))
        manifest["text_feature_dim"] = int(text_features.shape[1]) if text_features.ndim == 2 else 0
    if extract_image:
        from ditforrec.data.features import encode_image_features

        image_features = encode_image_features(image_inputs, image_backbone)
        np.save(feature_root / "item_image.npy", image_features.astype(np.float32))
        manifest["image_feature_dim"] = int(image_features.shape[1]) if image_features.ndim == 2 else 0

    if not extract_text:
        existing_text_path = feature_root / "item_text.npy"
        if existing_text_path.exists():
            existing_text = np.load(existing_text_path)
            manifest["text_feature_dim"] = int(existing_text.shape[1]) if existing_text.ndim == 2 else 0
        else:
            text_features = np.zeros((len(item_to_id) + 1, 512), dtype=np.float32)
            np.save(existing_text_path, text_features)
            manifest["text_feature_dim"] = 512
    if not extract_image:
        existing_image_path = feature_root / "item_image.npy"
        if existing_image_path.exists():
            existing_image = np.load(existing_image_path)
            manifest["image_feature_dim"] = int(existing_image.shape[1]) if existing_image.ndim == 2 else 0
        else:
            image_features = np.zeros((len(item_to_id) + 1, 768), dtype=np.float32)
            np.save(existing_image_path, image_features)
            manifest["image_feature_dim"] = 768

    write_json(manifest, feature_root / "feature_manifest.json")
    print(
        f"preprocess done: items={len(item_to_id)} image_candidates={image_candidate_count} "
        f"downloaded_images={image_downloaded_count} missing_images={image_missing_count}"
    )
    return processed_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Amazon Beauty/Toys or toy dataset.")
    parser.add_argument("--dataset", type=str, required=True, choices=["beauty", "toys", "toy"])
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--extract-text", action="store_true")
    parser.add_argument("--extract-image", action="store_true")
    parser.add_argument("--text-backbone", type=str, default="google-t5/t5-small")
    parser.add_argument("--image-backbone", type=str, default="google/siglip-base-patch16-224")
    args = parser.parse_args()
    preprocess_dataset(
        dataset=args.dataset,
        data_root=args.data_root,
        download=args.download,
        max_history=args.max_history,
        extract_text=args.extract_text,
        extract_image=args.extract_image,
        text_backbone=args.text_backbone,
        image_backbone=args.image_backbone,
    )


if __name__ == "__main__":
    main()
