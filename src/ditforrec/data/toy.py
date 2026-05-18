from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from ditforrec.utils import ensure_dir, set_seed


def build_toy_dataset(output_root: str | Path, seed: int = 2026) -> None:
    set_seed(seed)
    output_root = Path(output_root)
    raw_root = ensure_dir(output_root / "raw" / "toy")
    image_root = ensure_dir(raw_root / "images")

    categories = ["lipstick", "cream", "serum", "mask", "brush", "soap"]
    brands = ["Aster", "Noma", "Velin", "Purel", "Sana", "Mille"]

    items = []
    for idx in range(1, 37):
        category = categories[idx % len(categories)]
        brand = brands[idx % len(brands)]
        asin = f"TOYITEM{idx:04d}"
        image_path = image_root / f"{asin}.png"
        image = Image.new("RGB", (128, 128), color=(30 + idx, 70 + idx % 100, 90 + idx % 140))
        drawer = ImageDraw.Draw(image)
        drawer.rectangle((12, 12, 116, 116), outline=(250, 250, 250), width=3)
        drawer.text((18, 52), f"{category[:6]}-{idx}", fill=(255, 255, 255))
        image.save(image_path)

        items.append(
            {
                "asin": asin,
                "title": f"{brand} {category} item {idx}",
                "brand": brand,
                "category": category,
                "categories": [["Beauty", category.capitalize()]],
                "image_path": str(image_path.resolve()),
            }
        )

    with (raw_root / "meta_toy.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    rng = random.Random(seed)
    with (raw_root / "reviews_toy_5.jsonl").open("w", encoding="utf-8") as handle:
        timestamp = 1_600_000_000
        for user_idx in range(1, 61):
            user = f"TOYUSER{user_idx:04d}"
            favorite_cat = categories[user_idx % len(categories)]
            secondary_cat = categories[(user_idx + 1) % len(categories)]
            pool = [item for item in items if item["category"] in {favorite_cat, secondary_cat}]
            distractors = [item for item in items if item["category"] not in {favorite_cat, secondary_cat}]
            seq_len = rng.randint(10, 16)
            sequence = [rng.choice(pool) for _ in range(seq_len - 2)] + [rng.choice(distractors) for _ in range(2)]
            rng.shuffle(sequence)
            for item in sequence:
                timestamp += rng.randint(10, 200)
                handle.write(
                    json.dumps(
                        {
                            "reviewerID": user,
                            "asin": item["asin"],
                            "unixReviewTime": timestamp,
                            "overall": 5.0,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a toy dataset for DitForRec.")
    parser.add_argument("--output-root", type=str, default="data")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    build_toy_dataset(args.output_root, args.seed)


if __name__ == "__main__":
    main()
