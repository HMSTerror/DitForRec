from __future__ import annotations

import ast
import gzip
import json
import urllib.request
from pathlib import Path
from typing import Iterator

from ditforrec.utils import ensure_dir


AMAZON_2014_URLS = {
    "beauty": {
        "reviews": "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz",
        "meta": "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz",
    },
    "toys": {
        "reviews": "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz",
        "meta": "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Toys_and_Games.json.gz",
    },
}


def download_dataset(dataset: str, raw_root: str | Path) -> dict[str, Path]:
    dataset = dataset.lower()
    if dataset not in AMAZON_2014_URLS:
        raise ValueError(f"Unsupported dataset: {dataset}")

    target_root = ensure_dir(Path(raw_root) / dataset)
    outputs: dict[str, Path] = {}
    for name, url in AMAZON_2014_URLS[dataset].items():
        destination = target_root / Path(url).name
        if not destination.exists():
            urllib.request.urlretrieve(url, destination)
        outputs[name] = destination
    return outputs


def _iter_gzip_lines(path: str | Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield ast.literal_eval(line)


def load_reviews(path: str | Path) -> list[dict]:
    return list(_iter_gzip_lines(path))


def load_metadata(path: str | Path) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for row in _iter_gzip_lines(path):
        asin = row.get("asin")
        if asin:
            output[asin] = row
    return output
