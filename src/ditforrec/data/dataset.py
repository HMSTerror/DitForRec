from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ditforrec.utils import read_json, read_jsonl


class SequentialRecommendationDataset(Dataset):
    def __init__(self, root: str | Path, split: str, max_history: int = 50) -> None:
        self.root = Path(root)
        self.split = split
        self.max_history = max_history
        self.records = read_jsonl(self.root / f"{split}.jsonl")
        self.mappings = read_json(self.root / "mappings.json")
        self.item_text = np.load(self.root / "features" / "item_text.npy")
        self.item_image = np.load(self.root / "features" / "item_image.npy")

        self.num_users = len(self.mappings["user_to_id"]) + 1
        self.num_items = len(self.mappings["item_to_id"]) + 1

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        history = record["history"][-self.max_history :]
        pad_len = self.max_history - len(history)
        padded_history = ([0] * pad_len) + history
        history_mask = ([0] * pad_len) + ([1] * len(history))

        history_array = np.asarray(padded_history, dtype=np.int64)
        text_cond = self.item_text[history_array]
        image_cond = self.item_image[history_array]

        return {
            "user_id": record["user_id"],
            "history": history_array,
            "history_mask": np.asarray(history_mask, dtype=np.float32),
            "target": int(record["target"]),
            "text_cond": text_cond.astype(np.float32),
            "image_cond": image_cond.astype(np.float32),
        }


def collate_batch(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "user_id": torch.tensor([item["user_id"] for item in batch], dtype=torch.long),
        "history": torch.tensor(np.stack([item["history"] for item in batch]), dtype=torch.long),
        "history_mask": torch.tensor(np.stack([item["history_mask"] for item in batch]), dtype=torch.float32),
        "target": torch.tensor([item["target"] for item in batch], dtype=torch.long),
        "text_cond": torch.tensor(np.stack([item["text_cond"] for item in batch]), dtype=torch.float32),
        "image_cond": torch.tensor(np.stack([item["image_cond"] for item in batch]), dtype=torch.float32),
    }
