from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


def _mean_pool(hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    masked = hidden_state * mask
    denom = mask.sum(dim=1).clamp(min=1.0)
    return masked.sum(dim=1) / denom


def encode_text_features(texts: list[str], model_name: str, batch_size: int = 16) -> np.ndarray:
    from transformers import AutoTokenizer, T5EncoderModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name)
    model.eval()

    vectors = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding text"):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
            outputs = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
            pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            vectors.append(pooled.cpu().numpy())
    return np.concatenate(vectors, axis=0) if vectors else np.zeros((0, model.config.d_model), dtype=np.float32)


def encode_image_features(image_paths: list[str | None], model_name: str, batch_size: int = 8) -> np.ndarray:
    from transformers import AutoProcessor, SiglipVisionModel
    import torch

    processor = AutoProcessor.from_pretrained(model_name)
    model = SiglipVisionModel.from_pretrained(model_name)
    model.eval()

    valid_fallback = Image.new("RGB", (224, 224), color=(0, 0, 0))
    vectors = []
    with torch.no_grad():
        for start in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[start : start + batch_size]
            images = []
            for maybe_path in batch_paths:
                if maybe_path and Path(maybe_path).exists():
                    images.append(Image.open(maybe_path).convert("RGB"))
                else:
                    images.append(valid_fallback.copy())
            inputs = processor(images=images, return_tensors="pt")
            outputs = model(pixel_values=inputs["pixel_values"])
            pooled = outputs.pooler_output
            vectors.append(pooled.cpu().numpy())
    return np.concatenate(vectors, axis=0) if vectors else np.zeros((0, model.config.hidden_size), dtype=np.float32)
