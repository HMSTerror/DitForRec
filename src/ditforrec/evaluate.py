from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ditforrec.config import load_config
from ditforrec.data.dataset import SequentialRecommendationDataset, collate_batch
from ditforrec.model.ditforrec import DitForRec
from ditforrec.model.factory import build_model
from ditforrec.utils import create_logger, ensure_dir, flatten_metrics


def _compute_ranks(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    sorted_indices = torch.argsort(logits, dim=1, descending=True)
    matches = sorted_indices.eq(targets.unsqueeze(1))
    return matches.float().argmax(dim=1) + 1


def _metric_name(metric: str, topk: int) -> str:
    return f"{metric}@{topk}"


def _resolve_topk(config) -> list[int]:
    evaluation_cfg = config.get("evaluation", {})
    if "topk" in evaluation_cfg:
        return list(evaluation_cfg["topk"])
    return list(config.training.get("topk", [10]))


def _resolve_sampling_config(config) -> tuple[int | None, str, float, bool]:
    sampling_cfg = config.get("sampling", {})
    inference_steps = sampling_cfg.get("inference_steps")
    if inference_steps is not None:
        inference_steps = int(inference_steps)
    return (
        inference_steps,
        str(sampling_cfg.get("strategy", "ddim")),
        float(sampling_cfg.get("eta", 0.0)),
        bool(sampling_cfg.get("noise_history", True)),
    )


def mask_history_items(logits: torch.Tensor, history_items: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.clone()
    for row in range(history_items.shape[0]):
        target_id = targets[row].item()
        target_logit = masked_logits[row, target_id].item()
        masked_logits[row, history_items[row].unique()] = -1e9
        masked_logits[row, target_id] = target_logit
    return masked_logits


def evaluate_model(model: DitForRec, data_loader: DataLoader, device: torch.device, topk: list[int], config) -> dict[str, float]:
    model.eval()
    total = 0
    metric_totals = {
        _metric_name(metric, k): 0.0
        for metric in ("Hit", "Recall", "NDCG", "MRR", "Precision")
        for k in topk
    }
    inference_steps, sampling_strategy, eta, noise_history = _resolve_sampling_config(config)

    with torch.no_grad():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model.sample_logits(
                user_id=batch["user_id"],
                history=batch["history"],
                history_mask=batch["history_mask"],
                text_cond=batch["text_cond"],
                image_cond=batch["image_cond"],
                inference_steps=inference_steps,
                sampling_strategy=sampling_strategy,
                eta=eta,
                noise_history=noise_history,
            )

            masked_logits = mask_history_items(logits, batch["history"], batch["target"])
            ranks = _compute_ranks(masked_logits, batch["target"])
            total += batch["target"].shape[0]
            for k in topk:
                hits = (ranks <= k).float()
                metric_totals[_metric_name("Hit", k)] += hits.sum().item()
                metric_totals[_metric_name("Recall", k)] += hits.sum().item()
                metric_totals[_metric_name("Precision", k)] += (hits / float(k)).sum().item()
                metric_totals[_metric_name("MRR", k)] += torch.where(
                    hits > 0,
                    1.0 / ranks.float(),
                    torch.zeros_like(ranks, dtype=torch.float32),
                ).sum().item()
                metric_totals[_metric_name("NDCG", k)] += torch.where(
                    hits > 0,
                    1.0 / torch.log2(ranks.float() + 1.0),
                    torch.zeros_like(ranks, dtype=torch.float32),
                ).sum().item()

    return {name: value / max(total, 1) for name, value in metric_totals.items()}


def resolve_metric_value(metrics: dict[str, float], metric_name: str) -> float:
    normalized = metric_name.lower()
    for key, value in metrics.items():
        if key.lower() == normalized:
            return value
    raise KeyError(f"Metric `{metric_name}` not found in metrics: {sorted(metrics)}")


def evaluate(config_path: str, checkpoint_path: str) -> None:
    config = load_config(config_path)
    processed_root = Path(config.data.root) / "processed" / config.data.dataset
    dataset = SequentialRecommendationDataset(processed_root, "test", max_history=config.data.max_history)
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_batch,
    )

    device = torch.device(config.training.device if torch.cuda.is_available() or config.training.device == "cpu" else "cpu")
    model = build_model(config, dataset).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate_model(model, loader, device, topk=_resolve_topk(config), config=config)

    output_root = ensure_dir(Path(checkpoint_path).resolve().parent)
    logger = create_logger(output_root / "eval.log", f"ditforrec.eval.{output_root.name}")
    logger.info("Loaded checkpoint from %s", checkpoint_path)
    logger.info("test result: %s", flatten_metrics(metrics, precision=6))

    with (output_root / "eval_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DitForRec.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
