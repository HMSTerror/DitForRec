from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ditforrec.config import load_config
from ditforrec.data.dataset import SequentialRecommendationDataset, collate_batch
from ditforrec.evaluate import evaluate_model, resolve_metric_value
from ditforrec.model.factory import build_model
from ditforrec.utils import create_logger, ensure_dir, flatten_metrics, set_seed, write_jsonl


def _resolve_topk(config) -> list[int]:
    evaluation_cfg = config.get("evaluation", {})
    if "topk" in evaluation_cfg:
        return list(evaluation_cfg["topk"])
    return list(config.training.get("topk", [10]))


def train(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(config.seed)

    processed_root = Path(config.data.root) / "processed" / config.data.dataset
    train_dataset = SequentialRecommendationDataset(processed_root, "train", max_history=config.data.max_history)
    val_dataset = SequentialRecommendationDataset(processed_root, "val", max_history=config.data.max_history)
    test_dataset = SequentialRecommendationDataset(processed_root, "test", max_history=config.data.max_history)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_batch,
    )

    device = torch.device(config.training.device if torch.cuda.is_available() or config.training.device == "cpu" else "cpu")
    model = build_model(config, train_dataset).to(device)
    optimizer = AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)

    output_root = ensure_dir(Path("outputs") / config.experiment_name)
    shutil.copy2(config_path, output_root / "config_snapshot.yaml")
    logger = create_logger(output_root / "train.log", f"ditforrec.train.{config.experiment_name}")
    logger.info("Experiment: %s", config.experiment_name)
    logger.info("Device: %s", device)

    topk = _resolve_topk(config)
    valid_metric_name = str(config.get("evaluation", {}).get("valid_metric", "NDCG@10"))
    valid_metric_bigger = bool(config.get("evaluation", {}).get("valid_metric_bigger", True))
    eval_every_epochs = int(config.training.get("eval_every_epochs", 1))
    log_every_steps = int(config.get("logging", {}).get("log_every_steps", 50))

    best_metric = float("-inf") if valid_metric_bigger else float("inf")
    best_epoch = 0
    best_valid_metrics: dict[str, float] = {}
    best_test_metrics: dict[str, float] = {}
    history_records: list[dict] = []
    patience = 0

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        denoise_loss_total = 0.0
        target_recon_loss_total = 0.0
        prior_loss_total = 0.0
        ce_loss_total = 0.0
        direct_ce_loss_total = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            optimizer.step()

            total_loss += outputs.loss.item()
            denoise_loss_total += outputs.denoise_loss.item()
            target_recon_loss_total += outputs.target_recon_loss.item()
            prior_loss_total += outputs.prior_loss.item()
            ce_loss_total += outputs.ce_loss.item()
            direct_ce_loss_total += outputs.direct_ce_loss.item()

            if step % log_every_steps == 0 or step == len(train_loader):
                logger.info(
                    "epoch %d training [%d/%d] loss=%.4f, denoise=%.4f, target_recon=%.4f, prior=%.4f, ce=%.4f, direct_ce=%.4f",
                    epoch,
                    step,
                    len(train_loader),
                    total_loss / step,
                    denoise_loss_total / step,
                    target_recon_loss_total / step,
                    prior_loss_total / step,
                    ce_loss_total / step,
                    direct_ce_loss_total / step,
                )

        num_train_steps = max(len(train_loader), 1)
        train_metrics = {
            "loss": total_loss / num_train_steps,
            "denoise_loss": denoise_loss_total / num_train_steps,
            "target_recon_loss": target_recon_loss_total / num_train_steps,
            "prior_loss": prior_loss_total / num_train_steps,
            "ce_loss": ce_loss_total / num_train_steps,
            "direct_ce_loss": direct_ce_loss_total / num_train_steps,
        }
        epoch_time = time.time() - epoch_start
        logger.info("epoch %d finished [time: %.2fs, train: %s]", epoch, epoch_time, flatten_metrics(train_metrics))

        epoch_record: dict[str, float | int] = {"epoch": epoch, **train_metrics}

        if epoch % eval_every_epochs != 0:
            history_records.append(epoch_record)
            continue

        valid_metrics = evaluate_model(model, val_loader, device, topk=topk, config=config)
        current_metric = resolve_metric_value(valid_metrics, valid_metric_name)
        logger.info(
            "epoch %d evaluating [valid_score: %.6f, valid: %s]",
            epoch,
            current_metric,
            flatten_metrics(valid_metrics, precision=6),
        )
        epoch_record.update({f"valid_{key}": value for key, value in valid_metrics.items()})

        is_better = current_metric > best_metric if valid_metric_bigger else current_metric < best_metric
        if is_better:
            best_metric = current_metric
            best_epoch = epoch
            patience = 0
            best_valid_metrics = valid_metrics
            best_test_metrics = evaluate_model(model, test_loader, device, topk=topk, config=config)
            logger.info("epoch %d new best [test: %s]", epoch, flatten_metrics(best_test_metrics, precision=6))
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config_path": str(config_path),
                    "epoch": epoch,
                    "best_valid_score": best_metric,
                    "best_valid_metrics": best_valid_metrics,
                    "best_test_metrics": best_test_metrics,
                },
                output_root / "best.pt",
            )
        else:
            patience += 1
            if patience >= config.training.early_stop_patience:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                history_records.append(epoch_record)
                break

        history_records.append(epoch_record)

    summary = {
        "best_epoch": best_epoch,
        "best_valid_score": best_metric,
        "valid_metric": valid_metric_name,
        "best_valid_result": best_valid_metrics,
        "best_test_result": best_test_metrics,
    }
    with (output_root / "best_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    if config.get("logging", {}).get("save_train_history", True):
        write_jsonl(history_records, output_root / "train_history.jsonl")

    logger.info("Finished training. best epoch=%d, best valid score=%.6f", best_epoch, best_metric)
    if best_valid_metrics:
        logger.info("best valid result: %s", flatten_metrics(best_valid_metrics, precision=6))
    if best_test_metrics:
        logger.info("best test result: %s", flatten_metrics(best_test_metrics, precision=6))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DitForRec.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
