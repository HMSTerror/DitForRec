from __future__ import annotations

import argparse
from pathlib import Path

from ditforrec.utils import read_jsonl


def _available_keys(records: list[dict], keys: list[str]) -> list[str]:
    return [key for key in keys if any(key in record for record in records)]


def _plot_group(axis, records: list[dict], keys: list[str], title: str, ylabel: str) -> bool:
    selected_keys = _available_keys(records, keys)
    if not selected_keys:
        return False

    epochs = [record["epoch"] for record in records]
    for key in selected_keys:
        values = [record.get(key) for record in records]
        axis.plot(epochs, values, marker="o", linewidth=1.6, markersize=3, label=key)
    axis.set_title(title)
    axis.set_xlabel("epoch")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    return True


def plot_training_curves(output_dir: str | Path, save_path: str | Path | None = None) -> Path:
    output_dir = Path(output_dir)
    history_path = output_dir / "train_history.jsonl"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing train history file: {history_path}")

    records = read_jsonl(history_path)
    if not records:
        raise ValueError(f"No records found in {history_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path = Path(save_path) if save_path is not None else output_dir / "training_curves.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    plotted = [
        _plot_group(
            axes[0, 0],
            records,
            ["loss", "denoise_loss", "target_recon_loss", "prior_loss"],
            "Training Losses",
            "loss",
        ),
        _plot_group(
            axes[0, 1],
            records,
            ["ce_loss", "direct_ce_loss"],
            "Ranking Losses",
            "cross entropy",
        ),
        _plot_group(
            axes[1, 0],
            records,
            ["valid_NDCG@5", "valid_NDCG@10", "valid_NDCG@20", "valid_MRR@10"],
            "Validation Ranking Quality",
            "metric",
        ),
        _plot_group(
            axes[1, 1],
            records,
            ["lr", "valid_Hit@10", "valid_Recall@10", "valid_Precision@10"],
            "Learning Rate / Selected Metrics",
            "value",
        ),
    ]

    for axis, was_plotted in zip(axes.ravel(), plotted):
        if not was_plotted:
            axis.axis("off")

    fig.suptitle(output_dir.name, fontsize=14)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DitForRec training curves from train_history.jsonl.")
    parser.add_argument("--output-dir", type=str, required=True, help="Experiment output directory, e.g. outputs/beauty_base_v8_regularized_dit")
    parser.add_argument("--save-path", type=str, default=None, help="Optional path for the output PNG.")
    args = parser.parse_args()
    save_path = plot_training_curves(args.output_dir, args.save_path)
    print(f"Saved training curves to {save_path}")


if __name__ == "__main__":
    main()
