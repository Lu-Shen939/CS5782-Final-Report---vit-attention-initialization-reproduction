import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item() * 100.0


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_run_dir(output_dir: str, run_name: str) -> Path:
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def plot_accuracy_curve(metrics, out_path: str | Path) -> None:
    if not metrics:
        return
    epochs = [row["epoch"] for row in metrics]
    train_acc = [row["train_acc"] for row in metrics]
    val_acc = [row["val_acc"] for row in metrics]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_acc, label="train")
    plt.plot(epochs, val_acc, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Top-1 accuracy (%)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_metrics_csv(metrics, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in metrics:
            f.write(",".join(str(row[key]) for key in header) + "\n")


def describe_device() -> dict:
    info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
    return info
