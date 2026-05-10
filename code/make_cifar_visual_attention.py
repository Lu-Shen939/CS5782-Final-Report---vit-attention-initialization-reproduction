from __future__ import annotations

import argparse
import os
import pickle
import tarfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from make_official_checkpoint_attention import INIT_LABELS, INIT_ORDER, find_run, load_model


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_cifar_root(path: Path) -> Path:
    if path.is_file() and path.suffixes[-2:] == [".tar", ".gz"]:
        extract_dir = path.parent
        expected = extract_dir / "cifar-10-batches-py"
        if not expected.exists():
            with tarfile.open(path, "r:gz") as tar:
                tar.extractall(extract_dir)
        return expected
    if (path / "test_batch").exists():
        return path
    if (path / "cifar-10-batches-py" / "test_batch").exists():
        return path / "cifar-10-batches-py"
    raise FileNotFoundError(
        f"Could not find CIFAR-10 test_batch under {path}. "
        "Pass --cifar-root pointing to cifar-10-batches-py or cifar-10-python.tar.gz."
    )


def load_cifar10_test(cifar_root: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(cifar_root / "test_batch", "rb") as handle:
        batch = pickle.load(handle, encoding="latin1")
    images = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(batch["labels"], dtype=np.int64)
    return images, labels


def resize_image(image: np.ndarray, size: int = 224) -> Image.Image:
    resample = getattr(Image.Resampling, "BICUBIC", Image.BICUBIC)
    return Image.fromarray(image).resize((size, size), resample)


def preprocess(image: np.ndarray) -> torch.Tensor:
    image_224 = np.asarray(resize_image(image), dtype=np.float32) / 255.0
    normalized = (image_224 - MEAN) / STD
    return torch.from_numpy(normalized.transpose(2, 0, 1)).float()


def load_models(results_dir: Path, dataset: str, seed: int):
    models = {}
    metrics = {}
    for init in INIT_ORDER:
        run_dir = find_run(results_dir, dataset, init, seed)
        model, checkpoint = load_model(run_dir / "model_best.pth.tar")
        models[init] = model
        metrics[init] = float(checkpoint.get("metric", np.nan))
    return models, metrics


def predict_batches(model, tensors: torch.Tensor, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    preds = []
    confs = []
    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            batch = tensors[start : start + batch_size]
            logits, _ = model(batch, layers=())
            probs = logits.softmax(dim=-1)
            conf, pred = probs.max(dim=-1)
            preds.append(pred.cpu().numpy())
            confs.append(conf.cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs)


def select_examples(
    models,
    images: np.ndarray,
    labels: np.ndarray,
    num_examples: int,
    max_scan: int,
    batch_size: int,
) -> list[int]:
    scan_count = min(max_scan, len(images))
    tensors = torch.stack([preprocess(image) for image in images[:scan_count]])

    pred_by_init = {}
    conf_by_init = {}
    for init, model in models.items():
        pred_by_init[init], conf_by_init[init] = predict_batches(model, tensors, batch_size)

    selected = []
    used_labels = set()
    all_correct = np.ones(scan_count, dtype=bool)
    mean_conf = np.zeros(scan_count, dtype=np.float64)
    for init in INIT_ORDER:
        all_correct &= pred_by_init[init] == labels[:scan_count]
        mean_conf += conf_by_init[init]
    mean_conf /= len(INIT_ORDER)

    candidates = np.where(all_correct)[0]
    candidates = sorted(candidates, key=lambda idx: mean_conf[idx], reverse=True)
    for idx in candidates:
        label = int(labels[idx])
        if label in used_labels:
            continue
        selected.append(int(idx))
        used_labels.add(label)
        if len(selected) == num_examples:
            return selected

    # Fallback: prefer images correctly classified by the structured model, then fill from the scan range.
    structured_correct = np.where(pred_by_init["impulse"] == labels[:scan_count])[0]
    structured_correct = sorted(structured_correct, key=lambda idx: conf_by_init["impulse"][idx], reverse=True)
    for idx in structured_correct:
        if int(idx) not in selected:
            selected.append(int(idx))
        if len(selected) == num_examples:
            return selected
    for idx in range(scan_count):
        if int(idx) not in selected:
            selected.append(int(idx))
        if len(selected) == num_examples:
            return selected
    return selected


def attention_rollout(attn_maps: dict[int, torch.Tensor]) -> np.ndarray:
    layers = sorted(attn_maps)
    first = attn_maps[layers[0]][0]
    num_tokens = first.shape[-1]
    eye = torch.eye(num_tokens)
    rollout = eye.clone()
    for layer in layers:
        attn = attn_maps[layer][0].mean(dim=0)
        attn = attn + eye
        attn = attn / attn.sum(dim=-1, keepdim=True)
        rollout = attn @ rollout
    cls_to_patch = rollout[0, 1:]
    grid = int(np.sqrt(cls_to_patch.numel()))
    heat = cls_to_patch.reshape(grid, grid).numpy()
    heat = heat - heat.min()
    denom = heat.max()
    return heat / denom if denom > 0 else heat


def overlay_heatmap(image: np.ndarray, heat: np.ndarray, alpha: float = 0.66) -> np.ndarray:
    base = np.asarray(resize_image(image), dtype=np.float32) / 255.0
    heat_img = Image.fromarray(np.uint8(np.clip(heat, 0, 1) * 255))
    resample = getattr(Image.Resampling, "BICUBIC", Image.BICUBIC)
    heat_up = np.asarray(heat_img.resize((base.shape[1], base.shape[0]), resample), dtype=np.float32) / 255.0
    color = plt.get_cmap("magma")(heat_up)[..., :3]
    mask = (heat_up**0.75)[..., None] * alpha
    overlay = base * (1 - mask) + color * mask
    return np.clip(overlay, 0, 1)


def make_visual_attention_panel(
    results_dir: Path,
    cifar_root: Path,
    output_path: Path,
    dataset: str,
    seed: int,
    num_examples: int,
    max_scan: int,
    batch_size: int,
) -> None:
    images, labels = load_cifar10_test(resolve_cifar_root(cifar_root))
    models, metrics = load_models(results_dir, dataset, seed)
    selected = select_examples(models, images, labels, num_examples, max_scan, batch_size)

    rows = len(selected)
    cols = 1 + len(INIT_ORDER)
    fig, axes = plt.subplots(rows, cols, figsize=(9.6, 1.95 * rows + 0.9))
    if rows == 1:
        axes = axes[None, :]

    column_titles = ["Image"] + [f"{INIT_LABELS[init]}\n({metrics[init]:.2f}%)" for init in INIT_ORDER]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=10.5, fontweight="bold", pad=8)

    all_layers = tuple(range(12))
    for row_idx, image_idx in enumerate(selected):
        image = images[image_idx]
        label = int(labels[image_idx])
        axes[row_idx, 0].imshow(resize_image(image))
        axes[row_idx, 0].set_ylabel(
            f"{CIFAR10_CLASSES[label]}\nidx {image_idx}",
            fontsize=9.3,
            rotation=0,
            ha="right",
            va="center",
            labelpad=28,
        )
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        tensor = preprocess(image).unsqueeze(0)
        for col_idx, init in enumerate(INIT_ORDER, start=1):
            model = models[init]
            with torch.no_grad():
                logits, attn_maps = model(tensor, layers=all_layers)
                probs = logits.softmax(dim=-1)[0]
                pred = int(probs.argmax().item())
                conf = float(probs[pred].item())
            heat = attention_rollout(attn_maps)
            axes[row_idx, col_idx].imshow(overlay_heatmap(image, heat))
            axes[row_idx, col_idx].text(
                0.5,
                -0.08,
                f"pred: {CIFAR10_CLASSES[pred]} ({conf:.2f})",
                transform=axes[row_idx, col_idx].transAxes,
                ha="center",
                va="top",
                fontsize=7.4,
            )
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "CIFAR-10 Visual Attention Rollout (Seed-0 Model-Best)",
        fontsize=12.8,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.02,
        "CLS-to-patch attention rollout over all 12 layers, averaged over heads. Same CIFAR-10 test images across models.",
        ha="center",
        fontsize=8.2,
        color="#333333",
    )
    fig.tight_layout(rect=(0.03, 0.065, 0.99, 0.91), w_pad=0.5, h_pad=0.85)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    index_path = output_path.with_suffix(".selected_indices.csv")
    with open(index_path, "w", encoding="utf-8") as handle:
        handle.write("test_index,label\n")
        for idx in selected:
            handle.write(f"{idx},{CIFAR10_CLASSES[int(labels[idx])]}\n")
    print(output_path)
    print(index_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make CIFAR-10 visual attention overlays from official checkpoints.")
    parser.add_argument("--results-dir", default="/Users/lushen/Desktop/results/seed0/official_structured_results")
    parser.add_argument("--cifar-root", default="/private/tmp/cifar-10-batches-py")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--max-scan", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output",
        default="vit_attention_init_project/poster_assets/poster_result_panels/04_cifar10_visual_attention_rollout.png",
    )
    args = parser.parse_args()

    make_visual_attention_panel(
        results_dir=Path(args.results_dir),
        cifar_root=Path(args.cifar_root),
        output_path=Path(args.output),
        dataset=args.dataset.lower(),
        seed=args.seed,
        num_examples=args.num_examples,
        max_scan=args.max_scan,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
