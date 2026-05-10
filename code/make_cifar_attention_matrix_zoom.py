from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from make_cifar_visual_attention import load_cifar10_test, preprocess, resolve_cifar_root
from make_official_checkpoint_attention import find_run, load_model


LAYER_SPECS = [(0, "Layer 1"), (5, "Layer 6"), (11, "Layer 12")]
ROW_SPECS = [
    ("Structured\nHead 1", "impulse", 0),
    ("Structured\nHead 2", "impulse", 1),
    ("Mimetic\nAll heads", "mimetic", "mean"),
    ("Default", "default", "mean"),
]
MODEL_LABELS = {
    "default": "Default",
    "mimetic": "Mimetic",
    "impulse": "Structured",
}


def normalize_for_display(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    # Log scaling makes both the global structure and diagonal bands visible.
    matrix = np.log1p(matrix / (matrix.mean() + 1e-12))
    low, high = np.percentile(matrix, [1.0, 99.7])
    matrix = (matrix - low) / max(high - low, 1e-12)
    return np.clip(matrix, 0, 1)


def load_seed0_models(results_dir: Path, dataset: str, seed: int):
    models = {}
    metrics = {}
    for init in ["default", "mimetic", "impulse"]:
        run_dir = find_run(results_dir, dataset, init, seed)
        model, checkpoint = load_model(run_dir / "model_best.pth.tar")
        models[init] = model
        metrics[init] = float(checkpoint.get("metric", np.nan))
    return models, metrics


def average_attention_maps(
    models: dict[str, torch.nn.Module],
    images: np.ndarray,
    num_images: int,
    batch_size: int,
) -> dict[str, dict[int, torch.Tensor]]:
    tensors = torch.stack([preprocess(image) for image in images[:num_images]])
    layers = tuple(layer for layer, _ in LAYER_SPECS)
    averaged = {}

    for init, model in models.items():
        sums = {}
        count = 0
        with torch.no_grad():
            for start in range(0, len(tensors), batch_size):
                batch = tensors[start : start + batch_size]
                _, maps = model(batch, layers=layers)
                batch_n = batch.shape[0]
                count += batch_n
                for layer in layers:
                    # B x H x T x T -> H x 196 x 196, excluding CLS token.
                    patch_attn = maps[layer][:, :, 1:, 1:].sum(dim=0)
                    sums[layer] = patch_attn if layer not in sums else sums[layer] + patch_attn
        averaged[init] = {layer: sums[layer] / count for layer in layers}

    return averaged


def row_matrix(avg_maps: dict[str, dict[int, torch.Tensor]], init: str, head, layer: int) -> np.ndarray:
    attn = avg_maps[init][layer]
    if head == "mean":
        matrix = attn.mean(dim=0).numpy()
    else:
        matrix = attn[int(head)].numpy()
    return normalize_for_display(matrix)


def add_zoomed_attention_cell(
    fig: plt.Figure,
    matrix: np.ndarray,
    x: float,
    y: float,
    inset_size: float,
    full_size: float,
    zoom_tokens: int,
    cmap: str,
) -> None:
    gap = 0.008
    total_w = inset_size + gap + full_size
    ax = fig.add_axes([x, y, total_w, full_size])
    ax.set_xlim(0, total_w)
    ax.set_ylim(full_size, 0)
    ax.axis("off")

    x_full = inset_size + gap
    ax.imshow(
        matrix[:zoom_tokens, :zoom_tokens],
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
        extent=(0, inset_size, inset_size, 0),
    )
    ax.imshow(
        matrix,
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
        extent=(x_full, x_full + full_size, full_size, 0),
    )

    dash = (0, (3.0, 2.0))
    crop = zoom_tokens / matrix.shape[0] * full_size
    rect_full = Rectangle(
        (x_full, 0),
        crop,
        crop,
        fill=False,
        edgecolor="red",
        linewidth=1.05,
        linestyle=dash,
    )
    rect_inset = Rectangle(
        (0, 0),
        inset_size,
        inset_size,
        fill=False,
        edgecolor="red",
        linewidth=1.05,
        linestyle=dash,
    )
    ax.add_patch(rect_full)
    ax.add_patch(rect_inset)

    # White dashed cross in the zoomed view, matching the paper-style callout.
    mid = inset_size / 2
    ax.plot([0, inset_size], [mid, mid], color="white", linewidth=0.95, linestyle=dash, alpha=0.9)
    ax.plot([mid, mid], [0, inset_size], color="white", linewidth=0.95, linestyle=dash, alpha=0.9)

    # Red connector lines from zoom box to full-map crop.
    ax.plot([inset_size, x_full], [0, 0], color="red", linewidth=0.85, linestyle=dash)
    ax.plot([inset_size, x_full], [inset_size, crop], color="red", linewidth=0.85, linestyle=dash)


def add_impulse_icon(fig: plt.Figure, x: float, y: float, head: int, size: float = 0.035) -> None:
    ax = fig.add_axes([x, y, size, size])
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    ax.axis("off")
    active = (1, 2) if head == 0 else (0, 1)
    for r in range(3):
        for c in range(3):
            yy = 2 - r
            fill = "#d627b8" if (r, c) == active else "#ffffff"
            ax.add_patch(Rectangle((c, yy), 1, 1, facecolor=fill, edgecolor="black", linewidth=0.7))


def make_figure(
    results_dir: Path,
    cifar_root: Path,
    output_path: Path,
    dataset: str,
    seed: int,
    num_images: int,
    batch_size: int,
    zoom_tokens: int,
) -> None:
    images, _labels = load_cifar10_test(resolve_cifar_root(cifar_root))
    models, metrics = load_seed0_models(results_dir, dataset, seed)
    avg_maps = average_attention_maps(models, images, num_images=num_images, batch_size=batch_size)

    fig = plt.figure(figsize=(10.2, 7.1))
    fig.patch.set_facecolor("white")

    left_label_x = 0.075
    grid_left = 0.19
    top = 0.86
    row_step = 0.195
    col_step = 0.278
    inset_size = 0.092
    full_size = 0.145
    cmap = "viridis"

    fig.text(
        0.53,
        0.965,
        "CIFAR-10 Trained Patch-to-Patch Attention Maps",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.53,
        0.928,
        f"Seed {seed} model-best checkpoints; averaged over {num_images} CIFAR-10 test images",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#333333",
    )

    for col, (layer, layer_label) in enumerate(LAYER_SPECS):
        x = grid_left + col * col_step + inset_size * 0.78
        fig.text(x + full_size * 0.54, 0.064, layer_label, ha="center", va="center", fontsize=15)

        for row, (row_label, init, head) in enumerate(ROW_SPECS):
            y = top - row * row_step - full_size
            matrix = row_matrix(avg_maps, init, head, layer)
            add_zoomed_attention_cell(
                fig,
                matrix,
                x=grid_left + col * col_step,
                y=y,
                inset_size=inset_size,
                full_size=full_size,
                zoom_tokens=zoom_tokens,
                cmap=cmap,
            )

    for row, (row_label, init, head) in enumerate(ROW_SPECS):
        y_mid = top - row * row_step - full_size / 2
        if init == "impulse":
            metric = metrics["impulse"]
            label = row_label.replace("Structured", f"Structured\n({metric:.2f}%)")
        else:
            metric = metrics[init]
            label = f"{row_label}\n({metric:.2f}%)"
        fig.text(left_label_x, y_mid, label, ha="right", va="center", fontsize=12.0)
        if init == "impulse":
            add_impulse_icon(fig, left_label_x - 0.035, y_mid - 0.06, int(head))

    fig.text(
        0.53,
        0.022,
        f"Red callout shows zoomed top-left {zoom_tokens}x{zoom_tokens} patch-token block; CLS token excluded.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#333333",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(output_path)


def make_compact_figure(
    results_dir: Path,
    cifar_root: Path,
    output_path: Path,
    dataset: str,
    seed: int,
    num_images: int,
    batch_size: int,
    zoom_tokens: int,
    show_caption: bool = True,
) -> None:
    images, _labels = load_cifar10_test(resolve_cifar_root(cifar_root))
    models, metrics = load_seed0_models(results_dir, dataset, seed)
    avg_maps = average_attention_maps(models, images, num_images=num_images, batch_size=batch_size)

    fig = plt.figure(figsize=(4.9, 3.35))
    fig.patch.set_facecolor("white")

    left_label_x = 0.16
    grid_left = 0.235
    top = 0.84
    row_step = 0.162
    col_step = 0.185
    inset_size = 0.068
    full_size = 0.106
    cmap = "viridis"

    for col, (layer, layer_label) in enumerate(LAYER_SPECS):
        layer_short = layer_label.replace("Layer ", "L")
        x = grid_left + col * col_step + inset_size + full_size * 0.45
        fig.text(x, 0.92, layer_short, ha="center", va="center", fontsize=8.9, fontweight="bold")

        for row, (_row_label, init, head) in enumerate(ROW_SPECS):
            y = top - row * row_step - full_size
            matrix = row_matrix(avg_maps, init, head, layer)
            add_zoomed_attention_cell(
                fig,
                matrix,
                x=grid_left + col * col_step,
                y=y,
                inset_size=inset_size,
                full_size=full_size,
                zoom_tokens=zoom_tokens,
                cmap=cmap,
            )

    compact_labels = {
        ("impulse", 0): f"Struct H1\n{metrics['impulse']:.2f}",
        ("impulse", 1): f"Struct H2\n{metrics['impulse']:.2f}",
        ("mimetic", "mean"): f"Mimetic\n{metrics['mimetic']:.2f}",
        ("default", "mean"): f"Default\n{metrics['default']:.2f}",
    }

    for row, (_row_label, init, head) in enumerate(ROW_SPECS):
        y_mid = top - row * row_step - full_size / 2
        fig.text(
            left_label_x,
            y_mid,
            compact_labels[(init, head)],
            ha="right",
            va="center",
            fontsize=7.6,
        )

    if show_caption:
        fig.text(
            0.5,
            0.045,
            f"Avg. {num_images} test images; CLS excluded; red box = {zoom_tokens}x{zoom_tokens} zoom.",
            ha="center",
            va="center",
            fontsize=5.9,
            color="#333333",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-style CIFAR-10 attention matrix figure with zoom callouts.")
    parser.add_argument("--results-dir", default="/Users/lushen/Desktop/results/seed0/official_structured_results")
    parser.add_argument("--cifar-root", default="/private/tmp/cifar-10-python.tar.gz")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--zoom-tokens", type=int, default=56)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--no-caption", action="store_true")
    parser.add_argument(
        "--output",
        default="vit_attention_init_project/poster_assets/poster_result_panels/05_cifar10_attention_matrix_zoom.png",
    )
    args = parser.parse_args()

    if args.compact:
        make_compact_figure(
            results_dir=Path(args.results_dir),
            cifar_root=Path(args.cifar_root),
            output_path=Path(args.output),
            dataset=args.dataset.lower(),
            seed=args.seed,
            num_images=args.num_images,
            batch_size=args.batch_size,
            zoom_tokens=args.zoom_tokens,
            show_caption=not args.no_caption,
        )
    else:
        make_figure(
            results_dir=Path(args.results_dir),
            cifar_root=Path(args.cifar_root),
            output_path=Path(args.output),
            dataset=args.dataset.lower(),
            seed=args.seed,
            num_images=args.num_images,
            batch_size=args.batch_size,
            zoom_tokens=args.zoom_tokens,
        )


if __name__ == "__main__":
    main()
