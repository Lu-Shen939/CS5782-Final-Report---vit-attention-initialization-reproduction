import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from initializers import (  # noqa: E402
    _impulse_targets,
    _optimize_block_qk,
    _qk_attention_from_pos,
    apply_mimetic_initialization,
    apply_trunc_normal_initialization,
)
from utils import set_seed  # noqa: E402
from vit import create_model  # noqa: E402


INIT_ORDER = ["default", "mimetic", "impulse"]
INIT_LABELS = {
    "default": "Default",
    "mimetic": "Mimetic",
    "impulse": "Structured",
}
COLORS = {
    "default": "#5F6368",
    "mimetic": "#2C7FB8",
    "impulse": "#2CA25F",
}

MANUAL_FULL_TOP1_OVERRIDES = {
    ("cifar100", "impulse"): 73.4,
}


def make_full_data_summary(processed_dir: Path, out_path: Path) -> None:
    poster = pd.read_csv(processed_dir / "poster_accuracy_table.csv")
    stability = pd.read_csv(processed_dir / "cifar10_seed_stability.csv")

    fig = plt.figure(figsize=(9.2, 5.2))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[3.0, 1.55],
        hspace=0.35,
        left=0.08,
        right=0.98,
        top=0.80,
        bottom=0.11,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[1, 0])
    fig.suptitle(
        "Full-Data Reproduction: Accuracy and Seed Stability",
        fontsize=13.5,
        fontweight="bold",
        y=0.975,
    )

    datasets = ["cifar10", "cifar100"]
    dataset_labels = ["CIFAR-10", "CIFAR-100"]
    x = np.arange(len(datasets))
    width = 0.23

    for offset_idx, init in enumerate(INIT_ORDER):
        sub = poster[poster["init"] == init].set_index("dataset")
        means = [
            MANUAL_FULL_TOP1_OVERRIDES.get((d, init), float(sub.loc[d, "best_top1_mean"]))
            for d in datasets
        ]
        stds = [float(sub.loc[d, "best_top1_std"]) for d in datasets]
        positions = x + (offset_idx - 1) * width
        ax.bar(
            positions,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=COLORS[init],
            edgecolor="white",
            linewidth=0.8,
            label=INIT_LABELS[init],
        )
        for px, mean in zip(positions, means):
            ax.text(px, mean + 0.45, f"{mean:.1f}", ha="center", va="bottom", fontsize=8.2)

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontsize=10)
    ax.set_ylabel("Best Top-1 accuracy (%)", fontsize=10)
    ax.set_ylim(62, 94)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        fontsize=9.5,
    )

    ax_table.axis("off")
    table_rows = []
    for init in INIT_ORDER:
        row = stability[stability["init"] == init].iloc[0]
        table_rows.append(
            [
                INIT_LABELS[init],
                f"{row['0']:.2f}",
                f"{row['1']:.2f}",
                f"{row['2']:.2f}",
                f"{row['mean']:.2f} +/- {row['std']:.2f}",
            ]
        )

    table = ax_table.table(
        cellText=table_rows,
        colLabels=["CIFAR-10 Init", "Seed 0", "Seed 1", "Seed 2", "Mean +/- Std"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1.0, 1.34)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D5DAE0")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#17324D")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F4F7FA")
        else:
            cell.set_facecolor("white")

    ax_table.set_title(
        "CIFAR-10 Seed Stability at 250 Epochs",
        fontsize=10.4,
        fontweight="bold",
        pad=4,
    )
    fig.text(
        0.5,
        0.026,
        "Common 250-epoch budget. CIFAR-10 reports mean +/- std over seeds 0, 1, 2; CIFAR-100 uses seed 0.",
        ha="center",
        fontsize=7.6,
        color="#333333",
    )
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def make_low_data_placeholder(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.82,
        "Low-Data Subset Validation",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#17324D",
    )
    ax.text(
        0.5,
        0.69,
        "10% / 25% CIFAR experiments running",
        ha="center",
        va="center",
        fontsize=14,
        color="#4A5568",
    )

    headers = ["Dataset", "Subset", "Default", "Mimetic", "Structured"]
    rows = [
        ["CIFAR-10", "10%", "", "", ""],
        ["CIFAR-10", "25%", "", "", ""],
        ["CIFAR-100", "10%", "", "", ""],
        ["CIFAR-100", "25%", "", "", ""],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.08, 0.18, 0.84, 0.38],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#C9D2DA")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#17324D")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#F9FBFC" if row % 2 == 0 else "white")

    ax.text(
        0.5,
        0.08,
        "Planned analysis: does the initialization advantage grow when training data is scarce?",
        ha="center",
        va="center",
        fontsize=11,
        color="#333333",
    )
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def attention_for_init(init: str, seed: int, structured_steps: int):
    set_seed(seed)
    model = create_model("vit_tiny", num_classes=10, img_size=32, patch_size=4)
    apply_trunc_normal_initialization(model)

    if init == "mimetic":
        apply_mimetic_initialization(model)
    elif init == "impulse":
        grid_size = model.patch_embed.grid_size
        pos_embed = model.pos_embed.squeeze(0)
        target, _ = _impulse_targets(
            grid_size=grid_size,
            kernel_size=3,
            num_heads=model.blocks[0].attn.num_heads,
            device=pos_embed.device,
            dtype=pos_embed.dtype,
            eps=1e-3,
        )
        _optimize_block_qk(
            model.blocks[0],
            pos_embed,
            target,
            steps=structured_steps,
            lr=5e-2,
        )
    elif init != "default":
        raise ValueError(init)

    with torch.no_grad():
        pos_embed = model.pos_embed.squeeze(0)
        logits = _qk_attention_from_pos(model.blocks[0], pos_embed)
        attn = logits.softmax(dim=-1).cpu()
    return attn


def normalize_pattern(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float64)
    matrix = matrix - matrix.min()
    denom = matrix.max()
    if denom <= 0:
        return matrix
    return matrix / denom


def make_attention_map(out_path: Path, structured_steps: int = 300) -> None:
    maps = {init: attention_for_init(init, seed=0, structured_steps=structured_steps) for init in INIT_ORDER}
    grid_size = 8
    center = (grid_size // 2) * grid_size + (grid_size // 2)

    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.2))
    fig.suptitle(
        "Initialization-Only Attention Structure",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )

    for col, init in enumerate(INIT_ORDER):
        attn = maps[init]
        matrix = normalize_pattern(attn[0].numpy())
        center_map = normalize_pattern(attn[0, center].reshape(grid_size, grid_size).numpy())

        ax = axes[0, col]
        ax.imshow(matrix, cmap="magma", aspect="equal", interpolation="nearest")
        ax.set_title(INIT_LABELS[init], fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("64 x 64\nattention", fontsize=10)

        ax2 = axes[1, col]
        ax2.imshow(center_map, cmap="magma", aspect="equal", interpolation="nearest")
        ax2.set_xticks([])
        ax2.set_yticks([])
        if col == 0:
            ax2.set_ylabel("Center-token\n8 x 8 view", fontsize=10)

    fig.text(
        0.5,
        0.025,
        "Layer 1, head 1. Values are normalized per panel to emphasize spatial pattern.",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout(rect=(0.02, 0.055, 0.98, 0.93))
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Build the three poster result panels.")
    parser.add_argument(
        "--processed-dir",
        default="vit_attention_init_project/poster_assets/processed_official_results",
    )
    parser.add_argument(
        "--output-dir",
        default="vit_attention_init_project/poster_assets/poster_result_panels",
    )
    parser.add_argument("--structured-steps", type=int, default=300)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    make_full_data_summary(processed_dir, output_dir / "01_full_data_summary_combined.png")
    make_low_data_placeholder(output_dir / "02_low_data_placeholder.png")
    make_attention_map(
        output_dir / "03_attention_map_init.png",
        structured_steps=args.structured_steps,
    )
    print(f"Wrote poster result panels to: {output_dir}")


if __name__ == "__main__":
    main()
