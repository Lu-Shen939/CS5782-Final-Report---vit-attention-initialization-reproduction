from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "poster_assets" / "processed_official_results"
OUT = ROOT / "poster_assets" / "poster_result_panels"

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

MANUAL_TOP1_OVERRIDES = {
    ("cifar100", "impulse"): 73.4,
}


def main() -> None:
    run_summary = pd.read_csv(PROCESSED / "run_summary.csv")
    stability = pd.read_csv(PROCESSED / "cifar10_seed_stability.csv")

    seed0 = run_summary[run_summary["seed"] == 0].copy()
    datasets = ["cifar10", "cifar100"]
    dataset_labels = ["CIFAR-10", "CIFAR-100"]

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
        "Full-Data Reproduction: Seed-0 Accuracy and Seed Stability",
        fontsize=13.5,
        fontweight="bold",
        y=0.975,
    )

    x = np.arange(len(datasets))
    width = 0.23
    for offset_idx, init in enumerate(INIT_ORDER):
        sub = seed0[seed0["init"] == init].set_index("dataset")
        vals = [
            MANUAL_TOP1_OVERRIDES.get((d, init), float(sub.loc[d, "best_top1_full"]))
            for d in datasets
        ]
        epochs = [int(sub.loc[d, "epochs_available"]) for d in datasets]
        positions = x + (offset_idx - 1) * width
        ax.bar(
            positions,
            vals,
            width=width,
            color=COLORS[init],
            edgecolor="white",
            linewidth=0.8,
            label=INIT_LABELS[init],
        )
        for px, val, ep in zip(positions, vals, epochs):
            ax.text(px, val + 0.45, f"{val:.1f}", ha="center", va="bottom", fontsize=8.2)
            ax.text(px, 63.0, f"{ep}ep", ha="center", va="bottom", fontsize=7.3, color="#4B5563")

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

    for (row, _col), cell in table.get_celld().items():
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
        "CIFAR-10 Seed Stability at Common 250-Epoch Budget",
        fontsize=10.4,
        fontweight="bold",
        pad=4,
    )
    fig.text(
        0.5,
        0.026,
        "Bars use seed-0 best checkpoint from each completed full-data run; table reports CIFAR-10 seed0/1/2 best Top-1 at 250 epochs.",
        ha="center",
        fontsize=7.6,
        color="#333333",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "01_full_data_summary_seed0_full_combined.png"
    fig.savefig(out_path, dpi=260)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
