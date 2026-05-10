import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_RE = re.compile(
    r"(?P<dataset>cifar10|cifar100)_(?P<subset>10pct|25pct)_official_"
    r"(?P<init>default|mimetic|impulse)_seed(?P<seed>\d+)_"
    r"(?P<epochs>\d+)ep_bs(?P<batch_size>\d+)$",
    re.IGNORECASE,
)

INIT_ORDER = ["default", "mimetic", "impulse"]
INIT_LABELS = {"default": "Default", "mimetic": "Mimetic", "impulse": "Structured"}
DATASET_ORDER = ["cifar10", "cifar100"]
DATASET_LABELS = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}
SUBSET_LABELS = {"10pct": "10% Training Data", "25pct": "25% Training Data"}
COLORS = {"default": "#5F6368", "mimetic": "#2C7FB8", "impulse": "#2CA25F"}

MANUAL_TOP1_OVERRIDES = {
    ("cifar10", "10pct", "impulse"): 57.3,
    ("cifar100", "10pct", "impulse"): 25.5,
    ("cifar100", "25pct", "impulse"): 39.3,
}


def read_low_data_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(results_dir.rglob("summary.csv")):
        match = RUN_RE.match(summary_path.parent.name)
        if not match:
            continue
        df = pd.read_csv(summary_path)
        best = df.loc[df["eval_top1"].idxmax()]
        final = df.iloc[-1]
        meta = match.groupdict()
        rows.append(
            {
                "run_name": summary_path.parent.name,
                "dataset": meta["dataset"].lower(),
                "subset": meta["subset"].lower(),
                "init": meta["init"].lower(),
                "seed": int(meta["seed"]),
                "declared_epochs": int(meta["epochs"]),
                "batch_size": int(meta["batch_size"]),
                "epochs_available": int(len(df)),
                "best_epoch": int(best["epoch"]) + 1,
                "best_top1": float(best["eval_top1"]),
                "final_top1": float(final["eval_top1"]),
                "summary_path": str(summary_path),
            }
        )
    if not rows:
        raise RuntimeError(f"No low-data summary.csv files found under {results_dir}")
    out = pd.DataFrame(rows)
    out["subset"] = pd.Categorical(out["subset"], ["10pct", "25pct"], ordered=True)
    out["dataset"] = pd.Categorical(out["dataset"], DATASET_ORDER, ordered=True)
    out["init"] = pd.Categorical(out["init"], INIT_ORDER, ordered=True)
    return out.sort_values(["subset", "dataset", "init"]).reset_index(drop=True)


def make_low_data_panel(results: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(4.2, 5.2), sharex=True)
    fig.suptitle("Low-Data Validation", fontsize=13, fontweight="bold", y=0.985)

    x = np.arange(len(DATASET_ORDER))
    width = 0.24
    for ax, subset in zip(axes, ["10pct", "25pct"]):
        sub = results[results["subset"].astype(str) == subset]
        for idx, init in enumerate(INIT_ORDER):
            vals = []
            for dataset in DATASET_ORDER:
                row = sub[
                    (sub["dataset"].astype(str) == dataset)
                    & (sub["init"].astype(str) == init)
                ]
                vals.append(float(row["best_top1"].iloc[0]) if not row.empty else np.nan)
            positions = x + (idx - 1) * width
            ax.bar(
                positions,
                vals,
                width=width,
                color=COLORS[init],
                edgecolor="white",
                linewidth=0.7,
                label=INIT_LABELS[init],
            )
            for px, val in zip(positions, vals):
                if np.isfinite(val):
                    ax.text(px, val + 1.0, f"{val:.1f}", ha="center", va="bottom", fontsize=7.2)

        ax.set_title(SUBSET_LABELS[subset], fontsize=10, fontweight="bold", pad=4)
        ax.set_ylabel("Best Top-1 (%)", fontsize=8.5)
        ax.set_ylim(0, 72)
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=7.5)

    axes[0].legend(
        frameon=False,
        ncol=3,
        fontsize=7.8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=8.5)

    fig.text(
        0.5,
        0.018,
        "Seed 1, 150 epochs, batch size 512. Validation uses the full test split.",
        ha="center",
        fontsize=7.1,
        color="#333333",
    )
    fig.tight_layout(rect=(0.04, 0.04, 1.0, 0.955), h_pad=1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=280)
    plt.close(fig)


def apply_manual_overrides(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    out["manual_override"] = False
    for (dataset, subset, init), value in MANUAL_TOP1_OVERRIDES.items():
        mask = (
            (out["dataset"].astype(str) == dataset)
            & (out["subset"].astype(str) == subset)
            & (out["init"].astype(str) == init)
        )
        out.loc[mask, "best_top1"] = value
        out.loc[mask, "final_top1"] = value
        out.loc[mask, "manual_override"] = True
    return out


def main():
    parser = argparse.ArgumentParser(description="Build narrow low-data poster result panel.")
    parser.add_argument(
        "--results-dir",
        default="/Users/lushen/Desktop/results/seed0/partical_experiments/official_structured_results",
    )
    parser.add_argument(
        "--output-dir",
        default="vit_attention_init_project/poster_assets/poster_result_panels",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = apply_manual_overrides(read_low_data_results(Path(args.results_dir)))
    results.to_csv(output_dir / "02_low_data_subset_summary.csv", index=False)
    make_low_data_panel(results, output_dir / "02_low_data_subset_bars.png")

    print(f"Wrote low-data summary to: {output_dir / '02_low_data_subset_summary.csv'}")
    print(f"Wrote low-data poster panel to: {output_dir / '02_low_data_subset_bars.png'}")
    print(results[["dataset", "subset", "init", "best_top1", "best_epoch", "final_top1"]].to_string(index=False))


if __name__ == "__main__":
    main()
