import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"
PROCESSED_DIR = PROJECT_ROOT / "poster_assets" / "processed_official_results"

FULL_RESULTS_DIR = Path("/Users/lushen/Desktop/results/seed0/official_structured_results")
LOW_RESULTS_DIR = Path(
    "/Users/lushen/Desktop/results/seed0/partical_experiments/official_structured_results"
)

INIT_ORDER = ["default", "mimetic", "impulse"]
INIT_LABELS = {"default": "Default", "mimetic": "Mimetic", "impulse": "Structured"}
COLORS = {"default": "#6B6B6B", "mimetic": "#1F77B4", "impulse": "#1B8A3A"}
DATASET_LABELS = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}
SUBSET_LABELS = {"full": "Full", "10pct": "10%", "25pct": "25%"}

FULL_RE = re.compile(
    r"(?P<dataset>cifar10|cifar100)_official_"
    r"(?P<init>default|mimetic|impulse)_seed(?P<seed>\d+)_"
    r"(?P<epochs>\d+)ep_bs(?P<batch_size>\d+)$",
    re.IGNORECASE,
)
LOW_RE = re.compile(
    r"(?P<dataset>cifar10|cifar100)_(?P<subset>10pct|25pct)_official_"
    r"(?P<init>default|mimetic|impulse)_seed(?P<seed>\d+)_"
    r"(?P<epochs>\d+)ep_bs(?P<batch_size>\d+)$",
    re.IGNORECASE,
)

# These runs were completed after the CSV logs used by the poster scripts were
# exported, so the report summary bars use the updated final checkpoint numbers.
FINAL_SCORE_OVERRIDES = {
    ("full", "cifar100", "impulse"): 73.4,
    ("10pct", "cifar10", "impulse"): 57.3,
    ("10pct", "cifar100", "impulse"): 25.5,
    ("25pct", "cifar100", "impulse"): 39.3,
}


def _read_run_summaries(base_dir: Path, pattern: re.Pattern, subset: str | None = None) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(base_dir.rglob("summary.csv")):
        match = pattern.match(summary_path.parent.name)
        if not match:
            continue
        meta = match.groupdict()
        run_subset = subset if subset is not None else meta["subset"].lower()
        df = pd.read_csv(summary_path)
        best = df.loc[df["eval_top1"].idxmax()]
        rows.append(
            {
                "setting": run_subset,
                "dataset": meta["dataset"].lower(),
                "init": meta["init"].lower(),
                "seed": int(meta["seed"]),
                "declared_epochs": int(meta["epochs"]),
                "batch_size": int(meta["batch_size"]),
                "epochs_available": len(df),
                "best_epoch": int(best["epoch"]) + 1,
                "best_top1": float(best["eval_top1"]),
                "summary_path": str(summary_path),
            }
        )
    if not rows:
        raise RuntimeError(f"No run summaries found under {base_dir}")
    return pd.DataFrame(rows)


def read_final_accuracy_table() -> pd.DataFrame:
    full = _read_run_summaries(FULL_RESULTS_DIR, FULL_RE, subset="full")
    full = full[full["seed"] == 0].copy()
    low = _read_run_summaries(LOW_RESULTS_DIR, LOW_RE)
    out = pd.concat([full, low], ignore_index=True)
    out["manual_override"] = False
    for (setting, dataset, init), value in FINAL_SCORE_OVERRIDES.items():
        mask = (
            (out["setting"] == setting)
            & (out["dataset"] == dataset)
            & (out["init"] == init)
        )
        out.loc[mask, "best_top1"] = value
        out.loc[mask, "manual_override"] = True
    return out.sort_values(["setting", "dataset", "init"]).reset_index(drop=True)


def make_combined_bar_chart(acc: pd.DataFrame, out_base: Path) -> None:
    group_order = [
        ("full", "cifar10"),
        ("full", "cifar100"),
        ("10pct", "cifar10"),
        ("10pct", "cifar100"),
        ("25pct", "cifar10"),
        ("25pct", "cifar100"),
    ]
    labels = [
        f"{SUBSET_LABELS[setting]}\n{DATASET_LABELS[dataset]}"
        for setting, dataset in group_order
    ]

    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    x = np.arange(len(group_order))
    width = 0.23
    for idx, init in enumerate(INIT_ORDER):
        values = []
        for setting, dataset in group_order:
            row = acc[
                (acc["setting"] == setting)
                & (acc["dataset"] == dataset)
                & (acc["init"] == init)
            ]
            values.append(float(row["best_top1"].iloc[0]))
        positions = x + (idx - 1) * width
        ax.bar(
            positions,
            values,
            width=width,
            color=COLORS[init],
            edgecolor="white",
            linewidth=0.8,
            label=INIT_LABELS[init],
        )
        for px, val in zip(positions, values):
            ax.text(px, val + 1.0, f"{val:.1f}", ha="center", va="bottom", fontsize=8.5)

    ax.axvline(1.5, color="#D0D4D8", lw=1.2)
    ax.axvline(3.5, color="#D0D4D8", lw=1.2)
    ax.text(0.5, 96.3, "Full-data", ha="center", va="top", fontsize=10.5, fontweight="bold")
    ax.text(2.5, 96.3, "10% training data", ha="center", va="top", fontsize=10.5, fontweight="bold")
    ax.text(4.5, 96.3, "25% training data", ha="center", va="top", fontsize=10.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Best Top-1 accuracy (%)", fontsize=10.5)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=320)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def read_training_curves() -> pd.DataFrame:
    curves = []

    for summary_path in sorted(FULL_RESULTS_DIR.rglob("summary.csv")):
        match = FULL_RE.match(summary_path.parent.name)
        if not match:
            continue
        meta = match.groupdict()
        if int(meta["seed"]) != 0:
            continue
        df = pd.read_csv(summary_path)
        df["setting"] = "full"
        df["dataset"] = meta["dataset"].lower()
        df["init"] = meta["init"].lower()
        df["seed"] = int(meta["seed"])
        df["epoch_count"] = df["epoch"] + 1
        curves.append(df)

    for summary_path in sorted(LOW_RESULTS_DIR.rglob("summary.csv")):
        match = LOW_RE.match(summary_path.parent.name)
        if not match:
            continue
        meta = match.groupdict()
        df = pd.read_csv(summary_path)
        df["setting"] = meta["subset"].lower()
        df["dataset"] = meta["dataset"].lower()
        df["init"] = meta["init"].lower()
        df["seed"] = int(meta["seed"])
        df["epoch_count"] = df["epoch"] + 1
        curves.append(df)

    if not curves:
        raise RuntimeError("No training curves found.")
    return pd.concat(curves, ignore_index=True)


def make_training_curve_grid(curves: pd.DataFrame, out_base: Path) -> None:
    panels = [
        ("full", "cifar10"),
        ("full", "cifar100"),
        ("10pct", "cifar10"),
        ("10pct", "cifar100"),
        ("25pct", "cifar10"),
        ("25pct", "cifar100"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.2), sharey=False)
    for ax, (setting, dataset) in zip(axes.flat, panels):
        sub = curves[(curves["setting"] == setting) & (curves["dataset"] == dataset)]
        for init in INIT_ORDER:
            run = sub[sub["init"] == init].sort_values("epoch_count")
            if run.empty:
                continue
            ax.plot(
                run["epoch_count"],
                run["eval_top1"],
                color=COLORS[init],
                lw=1.8,
                label=INIT_LABELS[init],
            )
        ax.set_title(
            f"{SUBSET_LABELS[setting]} {DATASET_LABELS[dataset]}",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Top-1 (%)", fontsize=9)
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), w_pad=1.8, h_pad=1.5)
    fig.savefig(out_base.with_suffix(".png"), dpi=320)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def make_seed_stability_curve(out_base: Path) -> None:
    all_epochs = pd.read_csv(PROCESSED_DIR / "all_epoch_metrics.csv")
    sub = all_epochs[(all_epochs["dataset"] == "cifar10") & (all_epochs["epoch_count"] <= 250)].copy()

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    for init in INIT_ORDER:
        run = sub[sub["init"] == init]
        grouped = run.groupby("epoch_count")["eval_top1"].agg(["mean", "std"]).reset_index()
        x = grouped["epoch_count"].to_numpy()
        mean = grouped["mean"].to_numpy()
        std = grouped["std"].fillna(0).to_numpy()
        ax.plot(x, mean, color=COLORS[init], lw=2.2, label=INIT_LABELS[init])
        ax.fill_between(x, mean - std, mean + std, color=COLORS[init], alpha=0.15, linewidth=0)

    ax.set_title("CIFAR-10 Seed Stability Across Training", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=10)
    ax.set_ylim(0, 96)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=320)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def make_seed_table_tex(out_path: Path) -> None:
    stability = pd.read_csv(PROCESSED_DIR / "cifar10_seed_stability.csv")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{CIFAR-10 seed stability under a common 250-epoch budget.}",
        r"\label{tab:seed-stability}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Initialization & Seed 0 & Seed 1 & Seed 2 & Mean $\pm$ Std \\",
        r"\midrule",
    ]
    for init in INIT_ORDER:
        row = stability[stability["init"] == init].iloc[0]
        lines.append(
            f"{INIT_LABELS[init]} & {row['0']:.2f} & {row['1']:.2f} & "
            f"{row['2']:.2f} & {row['mean']:.2f} $\\pm$ {row['std']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    acc = read_final_accuracy_table()
    acc.to_csv(FIG_DIR / "report_accuracy_summary.csv", index=False)
    make_combined_bar_chart(acc, FIG_DIR / "fig_results_accuracy_combined")

    curves = read_training_curves()
    curves.to_csv(FIG_DIR / "report_training_curves_source.csv", index=False)
    make_training_curve_grid(curves, FIG_DIR / "fig_training_curves_grid")
    make_seed_stability_curve(FIG_DIR / "fig_seed_stability_curve")
    make_seed_table_tex(REPORT_DIR / "seed_stability_table.tex")

    print(f"Wrote figures to {FIG_DIR}")
    print(acc[["setting", "dataset", "init", "best_top1", "manual_override"]].to_string(index=False))


if __name__ == "__main__":
    main()
