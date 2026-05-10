import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_RE = re.compile(
    r"(?P<dataset>cifar10|cifar100)_official_"
    r"(?P<init>default|mimetic|impulse)_seed(?P<seed>\d+)_"
    r"(?P<epochs>\d+)ep_bs(?P<batch_size>\d+)",
    re.IGNORECASE,
)

INIT_ORDER = ["default", "mimetic", "impulse"]
INIT_LABELS = {
    "default": "Default",
    "mimetic": "Mimetic",
    "impulse": "Structured",
}
DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
}

PAPER_TOP1 = {
    ("cifar10", "default"): 92.29,
    ("cifar10", "mimetic"): 93.50,
    ("cifar10", "impulse"): 94.67,
    ("cifar100", "default"): 71.67,
    ("cifar100", "mimetic"): 75.16,
    ("cifar100", "impulse"): 77.02,
}


def parse_run(csv_path: Path):
    run_name = csv_path.parent.name
    lowered = run_name.lower()
    if any(token in lowered for token in ["10pct", "25pct", "subset", "lowdata", "low_data"]):
        return None
    match = RUN_RE.search(run_name)
    if not match:
        return None
    meta = match.groupdict()
    return {
        "run_name": run_name,
        "dataset": meta["dataset"].lower(),
        "init": meta["init"].lower(),
        "seed": int(meta["seed"]),
        "declared_epochs": int(meta["epochs"]),
        "batch_size": int(meta["batch_size"]),
        "summary_path": str(csv_path),
    }


def read_runs(results_dir: Path):
    frames = []
    skipped = []
    seen = set()
    for csv_path in sorted(results_dir.rglob("*.csv")):
        if not csv_path.name.startswith("summary"):
            continue
        meta = parse_run(csv_path)
        if meta is None:
            skipped.append(str(csv_path))
            continue
        key = (meta["dataset"], meta["init"], meta["seed"], meta["declared_epochs"])
        if key in seen:
            skipped.append(str(csv_path))
            continue
        seen.add(key)

        df = pd.read_csv(csv_path)
        required = {"epoch", "train_loss", "eval_loss", "eval_top1", "eval_top5", "lr"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

        for key_name, value in meta.items():
            df[key_name] = value
        df["epoch"] = df["epoch"].astype(int)
        df["epoch_count"] = np.arange(1, len(df) + 1)
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No official full-data summary CSV files found under {results_dir}")
    all_epochs = pd.concat(frames, ignore_index=True)
    return all_epochs, skipped


def first_budget_metrics(df: pd.DataFrame, budget: int):
    sub = df.sort_values("epoch_count").head(budget)
    if sub.empty:
        return {}
    best_idx = sub["eval_top1"].idxmax()
    best = sub.loc[best_idx]
    final = sub.iloc[-1]
    return {
        f"best_top1_{budget}ep": float(best["eval_top1"]),
        f"best_epoch_{budget}ep": int(best["epoch_count"]),
        f"final_top1_{budget}ep": float(final["eval_top1"]),
        f"final_loss_{budget}ep": float(final["eval_loss"]),
        f"mean_top1_{budget}ep": float(sub["eval_top1"].mean()),
    }


def threshold_epoch(df: pd.DataFrame, threshold: float):
    hit = df[df["eval_top1"] >= threshold].sort_values("epoch_count")
    if hit.empty:
        return np.nan
    return int(hit.iloc[0]["epoch_count"])


def summarize_runs(all_epochs: pd.DataFrame):
    rows = []
    budgets = [50, 100, 150, 250, 300]
    thresholds = [65, 70, 72, 85, 90, 92]

    group_cols = ["dataset", "init", "seed", "run_name", "declared_epochs", "batch_size", "summary_path"]
    for keys, df in all_epochs.groupby(group_cols, sort=False):
        meta = dict(zip(group_cols, keys))
        df = df.sort_values("epoch_count").reset_index(drop=True)
        best_idx = df["eval_top1"].idxmax()
        best = df.loc[best_idx]
        final = df.iloc[-1]
        row = {
            **meta,
            "epochs_available": int(len(df)),
            "best_top1_full": float(best["eval_top1"]),
            "best_epoch_full": int(best["epoch_count"]),
            "final_top1_full": float(final["eval_top1"]),
            "final_loss_full": float(final["eval_loss"]),
            "paper_top1": PAPER_TOP1.get((meta["dataset"], meta["init"]), np.nan),
        }
        for budget in budgets:
            if len(df) >= budget:
                row.update(first_budget_metrics(df, budget))
        for threshold in thresholds:
            row[f"epochs_to_{threshold:g}"] = threshold_epoch(df, threshold)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["init"] = pd.Categorical(summary["init"], INIT_ORDER, ordered=True)
    summary = summary.sort_values(["dataset", "init", "seed"]).reset_index(drop=True)
    summary["init"] = summary["init"].astype(str)
    return summary


def aggregate_for_poster(run_summary: pd.DataFrame):
    rows = []
    for dataset, ds_df in run_summary.groupby("dataset", sort=False):
        common_budget = int(ds_df["epochs_available"].min())
        metric = f"best_top1_{common_budget}ep"
        final_metric = f"final_top1_{common_budget}ep"
        mean_metric = f"mean_top1_{common_budget}ep"
        default_mean = ds_df[ds_df["init"] == "default"][metric].mean()
        mimetic_mean = ds_df[ds_df["init"] == "mimetic"][metric].mean()

        for init in INIT_ORDER:
            sub = ds_df[ds_df["init"] == init]
            if sub.empty:
                continue
            values = sub[metric].dropna()
            finals = sub[final_metric].dropna()
            means = sub[mean_metric].dropna()
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row = {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset),
                "init": init,
                "init_label": INIT_LABELS.get(init, init),
                "n_seeds": int(len(values)),
                "common_epoch_budget": common_budget,
                "best_top1_mean": float(values.mean()),
                "best_top1_std": std,
                "best_top1_display": f"{values.mean():.2f} +/- {std:.2f}" if len(values) > 1 else f"{values.mean():.2f}",
                "final_top1_mean": float(finals.mean()) if len(finals) else np.nan,
                "mean_curve_top1": float(means.mean()) if len(means) else np.nan,
                "delta_vs_default": float(values.mean() - default_mean),
                "delta_vs_mimetic": float(values.mean() - mimetic_mean),
                "paper_top1": PAPER_TOP1.get((dataset, init), np.nan),
            }
            if not np.isnan(row["paper_top1"]):
                row["paper_gap"] = row["best_top1_mean"] - row["paper_top1"]
            rows.append(row)

    poster = pd.DataFrame(rows)
    poster["init"] = pd.Categorical(poster["init"], INIT_ORDER, ordered=True)
    poster = poster.sort_values(["dataset", "init"]).reset_index(drop=True)
    poster["init"] = poster["init"].astype(str)
    return poster


def aggregate_cifar10_stability(run_summary: pd.DataFrame):
    sub = run_summary[run_summary["dataset"] == "cifar10"].copy()
    if sub.empty:
        return pd.DataFrame()
    common_budget = int(sub["epochs_available"].min())
    metric = f"best_top1_{common_budget}ep"
    pivot = sub.pivot_table(index="init", columns="seed", values=metric, aggfunc="first")
    pivot = pivot.reindex(INIT_ORDER)
    seed_cols = list(pivot.columns)
    pivot["mean"] = pivot[seed_cols].mean(axis=1)
    pivot["std"] = pivot[seed_cols].std(axis=1, ddof=1)
    pivot["common_epoch_budget"] = common_budget
    return pivot.reset_index()


def make_learning_curve_plot(all_epochs: pd.DataFrame, dataset: str, out_path: Path):
    ds = all_epochs[all_epochs["dataset"] == dataset].copy()
    if ds.empty:
        return
    common_budget = int(ds.groupby(["init", "seed"])["epoch_count"].max().min())
    ds = ds[ds["epoch_count"] <= common_budget]

    colors = {
        "default": "#606060",
        "mimetic": "#2C7FB8",
        "impulse": "#2CA25F",
    }
    plt.figure(figsize=(7.2, 4.2))
    for init in INIT_ORDER:
        sub = ds[ds["init"] == init]
        if sub.empty:
            continue
        grouped = sub.groupby("epoch_count")["eval_top1"]
        mean = grouped.mean()
        std = grouped.std(ddof=1).fillna(0.0)
        x = mean.index.to_numpy()
        y = mean.to_numpy()
        s = std.to_numpy()
        label = INIT_LABELS.get(init, init)
        plt.plot(x, y, label=label, linewidth=2.2, color=colors[init])
        if sub["seed"].nunique() > 1:
            plt.fill_between(x, y - s, y + s, alpha=0.16, color=colors[init], linewidth=0)

    plt.title(f"{DATASET_LABELS.get(dataset, dataset)} Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Top-1 accuracy (%)")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def make_accuracy_bar_plot(poster: pd.DataFrame, out_path: Path):
    if poster.empty:
        return
    datasets = [d for d in ["cifar10", "cifar100"] if d in set(poster["dataset"])]
    x = np.arange(len(datasets))
    width = 0.24
    colors = {
        "default": "#606060",
        "mimetic": "#2C7FB8",
        "impulse": "#2CA25F",
    }

    plt.figure(figsize=(7.4, 4.3))
    for i, init in enumerate(INIT_ORDER):
        sub = poster[poster["init"] == init].set_index("dataset")
        values = [sub.loc[d, "best_top1_mean"] if d in sub.index else np.nan for d in datasets]
        errors = [sub.loc[d, "best_top1_std"] if d in sub.index else 0.0 for d in datasets]
        offset = (i - 1) * width
        plt.bar(
            x + offset,
            values,
            width,
            yerr=errors,
            capsize=4,
            label=INIT_LABELS[init],
            color=colors[init],
            edgecolor="white",
            linewidth=0.8,
        )
        for xi, value in zip(x + offset, values):
            if np.isfinite(value):
                plt.text(xi, value + 0.35, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, [DATASET_LABELS[d] for d in datasets])
    plt.ylabel("Best Top-1 accuracy (%)")
    plt.title("Official Full-Data Reproduction Results")
    plt.grid(True, axis="y", alpha=0.22)
    plt.legend(frameon=False, ncol=3, loc="upper left")
    ymax = max(poster["best_top1_mean"].max() + 5, 80)
    plt.ylim(0, min(100, ymax))
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def make_accuracy_table_image(poster: pd.DataFrame, out_path: Path):
    if poster.empty:
        return
    display = poster.copy()
    display["Delta"] = display["delta_vs_default"].map(lambda x: f"{x:+.2f}")
    display["Paper"] = display["paper_top1"].map(lambda x: "" if np.isnan(x) else f"{x:.2f}")
    table = display[
        [
            "dataset_label",
            "init_label",
            "n_seeds",
            "common_epoch_budget",
            "best_top1_display",
            "Delta",
            "Paper",
        ]
    ]
    table.columns = ["Dataset", "Init", "Seeds", "Epochs", "Best Top-1", "Delta", "Paper"]

    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    ax.axis("off")
    ax.set_title(
        "Official Full-Data Reproduction Summary",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    mpl_table = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(8.7)
    mpl_table.scale(1, 1.32)

    for (row, col), cell in mpl_table.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor("#D6DCE2")
        if row == 0:
            cell.set_facecolor("#17324D")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F3F6F8")
        else:
            cell.set_facecolor("white")

    fig.text(
        0.5,
        0.02,
        "CIFAR-10 reports mean +/- std over seeds 0, 1, 2 at the common 250-epoch budget; CIFAR-100 is seed 0.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def make_markdown_table(poster: pd.DataFrame):
    lines = [
        "| Dataset | Init | Seeds | Epoch budget | Best Top-1 | Delta vs Default | Paper Top-1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in poster.iterrows():
        paper = "" if np.isnan(row["paper_top1"]) else f"{row['paper_top1']:.2f}"
        lines.append(
            "| {dataset} | {init} | {seeds} | {budget} | {best} | {delta:+.2f} | {paper} |".format(
                dataset=row["dataset_label"],
                init=row["init_label"],
                seeds=int(row["n_seeds"]),
                budget=int(row["common_epoch_budget"]),
                best=row["best_top1_display"],
                delta=float(row["delta_vs_default"]),
                paper=paper,
            )
        )
    return "\n".join(lines) + "\n"


def write_takeaways(poster: pd.DataFrame, out_path: Path):
    lines = []
    for dataset in ["cifar10", "cifar100"]:
        ds = poster[poster["dataset"] == dataset]
        if ds.empty:
            continue
        default = float(ds[ds["init"] == "default"]["best_top1_mean"].iloc[0])
        mimetic = float(ds[ds["init"] == "mimetic"]["best_top1_mean"].iloc[0])
        impulse = float(ds[ds["init"] == "impulse"]["best_top1_mean"].iloc[0])
        lines.append(
            f"{DATASET_LABELS[dataset]}: Mimetic improves over Default by {mimetic - default:+.2f} pp; "
            f"Structured improves over Default by {impulse - default:+.2f} pp."
        )
        lines.append(
            f"{DATASET_LABELS[dataset]}: Structured vs Mimetic difference is {impulse - mimetic:+.2f} pp."
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize official ViT initialization runs.")
    parser.add_argument("--results-dir", default="/Users/lushen/Desktop/results")
    parser.add_argument(
        "--output-dir",
        default="vit_attention_init_project/poster_assets/processed_official_results",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_epochs, skipped = read_runs(results_dir)
    run_summary = summarize_runs(all_epochs)
    poster = aggregate_for_poster(run_summary)
    cifar10_stability = aggregate_cifar10_stability(run_summary)

    all_epochs.to_csv(output_dir / "all_epoch_metrics.csv", index=False)
    run_summary.to_csv(output_dir / "run_summary.csv", index=False)
    poster.to_csv(output_dir / "poster_accuracy_table.csv", index=False)
    cifar10_stability.to_csv(output_dir / "cifar10_seed_stability.csv", index=False)
    (output_dir / "poster_accuracy_table.md").write_text(make_markdown_table(poster), encoding="utf-8")
    write_takeaways(poster, output_dir / "poster_key_takeaways.txt")
    (output_dir / "skipped_files.txt").write_text("\n".join(skipped) + "\n", encoding="utf-8")

    make_learning_curve_plot(all_epochs, "cifar10", output_dir / "poster_cifar10_learning_curve.png")
    make_learning_curve_plot(all_epochs, "cifar100", output_dir / "poster_cifar100_learning_curve.png")
    make_accuracy_bar_plot(poster, output_dir / "poster_accuracy_bar.png")
    make_accuracy_table_image(poster, output_dir / "poster_accuracy_table.png")

    print(f"Wrote processed outputs to: {output_dir}")
    print()
    print(make_markdown_table(poster))
    if skipped:
        print("Skipped files:")
        for path in skipped:
            print(f"  {path}")


if __name__ == "__main__":
    main()
