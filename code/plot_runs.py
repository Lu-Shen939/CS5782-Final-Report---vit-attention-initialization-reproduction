import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot validation curves for multiple runs.")
    parser.add_argument("runs", nargs="+", help="Run directories containing metrics.csv")
    parser.add_argument("--output", default="results/comparison_accuracy.png")
    return parser.parse_args()


def main():
    args = parse_args()
    plt.figure(figsize=(7, 4))
    for run in args.runs:
        run_dir = Path(run)
        csv_path = run_dir / "metrics.csv"
        if not csv_path.exists():
            print(f"Skipping {run_dir}: no metrics.csv")
            continue
        df = pd.read_csv(csv_path)
        plt.plot(df["epoch"], df["val_acc"], label=run_dir.name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation top-1 accuracy (%)")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
