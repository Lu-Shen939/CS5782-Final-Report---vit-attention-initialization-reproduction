"""Training sweep script for the ViT attention-initialization reproduction.

This script merges the two Colab notebooks used during the project:

- ``Copy of Untitled0.ipynb``: full-data / seed-stability experiments.
- ``Untitled2.ipynb``: 10% and 25% low-data CIFAR experiments.

It is designed for Google Colab with an A100-class GPU and the official
Structured Initialization repository.  The default paths follow the Colab
notebooks, but every path can be overridden from the command line.
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import time
from pathlib import Path


INIT_ORDER = ("default", "mimetic", "impulse")
DATASETS = {
    "cifar10": ("CIFAR10", 10),
    "cifar100": ("CIFAR100", 100),
}


def stream_command(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n" + "=" * 90)
    print("Running:", " ".join(cmd))
    if cwd is not None:
        print("cwd:", cwd)
    print("=" * 90 + "\n", flush=True)

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    recent_lines: list[str] = []
    last_progress_time = time.time()
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        recent_lines.append(line)
        recent_lines = recent_lines[-8:]

        important = (
            "Train:" in line
            or "Test:" in line
            or "Epoch:" in line
            or "Acc@" in line
            or "loss" in line.lower()
            or "error" in line.lower()
            or "warning" in line.lower()
        )
        if important:
            print(line, flush=True)

        if time.time() - last_progress_time > 60:
            print("\n--- still running; recent output ---")
            for recent in recent_lines:
                print(recent)
            print("-----------------------------------\n", flush=True)
            last_progress_time = time.time()

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(cmd)}")


def bootstrap_official_repo(args: argparse.Namespace) -> None:
    if args.skip_bootstrap:
        return

    args.local_repo.parent.mkdir(parents=True, exist_ok=True)
    if args.local_repo.exists():
        shutil.rmtree(args.local_repo)

    stream_command(
        [
            "git",
            "clone",
            "--depth",
            "1",
            args.official_repo_url,
            str(args.local_repo),
        ]
    )
    stream_command(
        ["python", "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"],
        cwd=args.local_code,
    )
    stream_command(["python", "-m", "pip", "install", "-r", "requirements.txt"], cwd=args.local_code)
    stream_command(["python", "-m", "pip", "install", "-e", "."], cwd=args.local_code)


def common_train_args(args: argparse.Namespace, num_classes: int, init: str, seed: int) -> list[str]:
    return [
        "--input-size",
        "3",
        "224",
        "224",
        "--mean",
        "0.485",
        "0.456",
        "0.406",
        "--std",
        "0.229",
        "0.224",
        "0.225",
        "--seed",
        str(seed),
        "--num-classes",
        str(num_classes),
        "--model",
        "vit_tiny_patch16_224",
        "--model-kwargs",
        "img_size=224",
        "weight_init=skip",
        f"post_weight_init={init}",
        "--model-ema-decay",
        "0.9999",
        "-j",
        str(args.workers),
        "--lr",
        "2e-3",
        "--layer-decay",
        "1.0",
        "--warmup-lr",
        "1e-6",
        "--min-lr",
        "1e-6",
        "--weight-decay",
        "0.05",
        "--opt",
        "adamw",
        "--opt-eps",
        "1e-8",
        "--sched",
        "cosine",
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--amp",
        "--aa",
        "rand-m9-mstd0.5-inc1",
        "--cutmix",
        "1.0",
        "--mixup",
        "0.8",
        "--reprob",
        "0.25",
        "--smoothing",
        "0.1",
        "--drop",
        "0.0",
        "--color-jitter",
        "0.4",
        "--drop-path",
        "0.1",
        "--crop-pct",
        "0.875",
        "--pin-mem",
        "--checkpoint-hist",
        "1",
        "--recovery-interval",
        "0",
        "--output",
        str(args.local_results),
    ]


def run_torchvision_cifar(
    args: argparse.Namespace,
    dataset_key: str,
    init: str,
    seed: int,
    epochs: int,
    batch_size: int,
) -> None:
    dataset_name, num_classes = DATASETS[dataset_key]
    exp = f"{dataset_key}_official_{init}_seed{seed}_{epochs}ep_bs{batch_size}"
    cmd = [
        "python",
        "-u",
        "train.py",
        str(args.local_data),
        "--dataset",
        f"torch/{dataset_name}",
        "--dataset-download",
        *common_train_args(args, num_classes, init, seed),
        "-b",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--experiment",
        exp,
    ]
    stream_command(cmd, cwd=args.local_code)


def build_subset_imagefolder(
    args: argparse.Namespace,
    dataset_key: str,
    subset_frac: float,
    seed: int,
    rebuild: bool = False,
) -> Path:
    from torchvision.datasets import CIFAR10, CIFAR100

    dataset_key = dataset_key.lower()
    dataset_class = CIFAR10 if dataset_key == "cifar10" else CIFAR100
    pct_tag = f"{int(subset_frac * 100)}pct"
    raw_root = args.local_data / "raw_torchvision"
    out_root = args.local_data / f"{dataset_key}_{pct_tag}_seed{seed}_imagefolder"

    if out_root.exists() and not rebuild:
        print(f"Using existing subset: {out_root}")
        return out_root

    if out_root.exists():
        shutil.rmtree(out_root)

    train_set = dataset_class(root=str(raw_root), train=True, download=True)
    test_set = dataset_class(root=str(raw_root), train=False, download=True)
    classes = train_set.classes

    rng = random.Random(seed)
    by_class = {label: [] for label in range(len(classes))}
    for idx, label in enumerate(train_set.targets):
        by_class[label].append(idx)

    keep_indices = []
    for label, indices in by_class.items():
        rng.shuffle(indices)
        keep_count = max(1, int(len(indices) * subset_frac))
        keep_indices.extend(indices[:keep_count])

    def save_split(dataset, indices: list[int], split_name: str) -> None:
        for n, idx in enumerate(indices):
            img, label = dataset[idx]
            class_dir = out_root / split_name / classes[label]
            class_dir.mkdir(parents=True, exist_ok=True)
            img.save(class_dir / f"{idx:06d}.png")
            if n % 2000 == 0:
                print(f"{dataset_key} {pct_tag} {split_name}: saved {n}/{len(indices)}")

    save_split(train_set, keep_indices, "train")
    save_split(test_set, list(range(len(test_set))), "validation")
    print(f"Built {dataset_key} {pct_tag} subset at {out_root}")
    print(f"Train images: {len(keep_indices)} | Validation images: {len(test_set)}")
    return out_root


def run_imagefolder_cifar(
    args: argparse.Namespace,
    dataset_key: str,
    dataset_root: Path,
    subset_frac: float,
    init: str,
    seed: int,
    epochs: int,
    batch_size: int,
) -> None:
    _dataset_name, num_classes = DATASETS[dataset_key]
    pct_tag = f"{int(subset_frac * 100)}pct"
    exp = f"{dataset_key}_{pct_tag}_official_{init}_seed{seed}_{epochs}ep_bs{batch_size}"
    cmd = [
        "python",
        "-u",
        "train.py",
        str(dataset_root),
        "--train-split",
        "train",
        "--val-split",
        "validation",
        *common_train_args(args, num_classes, init, seed),
        "-b",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--experiment",
        exp,
    ]
    stream_command(cmd, cwd=args.local_code)


def backup_results(args: argparse.Namespace) -> None:
    args.drive_results_backup.mkdir(parents=True, exist_ok=True)
    if not args.local_results.exists():
        print(f"No local results to back up: {args.local_results}")
        return
    for item in args.local_results.iterdir():
        dst = args.drive_results_backup / item.name
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    print(f"Backed up results to {args.drive_results_backup}")


def run_full_seed0(args: argparse.Namespace) -> None:
    for dataset_key in ("cifar10", "cifar100"):
        epochs = args.cifar10_full_epochs if dataset_key == "cifar10" else args.cifar100_full_epochs
        for init in INIT_ORDER:
            run_torchvision_cifar(
                args,
                dataset_key=dataset_key,
                init=init,
                seed=0,
                epochs=epochs,
                batch_size=args.full_batch_size,
            )


def run_seed_stability(args: argparse.Namespace) -> None:
    for seed in args.seed_stability_seeds:
        for init in INIT_ORDER:
            run_torchvision_cifar(
                args,
                dataset_key="cifar10",
                init=init,
                seed=seed,
                epochs=args.seed_stability_epochs,
                batch_size=args.full_batch_size,
            )


def run_low_data(args: argparse.Namespace) -> None:
    for subset_frac in args.subset_fracs:
        for seed in args.low_data_seeds:
            roots = {
                dataset_key: build_subset_imagefolder(args, dataset_key, subset_frac, seed)
                for dataset_key in ("cifar10", "cifar100")
            }
            for dataset_key, root in roots.items():
                for init in INIT_ORDER:
                    run_imagefolder_cifar(
                        args,
                        dataset_key=dataset_key,
                        dataset_root=root,
                        subset_frac=subset_frac,
                        init=init,
                        seed=seed,
                        epochs=args.low_data_epochs,
                        batch_size=args.low_data_batch_size,
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ViT initialization reproduction sweeps.")
    parser.add_argument("--suite", choices=["full-seed0", "seed-stability", "low-data", "all"], default="all")
    parser.add_argument("--official-repo-url", default="https://github.com/osiriszjq/structured_initialization.git")
    parser.add_argument("--local-repo", type=Path, default=Path("/content/official_structured_initialization"))
    parser.add_argument("--local-code", type=Path, default=Path("/content/official_structured_initialization/pytorch-image-models-1.0.22"))
    parser.add_argument("--local-data", type=Path, default=Path("/content/official_structured_data"))
    parser.add_argument("--local-results", type=Path, default=Path("/content/official_structured_results"))
    parser.add_argument("--drive-results-backup", type=Path, default=Path("/content/drive/MyDrive/official_structured_results"))
    parser.add_argument("--skip-bootstrap", action="store_true", help="Use an existing official repo checkout.")
    parser.add_argument("--no-backup", action="store_true", help="Do not copy results to Google Drive at the end.")

    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--full-batch-size", type=int, default=1024)
    parser.add_argument("--low-data-batch-size", type=int, default=512)
    parser.add_argument("--cifar10-full-epochs", type=int, default=300)
    parser.add_argument("--cifar100-full-epochs", type=int, default=250)
    parser.add_argument("--seed-stability-epochs", type=int, default=250)
    parser.add_argument("--seed-stability-seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--low-data-epochs", type=int, default=150)
    parser.add_argument("--low-data-seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--subset-fracs", type=float, nargs="+", default=[0.10, 0.25])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.local_data.mkdir(parents=True, exist_ok=True)
    args.local_results.mkdir(parents=True, exist_ok=True)

    bootstrap_official_repo(args)

    if args.suite in ("full-seed0", "all"):
        run_full_seed0(args)
    if args.suite in ("seed-stability", "all"):
        run_seed_stability(args)
    if args.suite in ("low-data", "all"):
        run_low_data(args)

    if not args.no_backup:
        backup_results(args)


if __name__ == "__main__":
    main()
