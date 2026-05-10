import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from datasets import build_dataloaders
from initializers import initialize_model
from utils import (
    accuracy_top1,
    describe_device,
    ensure_run_dir,
    plot_accuracy_curve,
    save_json,
    set_seed,
    write_metrics_csv,
)
from vit import create_model, model_config_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Train small ViT attention-initialization baselines.")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "svhn"])
    parser.add_argument(
        "--init",
        default="trunc_normal",
        choices=["default", "trunc_normal", "mimetic", "structured_imp3", "structured_imp5"],
    )
    parser.add_argument("--model", default="vit_tiny", choices=["vit_micro", "vit_mini", "vit_tiny"])
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--subset-frac", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--structured-steps", type=int, default=200)
    parser.add_argument("--structured-lr", type=float, default=5e-2)
    parser.add_argument("--mimetic-noise", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=0)
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_seen = 0
    progress = tqdm(loader, desc=f"epoch {epoch} train", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch = images.shape[0]
        total_seen += batch
        total_loss += float(loss.detach().cpu()) * batch
        total_acc += accuracy_top1(logits.detach(), targets) * batch
        progress.set_postfix(loss=total_loss / total_seen, acc=total_acc / total_seen)
    return total_loss / total_seen, total_acc / total_seen


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_seen = 0
    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        batch = images.shape[0]
        total_seen += batch
        total_loss += float(loss.detach().cpu()) * batch
        total_acc += accuracy_top1(logits, targets) * batch
    return total_loss / total_seen, total_acc / total_seen


def save_checkpoint(path: Path, model, optimizer, scheduler, args, epoch, best_acc):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "args": vars(args),
            "model_config": model_config_dict(model),
            "epoch": epoch,
            "best_acc": best_acc,
        },
        path,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    run_name = args.run_name
    if run_name is None:
        subset = f"_subset{args.subset_frac:g}" if args.subset_frac < 1.0 else ""
        run_name = f"{args.dataset}_{args.model}_{args.init}{subset}_seed{args.seed}"
    run_dir = ensure_run_dir(args.output_dir, run_name)
    save_json({"args": vars(args), "device": describe_device()}, run_dir / "config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    train_loader, test_loader, num_classes = build_dataloaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_fraction=args.subset_frac,
        seed=args.seed,
    )
    model = create_model(
        args.model,
        num_classes=num_classes,
        img_size=args.img_size,
        patch_size=args.patch_size,
        dropout=args.dropout,
    ).to(device)

    init_start = time.time()
    initialize_model(
        model,
        args.init,
        structured_steps=args.structured_steps,
        structured_lr=args.structured_lr,
        mimetic_noise=args.mimetic_noise,
    )
    print(f"Initialization completed in {(time.time() - init_start) / 60:.2f} minutes.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    metrics = []
    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 4),
            "lr": optimizer.param_groups[0]["lr"],
        }
        metrics.append(row)
        write_metrics_csv(metrics, run_dir / "metrics.csv")
        plot_accuracy_curve(metrics, run_dir / "accuracy_curve.png")
        print(
            f"epoch {epoch:03d}: train_acc={train_acc:.2f} "
            f"val_acc={val_acc:.2f} val_loss={val_loss:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, args, epoch, best_acc)
        if args.save_every and epoch % args.save_every == 0:
            save_checkpoint(
                run_dir / f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, args, epoch, best_acc
            )
        save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, args, epoch, best_acc)

    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run_name", "best_val_acc", "epochs", "init", "dataset"])
        writer.writeheader()
        writer.writerow(
            {
                "run_name": run_name,
                "best_val_acc": round(best_acc, 4),
                "epochs": args.epochs,
                "init": args.init,
                "dataset": args.dataset,
            }
        )
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
