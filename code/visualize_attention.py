import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from datasets import build_dataloaders
from vit import create_model_from_config


def parse_args():
    parser = argparse.ArgumentParser(description="Save averaged attention heatmaps from a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "svhn"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5, 11])
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def save_heatmap(matrix, path: Path, title: str) -> None:
    plt.figure(figsize=(5, 4.5))
    plt.imshow(matrix, cmap="magma", aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Key token")
    plt.ylabel("Query token")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


@torch.no_grad()
def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = create_model_from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    _, test_loader, _ = build_dataloaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_fraction=1.0,
        seed=0,
    )
    images, _ = next(iter(test_loader))
    images = images.to(device)
    _, attn_maps = model(images, return_attn=True)

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    depth = len(attn_maps)
    for layer in args.layers:
        if layer < 0 or layer >= depth:
            continue
        avg = attn_maps[layer].mean(dim=(0, 1)).detach().cpu()
        save_heatmap(avg, output_dir / f"attention_layer_{layer + 1:02d}.png", f"Layer {layer + 1}")

        grid_size = int(avg.shape[0] ** 0.5)
        center_token = (grid_size // 2) * grid_size + (grid_size // 2)
        center_map = avg[center_token].reshape(grid_size, grid_size)
        save_heatmap(
            center_map,
            output_dir / f"center_token_attention_layer_{layer + 1:02d}.png",
            f"Layer {layer + 1}, center query",
        )
    print(f"Saved attention visualizations to {output_dir}")


if __name__ == "__main__":
    main()
