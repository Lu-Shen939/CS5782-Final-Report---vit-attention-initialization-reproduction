import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


INIT_ORDER = ["default", "mimetic", "impulse"]
INIT_LABELS = {
    "default": "Default",
    "mimetic": "Mimetic",
    "impulse": "Structured",
}


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, dim=192, hidden_dim=768):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim=192, num_heads=3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, return_attn=False):
        bsz, num_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, dim)
        x = self.proj(x)
        if return_attn:
            return x, attn
        return x


class Block(nn.Module):
    def __init__(self, dim=192, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, return_attn=False):
        if return_attn:
            attn_out, attn = self.attn(self.norm1(x), return_attn=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class OfficialViTTiny(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 192))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, 192))
        self.patch_embed = PatchEmbed()
        self.blocks = nn.ModuleList([Block() for _ in range(12)])
        self.norm = nn.LayerNorm(192, eps=1e-6)
        self.fc_norm = nn.Identity()
        self.head_drop = nn.Identity()
        self.head = nn.Linear(192, num_classes)

    def forward_features(self, x, layers=(0, 5, 11)):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        attn_maps = {}
        for idx, block in enumerate(self.blocks):
            if idx in layers:
                x, attn = block(x, return_attn=True)
                attn_maps[idx] = attn.detach().cpu()
            else:
                x = block(x)
        x = self.norm(x)
        return x, attn_maps

    def forward(self, x, layers=(0, 5, 11)):
        features, attn_maps = self.forward_features(x, layers=layers)
        pooled = features[:, 0]
        return self.head(pooled), attn_maps


def find_run(results_dir: Path, dataset: str, init: str, seed: int):
    pattern = re.compile(
        rf"{dataset}_official_{init}_seed{seed}_\d+ep_bs\d+$",
        re.IGNORECASE,
    )
    matches = [p for p in results_dir.rglob("*") if p.is_dir() and pattern.search(p.name)]
    if not matches:
        raise FileNotFoundError(f"No run found for dataset={dataset}, init={init}, seed={seed}")
    matches = sorted(matches, key=lambda p: len(str(p)))
    return matches[0]


def load_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    num_classes = int(state_dict["head.weight"].shape[0])
    model = OfficialViTTiny(num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in missing if not k.startswith("head.")]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"State load mismatch for {checkpoint_path}: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model, checkpoint


def normalize(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr - arr.min()
    denom = arr.max()
    return arr / denom if denom > 0 else arr


def get_attention(model, input_tensor, layers):
    with torch.no_grad():
        _, maps = model(input_tensor, layers=layers)
    return maps


def make_attention_figure(results_dir: Path, output_path: Path, dataset: str, seed: int):
    layers = (0, 5, 11)
    layer_labels = ["Layer 1", "Layer 6", "Layer 12"]
    input_tensor = torch.zeros(1, 3, 224, 224)
    grid = 14
    center = (grid // 2) * grid + (grid // 2)

    all_maps = {}
    metrics = {}
    for init in INIT_ORDER:
        run_dir = find_run(results_dir, dataset, init, seed)
        model, checkpoint = load_model(run_dir / "model_best.pth.tar")
        all_maps[init] = get_attention(model, input_tensor, layers)
        metrics[init] = float(checkpoint.get("metric", np.nan))

    fig = plt.figure(figsize=(9.2, 5.2))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=3,
        left=0.08,
        right=0.98,
        bottom=0.12,
        top=0.84,
        wspace=0.18,
        hspace=0.22,
    )
    fig.suptitle(
        "Trained Checkpoint Attention Maps",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.90,
        f"{dataset.upper()} seed {seed}, model_best checkpoint, mean-image input",
        ha="center",
        fontsize=9,
        color="#333333",
    )

    for col, init in enumerate(INIT_ORDER):
        fig.text(
            0.205 + col * 0.295,
            0.855,
            f"{INIT_LABELS[init]} ({metrics[init]:.2f}%)",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
        for row, layer in enumerate(layers):
            ax = fig.add_subplot(gs[row, col])
            attn = all_maps[init][layer][0].mean(dim=0)
            patch_attn = attn[1:, 1:]
            if row < 2:
                matrix = normalize(patch_attn.numpy())
                ax.imshow(matrix, cmap="magma", aspect="equal", interpolation="nearest")
            else:
                center_map = normalize(patch_attn[center].reshape(grid, grid).numpy())
                ax.imshow(center_map, cmap="magma", aspect="equal", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                label = layer_labels[row]
                suffix = "\npatch-patch" if row < 2 else "\ncenter query"
                ax.set_ylabel(label + suffix, fontsize=9)

    fig.text(
        0.5,
        0.04,
        "Attention is averaged over heads; CLS token is excluded. Panels are normalized independently for visual contrast.",
        ha="center",
        fontsize=8.3,
        color="#333333",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def make_layer_head_grid_figure(results_dir: Path, output_path: Path, dataset: str, seed: int):
    layers = (0, 3, 7, 11)
    layer_labels = ["L1", "L4", "L8", "L12"]
    heads = (0, 1)
    input_tensor = torch.zeros(1, 3, 224, 224)

    all_maps = {}
    metrics = {}
    for init in INIT_ORDER:
        run_dir = find_run(results_dir, dataset, init, seed)
        model, checkpoint = load_model(run_dir / "model_best.pth.tar")
        all_maps[init] = get_attention(model, input_tensor, layers)
        metrics[init] = float(checkpoint.get("metric", np.nan))

    panel_rows = []
    for row, layer in enumerate(layers):
        row_panels = []
        for init_idx, init in enumerate(INIT_ORDER):
            attn = all_maps[init][layer][0]
            for head_idx, head in enumerate(heads):
                patch_attn = attn[head, 1:, 1:]
                row_panels.append(normalize(patch_attn.numpy()))
        panel_rows.append(np.concatenate(row_panels, axis=1))
    canvas = np.concatenate(panel_rows, axis=0)

    fig = plt.figure(figsize=(9.2, 5.2))
    ax = fig.add_axes([0.075, 0.075, 0.89, 0.80])
    ax.imshow(canvas, cmap="magma", aspect="equal", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ncols = len(INIT_ORDER) * len(heads)
    nrows = len(layers)
    for init_idx, init in enumerate(INIT_ORDER):
        x = (init_idx * len(heads) + len(heads) / 2) / ncols
        ax.text(
            x,
            1.105,
            f"{INIT_LABELS[init]} ({metrics[init]:.2f}%)",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9.8,
            fontweight="bold",
            clip_on=False,
        )
    for init_idx, _ in enumerate(INIT_ORDER):
        for head_idx, head in enumerate(heads):
            col = init_idx * len(heads) + head_idx
            x = (col + 0.5) / ncols
            ax.text(
                x,
                1.025,
                f"H{head + 1}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8.6,
                clip_on=False,
            )
    for row, label in enumerate(layer_labels):
        y = 1 - (row + 0.5) / nrows
        ax.text(
            -0.025,
            y,
            label,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=9.5,
            clip_on=False,
        )

    fig.text(
        0.5,
        0.018,
        "Patch-to-patch attention, CLS token excluded. Each panel is normalized independently for visual contrast.",
        ha="center",
        fontsize=7.5,
        color="#333333",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=280)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Draw attention maps from official checkpoints.")
    parser.add_argument(
        "--results-dir",
        default="/Users/lushen/Desktop/results/seed0/official_structured_results",
    )
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="vit_attention_init_project/poster_assets/poster_result_panels/03_attention_map_trained_checkpoint.png",
    )
    parser.add_argument(
        "--style",
        choices=["summary", "layers_heads"],
        default="summary",
        help="summary averages heads over three views; layers_heads draws L1/L4/L8/L12 for H1/H2.",
    )
    args = parser.parse_args()
    if args.style == "summary":
        make_attention_figure(
            results_dir=Path(args.results_dir),
            output_path=Path(args.output),
            dataset=args.dataset.lower(),
            seed=args.seed,
        )
    else:
        make_layer_head_grid_figure(
            results_dir=Path(args.results_dir),
            output_path=Path(args.output),
            dataset=args.dataset.lower(),
            seed=args.seed,
        )
    print(f"Wrote trained checkpoint attention map to: {args.output}")


if __name__ == "__main__":
    main()
