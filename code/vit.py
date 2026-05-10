import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ViTConfig:
    img_size: int = 32
    patch_size: int = 4
    in_chans: int = 3
    num_classes: int = 10
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.0


def _sincos_1d(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("1D sinusoidal embedding dimension must be even.")
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=positions.device)
    omega = 1.0 / (10000 ** (omega / (embed_dim // 2)))
    out = positions.float().unsqueeze(1) * omega.unsqueeze(0)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Return [grid_size * grid_size, embed_dim] fixed 2D sinusoidal embeddings."""
    if embed_dim % 4 != 0:
        raise ValueError("2D sinusoidal embedding dimension must be divisible by 4.")
    y, x = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float32),
        torch.arange(grid_size, dtype=torch.float32),
        indexing="ij",
    )
    y = y.reshape(-1)
    x = x.reshape(-1)
    half = embed_dim // 2
    return torch.cat([_sincos_1d(half, y), _sincos_1d(half, x)], dim=1)


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size.")
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        bsz, num_tokens, dim = x.shape
        q = self.q(x).reshape(bsz, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(bsz, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(bsz, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn_logits.softmax(dim=-1)
        out = self.attn_drop(attn) @ v
        out = out.transpose(1, 2).reshape(bsz, num_tokens, dim)
        out = self.proj_drop(self.proj(out))
        if return_attn:
            return out, attn
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if return_attn:
            attn_out, attn = self.attn(self.norm1(x), return_attn=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerSmall(nn.Module):
    """Small ViT with average pooling and fixed sinusoidal positional encoding."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(
            config.img_size, config.patch_size, config.in_chans, config.embed_dim
        )
        grid_size = self.patch_embed.grid_size
        pos_embed = get_2d_sincos_pos_embed(config.embed_dim, grid_size).unsqueeze(0)
        self.register_buffer("pos_embed", pos_embed, persistent=False)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.embed_dim, config.num_heads, config.mlp_ratio, config.dropout
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.patch_embed.proj.weight, mode="fan_out")
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        x = self.patch_embed(x)
        x = x + self.pos_embed.to(dtype=x.dtype, device=x.device)
        attn_maps = []
        for block in self.blocks:
            if return_attn:
                x, attn = block(x, return_attn=True)
                attn_maps.append(attn)
            else:
                x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        logits = self.head(x)
        if return_attn:
            return logits, attn_maps
        return logits


MODEL_PRESETS = {
    "vit_micro": dict(embed_dim=96, depth=4, num_heads=3),
    "vit_mini": dict(embed_dim=128, depth=6, num_heads=4),
    "vit_tiny": dict(embed_dim=192, depth=12, num_heads=3),
}


def create_model(
    model_name: str,
    num_classes: int,
    img_size: int = 32,
    patch_size: int = 4,
    dropout: float = 0.0,
) -> VisionTransformerSmall:
    if model_name not in MODEL_PRESETS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {sorted(MODEL_PRESETS)}")
    cfg = ViTConfig(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dropout=dropout,
        **MODEL_PRESETS[model_name],
    )
    return VisionTransformerSmall(cfg)


def model_config_dict(model: VisionTransformerSmall) -> dict:
    cfg = model.config
    return {
        "img_size": cfg.img_size,
        "patch_size": cfg.patch_size,
        "in_chans": cfg.in_chans,
        "num_classes": cfg.num_classes,
        "embed_dim": cfg.embed_dim,
        "depth": cfg.depth,
        "num_heads": cfg.num_heads,
        "mlp_ratio": cfg.mlp_ratio,
        "dropout": cfg.dropout,
    }


def create_model_from_config(config: dict) -> VisionTransformerSmall:
    return VisionTransformerSmall(ViTConfig(**config))
