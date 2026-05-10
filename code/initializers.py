import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


def apply_trunc_normal_initialization(model: nn.Module, std: float = 0.02) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def _set_linear_to_identity(linear: nn.Linear, sign: float = 1.0, noise_std: float = 0.0) -> None:
    if linear.weight.shape[0] != linear.weight.shape[1]:
        raise ValueError("Mimetic identity initialization expects square linear layers.")
    dim = linear.weight.shape[0]
    eye = torch.eye(dim, device=linear.weight.device, dtype=linear.weight.dtype) * sign
    if noise_std > 0:
        eye = eye + torch.randn_like(eye) * noise_std
    with torch.no_grad():
        linear.weight.copy_(eye)
        if linear.bias is not None:
            linear.bias.zero_()


def apply_mimetic_initialization(model: nn.Module, noise_std: float = 0.0) -> None:
    """Closed-form, practical version of mimetic attention initialization.

    This sets Q and K to matching identities and V/Proj to opposite signs,
    approximating the QK=I and VO=-I pattern used by mimetic initialization.
    """
    for block in getattr(model, "blocks", []):
        attn = block.attn
        _set_linear_to_identity(attn.q, sign=1.0, noise_std=noise_std)
        _set_linear_to_identity(attn.k, sign=1.0, noise_std=noise_std)
        _set_linear_to_identity(attn.v, sign=1.0, noise_std=noise_std)
        _set_linear_to_identity(attn.proj, sign=-1.0, noise_std=noise_std)


def _impulse_targets(
    grid_size: int,
    kernel_size: int,
    num_heads: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build local impulse attention targets of shape [H, N, N]."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    radius = kernel_size // 2
    offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    offset_ids = torch.randint(0, len(offsets), (num_heads,), device=device)
    chosen_offsets = torch.tensor([offsets[i] for i in offset_ids.tolist()], device=device)

    num_tokens = grid_size * grid_size
    target = torch.full((num_heads, num_tokens, num_tokens), eps / num_tokens, device=device, dtype=dtype)
    for head, (dy, dx) in enumerate(chosen_offsets.tolist()):
        for y in range(grid_size):
            for x in range(grid_size):
                src = y * grid_size + x
                # Circular padding keeps every row equally normalized and easy to visualize.
                yy = (y + dy) % grid_size
                xx = (x + dx) % grid_size
                dst = yy * grid_size + xx
                target[head, src, dst] = 1.0 - eps + eps / num_tokens
    return target, chosen_offsets


def _qk_attention_from_pos(block: nn.Module, pos_embed: torch.Tensor) -> torch.Tensor:
    attn = block.attn
    num_heads = attn.num_heads
    head_dim = attn.head_dim
    q = attn.q(pos_embed).reshape(pos_embed.shape[0], num_heads, head_dim).transpose(0, 1)
    k = attn.k(pos_embed).reshape(pos_embed.shape[0], num_heads, head_dim).transpose(0, 1)
    return (q @ k.transpose(-2, -1)) * attn.scale


def _optimize_block_qk(
    block: nn.Module,
    pos_embed: torch.Tensor,
    target: torch.Tensor,
    steps: int,
    lr: float,
) -> float:
    params = [block.attn.q.weight, block.attn.k.weight]
    if block.attn.q.bias is not None:
        params.append(block.attn.q.bias)
    if block.attn.k.bias is not None:
        params.append(block.attn.k.bias)
    optimizer = torch.optim.Adam(params, lr=lr)
    final_loss = math.nan
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = _qk_attention_from_pos(block, pos_embed)
        loss = F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    return final_loss


def apply_structured_impulse_initialization(
    model: nn.Module,
    kernel_size: int,
    steps: int = 200,
    lr: float = 5e-2,
    eps: float = 1e-3,
    show_progress: bool = True,
) -> None:
    """Initialize Q/K so positional-encoding attention matches local impulse filters.

    This is a compact reproduction-oriented implementation of the structured
    initialization idea. It uses the fixed sinusoidal positional encoding as the
    pseudo input and optimizes Q/K in each layer to approximate an impulse-like
    convolutional attention target.
    """
    apply_trunc_normal_initialization(model)
    grid_size = model.patch_embed.grid_size
    pos_embed = model.pos_embed.squeeze(0).to(next(model.parameters()).device)
    pos_embed = pos_embed.to(dtype=next(model.parameters()).dtype)

    iterator = enumerate(model.blocks)
    if show_progress:
        iterator = tqdm(list(iterator), desc=f"structured Imp-{kernel_size} init")

    for _, block in iterator:
        target, _ = _impulse_targets(
            grid_size=grid_size,
            kernel_size=kernel_size,
            num_heads=block.attn.num_heads,
            device=pos_embed.device,
            dtype=pos_embed.dtype,
            eps=eps,
        )
        _optimize_block_qk(block, pos_embed, target, steps=steps, lr=lr)


def initialize_model(
    model: nn.Module,
    method: str,
    structured_steps: int = 200,
    structured_lr: float = 5e-2,
    mimetic_noise: float = 0.0,
) -> None:
    method = method.lower()
    if method in {"default", "trunc_normal"}:
        apply_trunc_normal_initialization(model)
    elif method == "mimetic":
        apply_trunc_normal_initialization(model)
        apply_mimetic_initialization(model, noise_std=mimetic_noise)
    elif method == "structured_imp3":
        apply_structured_impulse_initialization(
            model, kernel_size=3, steps=structured_steps, lr=structured_lr
        )
    elif method == "structured_imp5":
        apply_structured_impulse_initialization(
            model, kernel_size=5, steps=structured_steps, lr=structured_lr
        )
    else:
        raise ValueError(
            "Unknown initialization method. Choose default, trunc_normal, mimetic, "
            "structured_imp3, or structured_imp5."
        )
