from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "poster_assets" / "poster_result_panels"


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", transparent=True, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT / f"{stem}.png", dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def _make_mimetic_icon() -> None:
    rng = np.random.default_rng(7)
    n = 13
    rr, cc = np.indices((n, n))
    diag = np.exp(-((rr - cc) ** 2) / 1.75)
    off_diag = 0.45 * np.exp(-((rr - cc - 3) ** 2) / 3.0)
    anti = 0.22 * np.exp(-((rr + cc - (n - 1)) ** 2) / 5.5)
    texture = 0.16 * rng.random((n, n))
    values = 0.08 + 0.72 * diag + off_diag + anti + texture
    values = np.clip(values, 0, 1)

    cmap = LinearSegmentedColormap.from_list(
        "poster_mimetic",
        ["#071d3a", "#075985", "#0ea5a4", "#22c55e", "#fde047"],
    )

    fig, ax = plt.subplots(figsize=(1.5, 1.5))

    for r in range(n):
        for c in range(n):
            color = to_hex(cmap(values[r, c]))
            rect = Rectangle((c - 0.5, r - 0.5), 1.0, 1.0, facecolor=color, edgecolor="none")
            ax.add_patch(rect)

    for i in range(n + 1):
        ax.plot([-0.5, n - 0.5], [i - 0.5, i - 0.5], color=(1, 1, 1, 0.15), lw=0.35)
        ax.plot([i - 0.5, i - 0.5], [-0.5, n - 0.5], color=(1, 1, 1, 0.15), lw=0.35)

    # A thin highlight on the main diagonal makes the mimetic cue visible at poster scale.
    ax.plot(np.arange(n), np.arange(n), color="#fef3c7", lw=1.6, alpha=0.9)
    ax.set_axis_off()
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    fig.subplots_adjust(0, 0, 1, 1)
    _save(fig, "top_icon_mimetic_attention")


def _make_structured_icon() -> None:
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.set_axis_off()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")

    active = {(1, 1): "#22a447", (2, 1): "#86d88b"}
    for r in range(3):
        for c in range(3):
            y = 2 - r
            fill = active.get((r, c), "#f8fafc")
            edge = "#64748b"
            rect = Rectangle((c + 0.04, y + 0.04), 0.92, 0.92, facecolor=fill, edgecolor=edge, linewidth=1.6)
            ax.add_patch(rect)

    # Small center dot emphasizes that the icon represents a programmed local impulse.
    ax.scatter([1.5], [1.5], s=70, color="#147a35", zorder=5)
    fig.subplots_adjust(0, 0, 1, 1)
    _save(fig, "top_icon_structured_impulse")


def main() -> None:
    _make_mimetic_icon()
    _make_structured_icon()
    print(f"Wrote icons to {OUT}")


if __name__ == "__main__":
    main()
