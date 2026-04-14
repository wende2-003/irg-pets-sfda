#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "ink": "#222222",
    "muted": "#4c4c4c",
    "box_fill": "#d9d9d9",
    "box_edge": "#a0a0a0",
    "dash_fill": "#dcdcdc",
    "dash_edge": "#9d9d9d",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a minimalist PPT-like PETS-SR mechanism diagram."
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("imgs/pets_sr_mechanism"),
        help="Output prefix, saved as .png/.pdf/.svg",
    )
    parser.add_argument("--alpha", type=float, default=0.9, help="EMA retention factor")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up cycles")
    parser.add_argument("--period", type=int, default=1, help="ST refresh period")
    parser.add_argument("--dpi", type=int, default=320, help="PNG DPI")
    return parser.parse_args()


def setup_style():
    mpl.rcParams.update(
        {
            "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "axes.facecolor": "#ececec",
            "figure.facecolor": "#ececec",
            "savefig.facecolor": "#ececec",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def add_gradient_background(ax):
    x = np.linspace(0, 1, 700)
    y = np.linspace(0, 1, 500)
    xx, yy = np.meshgrid(x, y)
    cx, cy = 0.5, 0.52
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    grad = 0.84 + 0.14 * (1 - np.clip(r / 0.75, 0, 1))
    ax.imshow(
        grad,
        extent=(0, 100, 0, 72),
        origin="lower",
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="bicubic",
        zorder=0,
    )


def box(ax, x, y, w, h, text, dashed=False, fs=14):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.35,rounding_size=1.0",
        facecolor=COLORS["dash_fill"] if dashed else COLORS["box_fill"],
        edgecolor=COLORS["dash_edge"] if dashed else COLORS["box_edge"],
        linewidth=1.2,
        linestyle=(0, (4, 3)) if dashed else "-",
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, fontsize=fs, color=COLORS["ink"], ha="center", va="center", zorder=4)


def arrow(ax, start, end, lw=1.6):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=lw,
            color=COLORS["ink"],
            zorder=5,
        )
    )


def draw(ax, alpha, warmup, period):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 72)
    ax.axis("off")
    add_gradient_background(ax)

    ax.text(
        50,
        64,
        "PETS-SR Consensus Pseudo-Labeling",
        fontsize=23,
        color=COLORS["ink"],
        ha="center",
        va="center",
        fontweight="normal",
        zorder=6,
    )

    box(ax, 16.0, 48.0, 27.0, 7.0, "Static Teacher", fs=18)
    box(ax, 57.0, 48.0, 27.0, 7.0, "Dynamic Teacher", fs=18)

    ax.text(29.5, 43.3, "Predictions", fontsize=15, color=COLORS["ink"], ha="center", va="center", zorder=6)
    ax.text(70.5, 43.3, "Predictions", fontsize=15, color=COLORS["ink"], ha="center", va="center", zorder=6)

    arrow(ax, (29.5, 47.6), (29.5, 40.7))
    arrow(ax, (70.5, 47.6), (70.5, 40.7))

    box(ax, 17.5, 31.6, 65.0, 6.2, "Agreement", dashed=True, fs=19)
    arrow(ax, (50.0, 31.6), (50.0, 26.8))

    box(ax, 40.2, 20.3, 19.6, 6.1, "Student", fs=18)
    arrow(ax, (50.0, 20.3), (50.0, 14.1))

    ax.text(
        50.0,
        9.7,
        "Consensus Pseudo-Labels",
        fontsize=19,
        color=COLORS["ink"],
        ha="center",
        va="center",
        zorder=6,
    )

    foot = (
        rf"Cycle-end update:  $\theta_{{DT}}={alpha:.1f}\theta_{{DT}}+(1-{alpha:.1f})\theta_{{S}}$,   "
        rf"ST $\leftarrow$ S after warm-up ({warmup}) every {period} cycle, no ST$\rightarrow$S write-back."
    )
    ax.text(50.0, 2.1, foot, fontsize=10.7, color=COLORS["muted"], ha="center", va="center", zorder=6)


def save(fig, output_prefix, dpi):
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        out = output_prefix.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = dpi
        fig.savefig(out, **kwargs)
        print(f"Saved {out}")


def main():
    args = parse_args()
    setup_style()
    fig, ax = plt.subplots(figsize=(12.2, 7.4))
    draw(ax, args.alpha, args.warmup, args.period)
    save(fig, args.output_prefix, args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
