"""Shared CVPR/MICCAI-style matplotlib styling and small drawing helpers.

Import `apply_style()` at the top of every figure script for a consistent,
publication-grade look (clean, high-contrast, readable labels).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# A restrained, high-contrast palette (colour-blind friendly).
PALETTE = {
    "image": "#2E5EAA",      # deep blue   -> histology / image modality
    "gene": "#C44E52",       # warm red    -> gene / omics modality
    "predict": "#55A868",    # green       -> predicted latents
    "target": "#8172B3",     # purple      -> stop-grad / EMA target
    "fusion": "#CC8400",     # amber       -> fusion / prediction head
    "neutral": "#4C4C4C",
    "light": "#EAEAF2",
    "accent": "#000000",
}


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.linewidth": 1.1,
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "lines.linewidth": 2.0,
    })


def rounded_box(ax, xy, w, h, text, facecolor, textcolor="white",
                fontsize=11, edgecolor=None, lw=1.5, alpha=1.0, fontweight="bold"):
    """Draw a rounded rectangle with centred text; returns the patch."""
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch(
        (xy[0], xy[1]), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw, edgecolor=edgecolor or facecolor,
        facecolor=facecolor, alpha=alpha, mutation_aspect=1.0, zorder=2,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            color=textcolor, fontsize=fontsize, fontweight=fontweight,
            zorder=3, wrap=True)
    return box


def arrow(ax, start, end, color="#333333", lw=2.0, style="-|>", ls="-",
          rad=0.0):
    from matplotlib.patches import FancyArrowPatch

    a = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=18,
        linewidth=lw, color=color, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", zorder=1,
    )
    ax.add_patch(a)
    return a


def save(fig, path):
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    # Also write a PNG companion for quick previews / slides.
    if path.endswith(".pdf"):
        fig.savefig(path[:-4] + ".png")
    print(f"[viz] saved {path}", flush=True)
