#!/usr/bin/env python
"""Conceptual figure: image-gene alignment.

Same-patient image/gene embeddings are pulled together; different-patient pairs
are pushed apart (optional InfoNCE alignment). If a real alignment embedding file
exists (outputs/embeddings/pathgene_alignment_embeddings.npz) a small real
similarity-distribution panel is added; otherwise only the schematic is drawn.

Output: outputs/figures/image_gene_alignment.pdf
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utils.visualization import PALETTE, apply_style, arrow, save  # noqa: E402
from utils.io import OUT_ROOT as _OUT  # namespaced output root

OUT = os.path.join(_OUT, "figures/image_gene_alignment.pdf")
EMB = os.path.join(_OUT, "embeddings/pathgene_alignment_embeddings.npz")


def schematic(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    rng = np.random.RandomState(1)
    # three patients, each with an image (circle) and gene (square) token nearby
    centers = [(2.5, 7.0), (7.0, 7.2), (4.8, 3.0)]
    cols = [PALETTE["image"], PALETTE["gene"], PALETTE["predict"]]
    for (cx, cy), c in zip(centers, cols):
        ax.scatter([cx - 0.25], [cy], s=320, marker="o", color=c,
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.scatter([cx + 0.25], [cy], s=300, marker="s", color=c,
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.plot([cx - 0.25, cx + 0.25], [cy, cy], color=c, lw=2.0, zorder=2)
        ax.text(cx, cy + 0.7, "patient", ha="center", fontsize=8.5,
                color=PALETTE["neutral"])
    # push-apart arrows between different patients
    arrow(ax, (3.4, 6.7), (6.1, 7.0), color="#999999", style="<|-|>")
    ax.text(4.8, 6.2, "different patients\npushed apart", ha="center",
            fontsize=8.5, color="#777777")
    ax.scatter([], [], marker="o", color="#444444", label="image embedding")
    ax.scatter([], [], marker="s", color="#444444", label="gene embedding")
    ax.legend(loc="lower center", ncol=2, fontsize=9)
    ax.set_title("Image-gene contrastive alignment", fontsize=12)


def real_panel(ax):
    d = np.load(EMB, allow_pickle=True)
    if "same_sim" not in d:
        return False
    same = d["same_sim"]
    diff = d["diff_sim"]
    bins = np.linspace(min(diff.min(), same.min()), max(diff.max(), same.max()), 30)
    ax.hist(diff, bins=bins, alpha=0.6, color=PALETTE["neutral"],
            label="different patient", density=True)
    ax.hist(same, bins=bins, alpha=0.8, color=PALETTE["image"],
            label="same patient", density=True)
    ax.set_xlabel("image-gene cosine similarity")
    ax.set_ylabel("density")
    ax.set_title("Measured similarity distribution", fontsize=12)
    ax.legend()
    return True


def main():
    apply_style()
    has_real = os.path.exists(EMB)
    if has_real:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        schematic(axes[0])
        if not real_panel(axes[1]):
            axes[1].axis("off")
            axes[1].text(0.5, 0.5, "[TODO: run run_alignment_analysis.py\n"
                                   "to populate similarity distribution]",
                         ha="center", va="center", color="red")
    else:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        schematic(ax)
    save(fig, OUT)


if __name__ == "__main__":
    main()
