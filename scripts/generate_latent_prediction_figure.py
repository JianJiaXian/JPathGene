#!/usr/bin/env python
"""Conceptual figure: latent biological prediction target construction.

Shows that, instead of reconstructing raw high-dimensional gene expression, the
histology latent predicts a compact PCA / pathway / learned gene latent target,
with a stop-gradient (or EMA) target branch.

Output: outputs/figures/latent_biological_prediction.pdf
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utils.visualization import PALETTE, apply_style, arrow, rounded_box, save  # noqa: E402

OUT = "outputs/figures/latent_biological_prediction.pdf"


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # raw gene expression heat-strip (high-dim)
    rng = np.random.RandomState(0)
    strip = rng.rand(1, 40)
    ax.imshow(strip, extent=[0.3, 4.3, 5.6, 6.4], aspect="auto", cmap="magma",
              zorder=2)
    ax.text(2.3, 6.7, "Raw gene expression / mutation profile  (high-dim)",
            ha="center", fontsize=10, fontweight="bold")
    ax.text(2.3, 5.2, r"$G \sim 10^3$–$10^4$ genes", ha="center", fontsize=9,
            color=PALETTE["neutral"])

    # crossed-out reconstruction path
    rounded_box(ax, (0.5, 2.6), 3.6, 1.0, "Raw reconstruction\n(decode all genes)",
                "#BDBDBD", textcolor="#555555", fontsize=9.5)
    ax.plot([0.7, 4.0], [3.55, 2.65], color="#C0392B", lw=2.5, zorder=5)
    ax.plot([0.7, 4.0], [2.65, 3.55], color="#C0392B", lw=2.5, zorder=5)
    ax.text(2.3, 2.25, "avoided", ha="center", fontsize=9, color="#C0392B",
            style="italic")

    # target construction
    rounded_box(ax, (5.4, 5.2), 3.6, 1.4,
                "Gene latent target\n(PCA / pathway / learned)", PALETTE["target"],
                fontsize=10)
    arrow(ax, (4.3, 6.0), (5.4, 6.0), color=PALETTE["gene"])
    ax.text(4.85, 6.3, "reduce", ha="center", fontsize=8.5,
            color=PALETTE["neutral"])

    # compact latent strip
    strip2 = rng.rand(1, 8)
    ax.imshow(strip2, extent=[5.8, 8.6, 4.1, 4.7], aspect="auto", cmap="viridis",
              zorder=2)
    ax.text(7.2, 3.8, r"compact biological program  $z_{gene}^{target}\in\mathbb{R}^{d}$",
            ha="center", fontsize=9, color=PALETTE["neutral"])

    # stop-grad branch
    rounded_box(ax, (5.4, 1.4), 3.6, 1.0, "stop-gradient / EMA", PALETTE["target"],
                fontsize=9.5)
    arrow(ax, (7.2, 5.2), (7.2, 2.4), color=PALETTE["target"], ls="--")

    # histology predicts target
    rounded_box(ax, (10.2, 5.4), 3.4, 1.3, "Histology latent\n$z_{img}$",
                PALETTE["image"], fontsize=11)
    rounded_box(ax, (10.2, 2.7), 3.4, 1.3, "Image$\\rightarrow$Gene\nPredictor",
                PALETTE["predict"], fontsize=10)
    arrow(ax, (11.9, 5.4), (11.9, 4.0), color=PALETTE["image"])
    arrow(ax, (10.2, 3.35), (9.0, 4.4), color=PALETTE["predict"], rad=0.2)
    ax.text(11.9, 1.9,
            "cosine / Smooth-L1\nto stop-grad target", ha="center", fontsize=8.8,
            color=PALETTE["neutral"], style="italic")

    ax.text(7.5, 7.55,
            "Predict genetic programs in latent space, not raw expression",
            ha="center", fontsize=13, fontweight="bold")
    save(fig, OUT)


if __name__ == "__main__":
    main()
