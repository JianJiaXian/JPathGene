#!/usr/bin/env python
"""One-page CVPR/MICCAI-style overview figure (the paper's teaser / Figure 1).

Conceptual only; safe to generate immediately.
Output: outputs/figures/cvpr_style_overview.pdf
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt  # noqa: E402

from utils.visualization import PALETTE, apply_style, arrow, rounded_box, save  # noqa: E402

OUT = "outputs/figures/cvpr_style_overview.pdf"


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(8, 6.5, "Same tumor biology, two modalities — aligned in latent space",
            ha="center", fontsize=13.5, fontweight="bold")

    # left: two modalities of the SAME tumor
    rounded_box(ax, (0.4, 4.0), 3.0, 1.6, "Histopathology\n(WSI features)",
                PALETTE["image"], fontsize=10.5)
    rounded_box(ax, (0.4, 1.0), 3.0, 1.6, "Genetic / omics\nprofile",
                PALETTE["gene"], fontsize=10.5)
    ax.text(1.9, 3.35, "one patient / tumor", ha="center", fontsize=8.5,
            color=PALETTE["neutral"], style="italic")

    # middle: JEPA cross prediction
    rounded_box(ax, (5.0, 4.0), 3.4, 1.6,
                "predict gene latent\nfrom histology", PALETTE["predict"],
                fontsize=10)
    rounded_box(ax, (5.0, 1.0), 3.4, 1.6,
                "predict histology latent\nfrom genes", PALETTE["predict"],
                fontsize=10)
    arrow(ax, (3.4, 4.8), (5.0, 4.8), color=PALETTE["image"])
    arrow(ax, (3.4, 1.8), (5.0, 1.8), color=PALETTE["gene"])
    # cross links
    arrow(ax, (6.7, 4.0), (6.7, 2.6), color=PALETTE["target"], ls="--",
          style="<|-|>")
    ax.text(7.5, 3.3, "latent\ncorrespondence", ha="center", fontsize=8.5,
            color=PALETTE["neutral"])

    # right: outcomes
    rounded_box(ax, (10.0, 4.4), 5.4, 1.2,
                "Cancer prediction (AUC / F1)", PALETTE["fusion"], fontsize=10.5)
    rounded_box(ax, (10.0, 2.8), 5.4, 1.2,
                "Image$\\leftrightarrow$gene retrieval (Recall@k)",
                PALETTE["neutral"], fontsize=10)
    rounded_box(ax, (10.0, 1.2), 5.4, 1.2,
                "Interpretable gene-program prediction", PALETTE["target"],
                fontsize=10)
    arrow(ax, (8.4, 4.8), (10.0, 5.0), color=PALETTE["predict"])
    arrow(ax, (8.4, 1.8), (10.0, 3.4), color=PALETTE["predict"], rad=-0.1)
    arrow(ax, (8.4, 1.8), (10.0, 1.8), color=PALETTE["predict"])

    fig.text(0.5, 0.02,
             "JPathGene — predict genetic programs in latent space, "
             "not raw gene expression.",
             ha="center", fontsize=10.5, style="italic", color=PALETTE["gene"])
    save(fig, OUT)


if __name__ == "__main__":
    main()
