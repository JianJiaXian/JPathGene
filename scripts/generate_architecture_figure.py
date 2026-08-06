#!/usr/bin/env python
"""Generate the JPathGene architecture figure (conceptual, CVPR/MICCAI style).

Output: outputs/figures/pathgene_jepa_architecture.pdf (+ .png)
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt  # noqa: E402

from utils.visualization import PALETTE, apply_style, arrow, rounded_box, save  # noqa: E402

OUT = "outputs/figures/pathgene_jepa_architecture.pdf"


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.set_xlim(0, 16.0)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # ---------- inputs ----------
    rounded_box(ax, (0.2, 6.1), 2.7, 1.3,
                "Histopathology\nimage features", PALETTE["image"], fontsize=11)
    rounded_box(ax, (0.2, 1.5), 2.7, 1.3,
                "Genetic / omics\nprofile", PALETTE["gene"], fontsize=11)

    # ---------- encoders ----------
    rounded_box(ax, (3.6, 6.1), 2.4, 1.3, "Image\nEncoder", PALETTE["image"])
    rounded_box(ax, (3.6, 1.5), 2.4, 1.3, "Gene\nEncoder", PALETTE["gene"])
    arrow(ax, (2.9, 6.75), (3.6, 6.75))
    arrow(ax, (2.9, 2.15), (3.6, 2.15))

    # ---------- latents ----------
    rounded_box(ax, (6.5, 6.25), 1.7, 1.0, r"$z_{img}$", PALETTE["image"],
                fontsize=14)
    rounded_box(ax, (6.5, 1.65), 1.7, 1.0, r"$z_{gene}$", PALETTE["gene"],
                fontsize=14)
    arrow(ax, (6.0, 6.75), (6.5, 6.75))
    arrow(ax, (6.0, 2.15), (6.5, 2.15))

    # ---------- predictors (cross-modal, diffusion) ----------
    rounded_box(ax, (9.0, 5.0), 2.6, 1.1, "Image$\\rightarrow$Gene\nDiffusion pred.",
                PALETTE["predict"], fontsize=9.5)
    rounded_box(ax, (9.0, 2.9), 2.6, 1.1, "Gene$\\rightarrow$Image\nDiffusion pred.",
                PALETTE["predict"], fontsize=9.5)
    # z_img feeds img->gene predictor; z_gene feeds gene->img predictor
    arrow(ax, (8.2, 6.6), (9.0, 5.7), color=PALETTE["image"], rad=-0.2)
    arrow(ax, (8.2, 2.3), (9.0, 3.3), color=PALETTE["gene"], rad=0.2)

    # ---------- predicted vs target latents ----------
    rounded_box(ax, (12.0, 5.05), 2.0, 1.0, "pred.\ngene latent",
                PALETTE["predict"], fontsize=9.5)
    rounded_box(ax, (12.0, 2.95), 2.0, 1.0, "pred.\nimage latent",
                PALETTE["predict"], fontsize=9.5)
    arrow(ax, (11.6, 5.55), (12.0, 5.55))
    arrow(ax, (11.6, 3.45), (12.0, 3.45))

    # stop-grad / EMA targets (dashed). Short labels; the dashed arrow from the
    # encoder already conveys "target derived from encoder".
    rounded_box(ax, (12.3, 7.25), 2.9, 1.0,
                "EMA / stop-grad\ngene target", PALETTE["target"], fontsize=9.5)
    rounded_box(ax, (12.3, 0.45), 2.9, 1.0,
                "EMA / stop-grad\nimage target", PALETTE["target"], fontsize=9.5)
    arrow(ax, (13.75, 7.25), (13.0, 6.05), color=PALETTE["target"], style="-|>",
          ls="--")
    arrow(ax, (13.75, 1.45), (13.0, 2.95), color=PALETTE["target"], style="-|>",
          ls="--")
    # JEPA loss markers
    ax.text(13.2, 6.45, "JEPA loss", ha="center", fontsize=8.5,
            color=PALETTE["neutral"], style="italic")
    ax.text(13.2, 2.55, "JEPA loss", ha="center", fontsize=8.5,
            color=PALETTE["neutral"], style="italic")

    # ---------- fusion + prediction ----------
    rounded_box(ax, (9.0, 0.4), 2.2, 0.9, "image$\\leftrightarrow$gene\nInfoNCE",
                PALETTE["neutral"], fontsize=8.5, alpha=0.85)
    arrow(ax, (7.35, 6.25), (7.35, 4.0), color=PALETTE["image"], rad=0.0)

    rounded_box(ax, (6.2, 3.95), 2.2, 1.0, "CoRE Fusion\n(anchor+gate)",
                PALETTE["fusion"], fontsize=9.0)
    # feed latents + predicted latents into fusion
    arrow(ax, (7.0, 6.25), (7.1, 4.9), color=PALETTE["image"])
    arrow(ax, (7.4, 2.65), (7.4, 4.0), color=PALETTE["gene"])
    rounded_box(ax, (3.4, 3.9), 2.4, 1.05, "Cancer\nprediction", PALETTE["accent"],
                fontsize=10)
    arrow(ax, (6.3, 4.45), (5.8, 4.45), color=PALETTE["fusion"])

    # ---------- title banner ----------
    ax.text(8.0, 8.75, "JPathGene: cross-modal latent predictive learning",
            ha="center", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.015,
             "Latent biological prediction, not raw gene reconstruction.",
             ha="center", fontsize=11, style="italic", color=PALETTE["gene"])

    save(fig, OUT)


if __name__ == "__main__":
    main()
