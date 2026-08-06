#!/usr/bin/env python
"""Conceptual figure: the conditional latent-diffusion JEPA predictor.

Output: outputs/figures/diffusion_predictor.pdf
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utils.visualization import PALETTE, apply_style, arrow, rounded_box, save  # noqa: E402

OUT = "outputs/figures/diffusion_predictor.pdf"


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.text(8, 7.5, "Conditional latent-diffusion predictor  "
            r"$p(t_{gene}\,|\,z_{img})$",
            ha="center", fontsize=13.5, fontweight="bold")

    # condition
    rounded_box(ax, (0.3, 3.4), 2.3, 1.2, r"histology" "\n" r"latent $z_{img}$",
                PALETTE["image"], fontsize=10.5)

    # reverse diffusion chain: noise -> ... -> sample
    xs = [3.6, 6.2, 8.8, 11.4]
    labels = [r"$x_T\sim\mathcal{N}(0,I)$", r"$x_{t}$", r"$x_{t-1}$",
              r"$\hat{t}_{gene}$"]
    rng = np.random.RandomState(0)
    for i, (x, lab) in enumerate(zip(xs, labels)):
        col = PALETTE["target"] if i == len(xs) - 1 else PALETTE["neutral"]
        strip = rng.rand(1, 8) * (0.2 if i == len(xs) - 1 else 1.0)
        ax.imshow(strip, extent=[x, x + 1.9, 4.3, 4.9], aspect="auto",
                  cmap="viridis" if i == len(xs) - 1 else "Greys", zorder=2)
        ax.text(x + 0.95, 3.95, lab, ha="center", fontsize=10, color=col)
        if i < len(xs) - 1:
            arrow(ax, (x + 1.9, 4.6), (xs[i + 1], 4.6), color=PALETTE["neutral"])

    # denoiser conditioned on z_img + time
    rounded_box(ax, (5.4, 5.9), 4.0, 1.0,
                r"FiLM denoiser $\epsilon_\theta(x_t,\,\tau,\,z_{img})$",
                PALETTE["predict"], fontsize=10.5)
    arrow(ax, (2.6, 4.0), (7.4, 5.9), color=PALETTE["image"], rad=-0.15)
    arrow(ax, (7.4, 5.9), (7.4, 4.95), color=PALETTE["predict"], ls="--")
    ax.text(9.9, 6.4, "denoising loss\n= JEPA objective", ha="left", fontsize=9,
            color=PALETTE["neutral"], style="italic")

    # N samples -> mean + uncertainty
    rounded_box(ax, (13.0, 5.0), 2.7, 1.1, r"posterior mean $\bar{t}_{gene}$",
                PALETTE["target"], fontsize=9.5)
    rounded_box(ax, (13.0, 2.9), 2.7, 1.1, r"uncertainty $u_{gene}$",
                PALETTE["fusion"], fontsize=10)
    arrow(ax, (13.3, 4.6), (13.6, 5.0), color=PALETTE["target"])
    arrow(ax, (13.3, 4.6), (13.6, 4.0), color=PALETTE["fusion"])
    ax.text(12.4, 4.6, r"$\times N$" "\nDDIM\nsamples", ha="center", fontsize=8.5,
            color=PALETTE["neutral"])

    fig.text(0.5, 0.02,
             "Deterministic JEPA = single-step limit; diffusion models the full "
             "one-to-many image$\\rightarrow$gene distribution.",
             ha="center", fontsize=10, style="italic", color=PALETTE["gene"])
    save(fig, OUT)


if __name__ == "__main__":
    main()
