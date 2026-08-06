#!/usr/bin/env python
"""Predicted-vs-true gene latent prediction plot.

Generated ONLY from real arrays produced by run_gene_prediction_analysis.py
(outputs/embeddings/pathgene_gene_prediction_arrays.npz). If those arrays are
absent, a clearly-labelled TODO template is written instead, so no numbers are
ever fabricated.

Output: outputs/figures/gene_prediction_analysis.pdf  (real)
        outputs/figures/gene_prediction_analysis_template.pdf  (placeholder)
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utils.visualization import PALETTE, apply_style, save  # noqa: E402
from utils.io import OUT_ROOT as _OUT  # namespaced output root

ARR = os.path.join(_OUT, "embeddings/pathgene_gene_prediction_arrays.npz")
REAL = os.path.join(_OUT, "figures/gene_prediction_analysis.pdf")
TEMPLATE = os.path.join(_OUT, "figures/gene_prediction_analysis_template.pdf")


def template():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax in axes:
        ax.text(0.5, 0.5, "[TODO: real data]\nrun run_gene_prediction_analysis.py",
                ha="center", va="center", color="red", fontsize=12,
                transform=ax.transAxes)
    axes[0].set_title("Predicted vs. true gene latent (per-dim)")
    axes[0].set_xlabel("true latent value")
    axes[0].set_ylabel("predicted latent value")
    axes[1].set_title("Row-wise cosine similarity")
    axes[1].set_xlabel("cosine(pred, true)")
    axes[1].set_ylabel("count")
    save(fig, TEMPLATE)


def real():
    d = np.load(ARR, allow_pickle=True)
    pred, true = d["pred"], d["true"]
    cos = d["rowwise_cosine"]
    pear = d["per_dim_pearson"]
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # scatter of a few latent dims (subsample for clarity)
    k = min(6, pred.shape[1])
    idx = np.linspace(0, pred.shape[1] - 1, k).astype(int)
    t = true[:, idx].ravel()
    p = pred[:, idx].ravel()
    axes[0].scatter(t, p, s=8, alpha=0.4, color=PALETTE["predict"])
    lim = [min(t.min(), p.min()), max(t.max(), p.max())]
    axes[0].plot(lim, lim, "--", color=PALETTE["neutral"], lw=1.5)
    axes[0].set_xlabel("true gene latent")
    axes[0].set_ylabel("predicted gene latent")
    axes[0].set_title(f"Predicted vs. true ({k} dims)")

    axes[1].hist(cos, bins=25, color=PALETTE["image"], alpha=0.85)
    axes[1].axvline(float(cos.mean()), color=PALETTE["gene"], lw=2,
                    label=f"mean={cos.mean():.3f}")
    axes[1].set_xlabel("row-wise cosine(pred, true)")
    axes[1].set_ylabel("# patients")
    axes[1].set_title("Prediction cosine similarity")
    axes[1].legend()

    axes[2].hist(pear, bins=25, color=PALETTE["target"], alpha=0.85)
    axes[2].axvline(float(np.nanmean(pear)), color=PALETTE["gene"], lw=2,
                    label=f"mean={np.nanmean(pear):.3f}")
    axes[2].set_xlabel("per-dim Pearson r")
    axes[2].set_ylabel("# latent dims")
    axes[2].set_title("Per-dimension correlation")
    axes[2].legend()
    save(fig, REAL)


def main():
    if os.path.exists(ARR):
        real()
    else:
        print(f"[gene-pred-plot] {ARR} not found -> writing TODO template only")
    template()


if __name__ == "__main__":
    main()
