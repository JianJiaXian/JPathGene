#!/usr/bin/env python
"""UMAP/t-SNE of image and gene embeddings (real data only).

Reads outputs/embeddings/pathgene_alignment_embeddings.npz. Uses UMAP if
available, otherwise falls back to scikit-learn t-SNE / PCA. If the embedding
file is absent, writes a clearly-labelled TODO template.

Output: outputs/figures/embedding_umap.pdf
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utils.visualization import PALETTE, apply_style, save  # noqa: E402
from utils.io import OUT_ROOT as _OUT  # namespaced output root

EMB = os.path.join(_OUT, "embeddings/pathgene_alignment_embeddings.npz")
OUT = os.path.join(_OUT, "figures/embedding_umap.pdf")


def project(X):
    try:
        import umap  # type: ignore
        return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0
                         ).fit_transform(X), "UMAP"
    except Exception:
        pass
    try:
        from sklearn.manifold import TSNE
        perp = max(5, min(30, X.shape[0] // 4))
        return TSNE(n_components=2, perplexity=perp, random_state=0,
                    init="pca").fit_transform(X), "t-SNE"
    except Exception:
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=0).fit_transform(X), "PCA"


def main():
    apply_style()
    if not os.path.exists(EMB):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "[TODO: run run_alignment_analysis.py first]",
                ha="center", va="center", color="red", transform=ax.transAxes)
        ax.axis("off")
        save(fig, OUT)
        return

    d = np.load(EMB, allow_pickle=True)
    img, gene, y = d["image_latent"], d["gene_latent"], d["y"]
    X = np.concatenate([img, gene], axis=0)
    Z, name = project(X)
    zi, zg = Z[:len(img)], Z[len(img):]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # panel 1: colored by modality
    axes[0].scatter(zi[:, 0], zi[:, 1], s=22, color=PALETTE["image"],
                    label="image", alpha=0.8, edgecolor="white", linewidth=0.3)
    axes[0].scatter(zg[:, 0], zg[:, 1], s=22, color=PALETTE["gene"],
                    label="gene", alpha=0.8, edgecolor="white", linewidth=0.3)
    axes[0].set_title(f"{name}: embeddings by modality")
    axes[0].legend()
    axes[0].set_xlabel(f"{name}-1"); axes[0].set_ylabel(f"{name}-2")

    # panel 2: image embeddings colored by label
    sc = axes[1].scatter(zi[:, 0], zi[:, 1], s=26, c=y, cmap="coolwarm",
                         alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[1].set_title(f"{name}: image embeddings by label")
    axes[1].set_xlabel(f"{name}-1"); axes[1].set_ylabel(f"{name}-2")
    fig.colorbar(sc, ax=axes[1], label="class")
    save(fig, OUT)


if __name__ == "__main__":
    main()
