#!/usr/bin/env python
"""Reliability diagrams + ECE (15-bin, top-label, no post-hoc calibration) from
the pooled out-of-fold per-patient predictions (qualitative_preds.csv). Addresses
reviewer requests for ECE methodology and reliability visualization.

  python scripts/generate_reliability_figure.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from utils.visualization import PALETTE, apply_style, save  # noqa: E402
from utils.io import OUT_ROOT as _OUT  # noqa: E402

CSV = os.path.join(_OUT, "analysis", "qualitative_preds_pooled.csv")
OUT = os.path.join(_OUT, "figures", "reliability.pdf")
METHODS = [("image_only", "Image-only"), ("gene_only", "Gene-only"),
           ("concat_fusion", "Concat"), ("JPathGene (ours)", "JPathGene")]
NBINS = 15


def ece_and_curve(probs, labels, n_bins=NBINS):
    conf = np.maximum(probs, 1 - probs)          # top-label confidence
    pred = (probs > 0.5).astype(int)
    acc = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys, ece, n = [], [], 0.0, len(labels)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            xs.append(conf[m].mean()); ys.append(acc[m].mean())
            ece += (m.sum() / n) * abs(acc[m].mean() - conf[m].mean())
    return ece, np.array(xs), np.array(ys)


def main():
    if not os.path.exists(CSV):
        print(f"[reliability] {CSV} missing"); return
    rows = list(csv.DictReader(open(CSV)))
    y = np.array([int(r["label"]) for r in rows])
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5), sharey=True)
    for ax, (key, disp) in zip(axes, METHODS):
        p = np.array([float(r[key]) for r in rows])
        ece, xs, ys = ece_and_curve(p, y)
        ax.plot([0, 1], [0, 1], "--", color="#888", lw=1.3)
        col = PALETTE["gene"] if "PathGene" in disp else PALETTE["neutral"]
        ax.plot(xs, ys, "-o", color=col, markersize=4, lw=1.8)
        ax.fill_between(xs, ys, xs, color=col, alpha=0.15)
        ax.set_title(f"{disp}\nECE={ece:.3f}", fontsize=11)
        ax.set_xlabel("confidence"); ax.set_xlim(0.45, 1.0); ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("accuracy")
    fig.suptitle("Reliability diagrams (15-bin, top-label; no temperature scaling)",
                 fontsize=12, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, OUT)
    import shutil
    dst = "Springer Lecture Notes in Computer Science/figures/reliability.pdf"
    if os.path.exists(OUT):
        shutil.copyfile(OUT, dst)
    print("reliability figure done")


if __name__ == "__main__":
    main()
