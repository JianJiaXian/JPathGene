#!/usr/bin/env python
"""Result figures from the REAL CV comparison CSV (no fabricated numbers):
  (A) AUC bar chart with error bars, our methods highlighted.
  (B) AUC-vs-ECE 'trustworthiness frontier' -- our method sits top-left
      (high AUC + low ECE); gene-only is high-AUC but badly calibrated.

  python scripts/generate_result_figures.py \
      --csv outputs/tcga_brca_uni/tables/uni_final.csv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt  # noqa: E402

from utils.visualization import PALETTE, apply_style, save  # noqa: E402

OUTDIR = "outputs/tcga_brca_uni/figures"


def _is_ours(name):
    return "ours" in name.lower() or "pathgene" in name.lower()


def bar_auc(rows, out):
    apply_style()
    rows = sorted(rows, key=lambda r: float(r["auc"]))
    names = [r["method"].replace(" (ours)", "*") for r in rows]
    auc = [float(r["auc"]) for r in rows]
    err = [float(r["auc_std"]) for r in rows]
    cols = [PALETTE["gene"] if _is_ours(r["method"]) else PALETTE["neutral"]
            for r in rows]
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    y = range(len(names))
    ax.barh(list(y), auc, xerr=err, color=cols, edgecolor="white",
            error_kw=dict(ecolor="#888", lw=1.2, capsize=2.5))
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("AUC (5-fold $\\times$ 5-seed CV)")
    ax.set_xlim(0.74, 0.99)
    best = max(range(len(auc)), key=lambda i: auc[i])
    ax.text(auc[best] + err[best] + 0.003, best, "best", va="center",
            color=PALETTE["gene"], fontsize=9, fontweight="bold")
    ax.set_title("Multimodal cancer classification on TCGA-BRCA (UNI2-h features)")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["gene"], label="JPathGene (ours)"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["neutral"], label="baselines")],
        loc="lower right")
    save(fig, out)


def frontier(rows, out):
    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    for r in rows:
        ours = _is_ours(r["method"])
        x, yv = float(r["ece"]), float(r["auc"])
        col = PALETTE["gene"] if ours else PALETTE["neutral"]
        ax.scatter([x], [yv], s=130 if ours else 70, color=col,
                   edgecolor="white", linewidth=1.2, zorder=3, alpha=0.95)
        lab = r["method"].split(" (")[0]
        if ours or "Gene-only" in r["method"]:
            ax.annotate(r["method"].replace(" (ours)", "*").replace("JPathGene ", "PG "),
                        (x, yv), textcoords="offset points", xytext=(6, 4),
                        fontsize=8.2, color=col, fontweight="bold" if ours else "normal")
    ax.set_xlabel("Expected Calibration Error (ECE) $\\downarrow$ better")
    ax.set_ylabel("AUC $\\uparrow$ better")
    ax.set_title("Trustworthiness frontier: accuracy vs. calibration")
    ax.invert_xaxis()  # lower ECE to the right -> best is top-right
    ax.text(0.97, 0.06, "best region\n(high AUC, low ECE)",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            color=PALETTE["predict"], fontstyle="italic")
    save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/tcga_brca_uni/tables/uni_final.csv")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    rows = list(csv.DictReader(open(args.csv)))
    bar_auc(rows, os.path.join(OUTDIR, "uni_auc_bar.pdf"))
    frontier(rows, os.path.join(OUTDIR, "uni_frontier.pdf"))
    # copy to paper
    for f in ("uni_auc_bar.pdf", "uni_frontier.pdf"):
        import shutil
        src = os.path.join(OUTDIR, f)
        if os.path.exists(src):
            shutil.copyfile(src, f"Springer Lecture Notes in Computer Science/figures/{f}")
    print("result figures done")


if __name__ == "__main__":
    main()
