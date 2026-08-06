#!/usr/bin/env python
"""Qualitative results figure from REAL per-patient OOF predictions
(outputs/<cohort>/analysis/qualitative_preds.csv): select held-out patients where
image-only and other baselines predict the WRONG class but JPathGene predicts
CORRECTLY, and show each method's predicted probability against the threshold.

  python scripts/generate_qualitative_figure.py
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt  # noqa: E402

from utils.visualization import PALETTE, apply_style, save  # noqa: E402
from utils.io import OUT_ROOT as _OUT  # noqa: E402

CSV = os.path.join(_OUT, "analysis", "qualitative_preds.csv")
OUT = os.path.join(_OUT, "figures", "qualitative_cases.pdf")
METHODS = ["image_only", "gene_only", "concat_fusion", "JPathGene (ours)"]
DISP = {"image_only": "Image-only", "gene_only": "Gene-only",
        "concat_fusion": "Concat fusion", "JPathGene (ours)": "JPathGene (ours)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    if not os.path.exists(CSV):
        print(f"[qual] {CSV} missing; run run_qualitative.py first")
        return
    rows = list(csv.DictReader(open(CSV)))

    def wrong(r, m):
        return (float(r[m]) > 0.5) != bool(int(r["label"]))

    # rank cases: image-only wrong + ours correct, prefer also concat/gene wrong
    cases = [r for r in rows
             if wrong(r, "image_only") and not wrong(r, "JPathGene (ours)")]
    cases.sort(key=lambda r: -(int(wrong(r, "concat_fusion"))
                               + int(wrong(r, "gene_only"))
                               + abs(float(r["JPathGene (ours)"]) - 0.5)))
    cases = cases[:args.n]
    if not cases:
        print("[qual] no image-wrong/ours-right cases found")
        return

    apply_style()
    fig, ax = plt.subplots(figsize=(8.6, 0.85 * len(cases) + 1.6))
    yticks, ylabels = [], []
    for row_i, r in enumerate(cases):
        y0 = len(cases) - row_i - 1
        label = int(r["label"])
        # shade the correct half
        ax.axhspan(y0 - 0.42, y0 + 0.42, xmin=(0.5 if label == 1 else 0.0),
                   xmax=(1.0 if label == 1 else 0.5),
                   color=PALETTE["light"], zorder=0)
        for mi, m in enumerate(METHODS):
            p = float(r[m])
            ok = (p > 0.5) == bool(label)
            ours = m == "JPathGene (ours)"
            ax.scatter([p], [y0 + (mi - 1.5) * 0.16],
                       marker="o" if ok else "X",
                       s=150 if ours else 90,
                       color=PALETTE["predict"] if ok else PALETTE["gene"],
                       edgecolor="black" if ours else "white",
                       linewidth=1.4 if ours else 0.6, zorder=3)
        yticks.append(y0)
        ylabels.append(f"{r['patient_id'][:12]}\n(true={label})")
    ax.axvline(0.5, color="#333", lw=1.4, ls="--")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, len(cases) - 0.4)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.set_xlabel("predicted probability of positive class (threshold 0.5)")
    ax.set_title("Cases where baselines fail and JPathGene succeeds")
    # legend
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["predict"],
                  markersize=9, label="correct"),
           Line2D([0], [0], marker="X", color="w", markerfacecolor=PALETTE["gene"],
                  markersize=9, label="wrong"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="#777",
                  markeredgecolor="black", markersize=11, label="JPathGene (large)")]
    ax.legend(handles=leg, loc="lower center", ncol=3, fontsize=8.5,
              bbox_to_anchor=(0.5, 1.02))
    # within-row method order note
    ax.text(1.04, len(cases) / 2 - 0.5,
            "per row, top$\\to$bottom:\nImage-only, Gene-only,\nConcat, "
            "JPathGene",
            fontsize=7.5, va="center", color=PALETTE["neutral"])
    save(fig, OUT)
    import shutil
    dst = "Springer Lecture Notes in Computer Science/figures/qualitative_cases.pdf"
    if os.path.exists(OUT):
        shutil.copyfile(OUT, dst)
    print("qualitative figure done")


if __name__ == "__main__":
    main()
