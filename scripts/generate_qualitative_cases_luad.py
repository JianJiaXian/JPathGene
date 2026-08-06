#!/usr/bin/env python
"""Qualitative case-study figure (TCGA-LUAD stage endpoint).

For a handful of REAL held-out patients, show the two inputs that the model
actually consumes -- the pre-extracted histology (UNI2-h) feature vector and the
RNA-seq expression vector -- as heatmap strips, alongside every method's REAL
out-of-fold predicted probability. Cases are selected where JPathGene predicts
the correct class while two or more baselines (image-only / gene-only / concat
fusion) predict the wrong class, illustrating that cross-modal latent prediction
recovers the right call where single-modality and naive fusion experts fail.

Nothing here is synthesised: probabilities come from
``outputs/tcga_luad_stage/analysis/qualitative_preds.csv`` and the input strips
are the genuine feature rows from the LUAD-UNI cohort (z-scored only for display).

  python scripts/generate_qualitative_cases_luad.py
"""
import csv
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from utils.visualization import PALETTE, apply_style  # noqa: E402

DATA = "data/TCGA_LUAD_UNI"
CSV = "outputs/tcga_luad_stage/analysis/qualitative_preds.csv"
OUT = "outputs/tcga_luad_stage/figures/qualitative_luad_cases.pdf"
PAPER = ("JPathGene__Cross_Modal_Latent_Diffusion__10_/figures/"
         "qualitative_luad.pdf")

METHODS = ["image_only", "gene_only", "concat_fusion", "JPathGene (ours)"]
DISP = {"image_only": "Image-only", "gene_only": "Gene-only",
        "concat_fusion": "Concat fusion", "JPathGene (ours)": "JPathGene"}

# A curated, diverse set of REAL cases (verified present in the prediction CSV):
# two advanced-stage (y=1) and two early-stage (y=0); across the four, every
# baseline is shown failing. Falls back to data-driven selection if absent.
CURATED = ["TCGA-53-A4EZ", "TCGA-55-7284", "TCGA-NJ-A4YP", "TCGA-86-A456"]


def load_feature_row(path, wanted):
    """Return {patient_id: np.ndarray} for the wanted ids from a feature CSV."""
    wanted = set(wanted)
    out = {}
    with open(path) as f:
        next(f)  # header
        for line in f:
            pid, rest = line.split(",", 1)
            if pid in wanted:
                out[pid] = np.fromstring(rest, sep=",")
    return out


def zclip(v, c=2.5):
    s = v.std()
    z = (v - v.mean()) / (s if s > 1e-6 else 1.0)
    return np.clip(z, -c, c)


def tile(v, rows, cols):
    g = np.zeros(rows * cols)
    g[: len(v)] = v[: rows * cols]
    return g.reshape(rows, cols)


def wrong(r, m):
    return (float(r[m]) > 0.5) != bool(int(r["label"]))


def select_cases(rows, n=4):
    def nwrong_base(r):
        return sum(wrong(r, m) for m in METHODS[:3])

    def margin(r):
        return abs(float(r["JPathGene (ours)"]) - 0.5)

    by_id = {r["patient_id"]: r for r in rows}
    cases = [by_id[p] for p in CURATED if p in by_id
             and not wrong(by_id[p], "JPathGene (ours)")]
    if len(cases) < n:  # data-driven fallback keeps the figure reproducible
        cand = [r for r in rows if not wrong(r, "JPathGene (ours)")
                and nwrong_base(r) >= 2 and r["patient_id"] not in CURATED]
        cand.sort(key=lambda r: (-nwrong_base(r), -margin(r)))
        cases += cand[: n - len(cases)]
    return cases[:n]


def main():
    if not os.path.exists(CSV):
        print(f"[qual] {CSV} missing; run run_qualitative.py first")
        return
    rows = list(csv.DictReader(open(CSV)))
    cases = select_cases(rows)
    if not cases:
        print("[qual] no suitable cases found")
        return
    ids = [c["patient_id"] for c in cases]
    img = load_feature_row(os.path.join(DATA, "image_features.csv"), ids)
    gene = load_feature_row(os.path.join(DATA, "genomic_features.csv"), ids)

    apply_style()
    n = len(cases)
    fig = plt.figure(figsize=(9.8, 1.5 * n + 1.25))
    gs = fig.add_gridspec(n, 3, width_ratios=[1.0, 1.0, 3.7],
                          hspace=0.62, wspace=0.46,
                          left=0.16, right=0.985, top=0.82, bottom=0.11)

    for i, r in enumerate(cases):
        pid = r["patient_id"]
        y = int(r["label"])
        axi, axg, axp = (fig.add_subplot(gs[i, j]) for j in range(3))

        # ---- real input feature strips (z-scored for display only) ----
        axi.imshow(tile(zclip(img[pid]), 24, 64), cmap="RdBu_r",
                   vmin=-2.5, vmax=2.5, aspect="auto")
        axg.imshow(tile(zclip(gene[pid]), 10, 20), cmap="RdBu_r",
                   vmin=-2.5, vmax=2.5, aspect="auto")
        for a in (axi, axg):
            a.set_xticks([]); a.set_yticks([])
        axi.set_ylabel(f"{pid}\n(y={y}, {'advanced' if y else 'early'})",
                       fontsize=8.0, rotation=90, va="center", labelpad=6)
        if i == 0:
            axi.set_title("Histology (UNI2-h)", fontsize=9.3,
                          color=PALETTE["image"], fontweight="bold", pad=6)
            axg.set_title("Gene expression", fontsize=9.3,
                          color=PALETTE["gene"], fontweight="bold", pad=6)
            axp.set_title("Predicted $P(\\mathrm{advanced\\ stage})$",
                          fontsize=9.3, fontweight="bold", pad=6)

        # ---- real per-method predictions ----
        axp.axvspan(0.5 if y == 1 else 0.0, 1.0 if y == 1 else 0.5,
                    color=PALETTE["light"], zorder=0)
        axp.axvline(0.5, color="#333", lw=1.3, ls="--", zorder=1)
        for mi, m in enumerate(METHODS):
            p = float(r[m])
            ok = (p > 0.5) == bool(y)
            ours = m == "JPathGene (ours)"
            yy = len(METHODS) - 1 - mi
            axp.scatter([p], [yy], marker="o" if ok else "X",
                        s=190 if ours else 95,
                        color=PALETTE["predict"] if ok else PALETTE["gene"],
                        edgecolor="black" if ours else "white",
                        linewidth=1.6 if ours else 0.7, zorder=3)
            axp.text(p, yy + 0.30, f"{p:.2f}", ha="center", va="bottom",
                     fontsize=7.6,
                     fontweight="bold" if ours else "normal",
                     color="#222")
        axp.set_xlim(-0.02, 1.02)
        axp.set_ylim(-0.6, len(METHODS) - 0.4)
        axp.set_yticks(range(len(METHODS)))
        axp.set_yticklabels([DISP[m] for m in reversed(METHODS)], fontsize=8.6)
        axp.tick_params(axis="y", length=0)
        axp.grid(axis="y", visible=False)
        if i < n - 1:
            axp.set_xticklabels([])

    # ---- bottom note ----
    fig.text(0.71, 0.035,
             "predicted probability (threshold 0.5; shaded = correct half)",
             ha="center", fontsize=8.3, color=PALETTE["neutral"])

    leg = [Line2D([0], [0], marker="o", color="w",
                  markerfacecolor=PALETTE["predict"], markersize=9,
                  label="correct"),
           Line2D([0], [0], marker="X", color="w",
                  markerfacecolor=PALETTE["gene"], markersize=9, label="wrong"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
                  markeredgecolor="black", markersize=11,
                  label="JPathGene (emphasised)")]
    fig.legend(handles=leg, loc="upper center", ncol=3, fontsize=8.6,
               bbox_to_anchor=(0.5, 0.998))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT[:-4] + ".png")
    if os.path.isdir(os.path.dirname(PAPER)):
        shutil.copyfile(OUT, PAPER)
        print(f"[qual] copied -> {PAPER}")
    print(f"[qual] saved {OUT} with cases: {ids}")


if __name__ == "__main__":
    main()
