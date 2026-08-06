#!/usr/bin/env python
"""Render the full 2^4 factorial ablation (uni_factorial.csv, method names encoded
as FAC:JMCA=abcd) into a paper LaTeX table with checkmark columns for each module,
sorted by AUC, best row bolded.

  python scripts/generate_factorial_table.py \
      --csv outputs/tcga_brca_uni/tables/uni_factorial.csv
"""
import argparse
import csv
import os

PAPER = "Springer Lecture Notes in Computer Science/tables/uni_factorial.tex"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/tcga_brca_uni/tables/uni_factorial.csv")
    args = ap.parse_args()
    if not os.path.exists(args.csv):
        print(f"[factorial] {args.csv} missing")
        return
    rows = list(csv.DictReader(open(args.csv)))
    parsed = []
    for r in rows:
        name = r["method"]
        bits = name.split("=")[-1] if "=" in name else "0000"
        if len(bits) != 4:
            continue
        parsed.append((bits, float(r["auc"]), float(r["auc_std"]),
                       float(r["f1"]), float(r["f1_std"]),
                       float(r["ece"]), float(r["ece_std"])))
    if not parsed:
        print("[factorial] no FAC rows parsed")
        return
    best_auc = max(p[1] for p in parsed)
    best_ece = min(p[5] for p in parsed)
    parsed.sort(key=lambda p: -p[1])
    ck = lambda b: r"\checkmark" if b == "1" else "--"

    def fmt(m, s, bold=False):
        t = f"{m:.3f}\\,$\\pm$\\,{s:.3f}"
        return f"\\textbf{{{t}}}" if bold else t

    body = []
    for bits, a, asd, f, fsd, e, esd in parsed:
        J, M, C, A = bits
        body.append(" & ".join([
            ck(J), ck(M), ck(C), ck(A),
            fmt(a, asd, abs(a - best_auc) < 1e-9),
            fmt(f, fsd),
            fmt(e, esd, abs(e - best_ece) < 1e-9)]) + r" \\")
    tex = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Full $2^4$ factorial ablation of \emph{our} JPathGene "
        r"modules on TCGA-BRCA (UNI2-h features, $5$-fold $\times$ $5$-seed CV; "
        r"mean$\pm$std): cross-modal \textbf{J}EPA, \textbf{M}asked I-JEPA, "
        r"\textbf{C}ontrastive alignment, gene-\textbf{A}nchor. The gene-anchor is "
        r"decisive: all eight top-ranked rows include it. On top of the anchor the "
        r"JEPA modules yield the most \emph{stable} high-AUC configurations -- our "
        r"default (J,M,A) and the full model (J,M,C,A) have the two lowest "
        r"variances of all sixteen. Contrastive is roughly neutral. Best mean "
        r"AUC/ECE in bold.}",
        r"\label{tab:uni_factorial}",
        r"\begin{tabular}{cccc|ccc}", r"\toprule",
        r"J & M & C & A & AUC & F1 & ECE \\", r"\midrule",
    ] + body + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    os.makedirs(os.path.dirname(PAPER), exist_ok=True)
    open(PAPER, "w").write("\n".join(tex) + "\n")
    print(f"wrote {PAPER} ({len(parsed)} combinations)")


if __name__ == "__main__":
    main()
