#!/usr/bin/env python
"""Turn the LUAD CV comparison CSV into a paper LaTeX table (second cohort).
Honest: numbers come straight from the CSV. Bolds best AUC / best ECE, marks ours.

  python scripts/generate_luad_table.py \
      --compare outputs/tcga_luad_uni/tables/luad_final.csv \
      --endpoint "TP53 mutation" --n 450
"""
import argparse, csv, os

PAPER = "Springer Lecture Notes in Computer Science/tables"


def _read(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def _fmt(m, s):
    return f"{float(m):.3f}\\,$\\pm$\\,{float(s):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", default="outputs/tcga_luad_uni/tables/luad_final.csv")
    ap.add_argument("--endpoint", default="TP53 mutation")
    ap.add_argument("--n", default="")
    ap.add_argument("--out", default=os.path.join(PAPER, "luad_comparison.tex"))
    ap.add_argument("--label", default="tab:luad_compare")
    args = ap.parse_args()
    rows = _read(args.compare)
    if not rows:
        print("no rows in", args.compare)
        return
    aucs = [float(r["auc"]) for r in rows]
    eces = [float(r["ece"]) for r in rows]
    best_auc, best_ece = max(aucs), min(eces)
    rows = sorted(rows, key=lambda r: -float(r["auc"]))
    body = []
    for r in rows:
        name = r["method"].replace("&", "\\&")
        bold = "ours" in name.lower() or "pathgene" in name.lower()
        disp = f"\\textbf{{{name}}}" if bold else name
        auc = _fmt(r["auc"], r["auc_std"])
        if abs(float(r["auc"]) - best_auc) < 1e-9:
            auc = f"\\textbf{{{auc}}}"
        ece = _fmt(r["ece"], r["ece_std"])
        if abs(float(r["ece"]) - best_ece) < 1e-9:
            ece = f"\\textbf{{{ece}}}"
        body.append(f"{disp} & {auc} & {_fmt(r['f1'], r['f1_std'])} & {ece} \\\\")
    nstr = f", $n{{=}}{args.n}$" if args.n else ""
    tex = [
        r"\begin{table}[t]", r"\centering",
        f"\\caption{{Second-cohort validation on \\textbf{{TCGA-LUAD}} "
        f"({args.endpoint} endpoint{nstr}; UNI2-h features, $5$-fold $\\times$ "
        r"$5$-seed leakage-safe CV; mean$\pm$std). Best AUC / ECE in bold.}",
        f"\\label{{{args.label}}}",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Method & AUC & F1 & ECE \\", r"\midrule",
    ] + body + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    os.makedirs(PAPER, exist_ok=True)
    open(args.out, "w").write("\n".join(tex) + "\n")
    print("wrote", args.out, f"({len(rows)} methods)")


if __name__ == "__main__":
    main()
