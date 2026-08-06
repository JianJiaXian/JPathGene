#!/usr/bin/env python
"""Finalize the LUAD second-cohort reporting:
  (1) exact Welch t-tests (scipy) for the cited comparisons, t-based 95% CIs;
  (2) one consolidated cross-cohort table (key methods x {TP53, stage}).
Reads the per-seed CSVs from both LUAD runs. Writes the combined table to the
paper and a significance JSON. Run in the container (scipy) on a compute node.
"""
import csv, json, os
import numpy as np
from scipy import stats

TP53 = "outputs/tcga_luad_uni/tables/per_seed.csv"
STAGE = "outputs/tcga_luad_stage/tables/per_seed.csv"
PAPER = "Springer Lecture Notes in Computer Science/tables/luad_combined.tex"
SIGOUT = "outputs/tcga_luad_uni/analysis/luad_significance.json"

# canonical display set for the combined table (rows), mapped to per_seed names
ROWS = [
    ("Image-only", "Image-only"),
    ("Gene-only", "Gene-only"),
    ("Concat fusion", "Concat fusion"),
    ("Late fusion", "Late fusion"),
    ("JPathGene (full, ours)", "JPathGene (full, ours)"),
    ("JPathGene (gene-anchored, ours)", "JPathGene (gene-anchored, ours)"),
    ("JPathGene (CoRE, ours)", "JPathGene (CoRE, ours)"),
]


def load(path):
    d = {}
    for r in csv.DictReader(open(path)):
        d.setdefault(r["method"], {"auc": [], "ece": []})
        d[r["method"]]["auc"].append(float(r["auc"]))
        d[r["method"]]["ece"].append(float(r["ece"]))
    return d


def welch(a, b):
    a, b = np.array(a), np.array(b)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) +
                                     (vb / nb) ** 2 / (nb - 1))
    tcrit = stats.t.ppf(0.975, df)
    d = a.mean() - b.mean()
    return dict(mean_a=float(a.mean()), mean_b=float(b.mean()), diff=float(d),
                t=float(t), df=float(df), p=float(p),
                ci=[float(d - tcrit * se), float(d + tcrit * se)])


def main():
    tp, st = load(TP53), load(STAGE)
    sig = {}
    sig["tp53_CoRE_vs_gene_auc"] = welch(tp["JPathGene (CoRE, ours)"]["auc"],
                                         tp["Gene-only"]["auc"])
    sig["tp53_CoRE_vs_gene_ece"] = welch(tp["JPathGene (CoRE, ours)"]["ece"],
                                         tp["Gene-only"]["ece"])
    # stage: best simple fusion and best JPathGene vs gene-only (AUC)
    for m in ["Late fusion", "Early fusion", "JPathGene (MLP pred.)",
              "JPathGene (gene-anchored, ours)", "JPathGene (CoRE, ours)"]:
        if m in st:
            sig[f"stage_{m}_vs_gene_auc"] = welch(st[m]["auc"],
                                                  st["Gene-only"]["auc"])
    os.makedirs(os.path.dirname(SIGOUT), exist_ok=True)
    json.dump(sig, open(SIGOUT, "w"), indent=2)
    for k, v in sig.items():
        print(f"{k:42s} diff {v['diff']:+.3f}  p={v['p']:.4f}  "
              f"CI[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")

    # ---- consolidated table ----
    def cell(vals, best, lower_better=False):
        m, s = np.mean(vals), np.std(vals)
        txt = f"{m:.3f}\\,$\\pm$\\,{s:.3f}"
        is_best = (m <= best + 1e-9) if lower_better else (m >= best - 1e-9)
        return f"\\textbf{{{txt}}}" if is_best else txt

    tp_auc_best = max(np.mean(tp[r[1]]["auc"]) for r in ROWS if r[1] in tp)
    tp_ece_best = min(np.mean(tp[r[1]]["ece"]) for r in ROWS if r[1] in tp)
    st_auc_best = max(np.mean(st[r[1]]["auc"]) for r in ROWS if r[1] in st)
    st_ece_best = min(np.mean(st[r[1]]["ece"]) for r in ROWS if r[1] in st)
    body = []
    for disp, key in ROWS:
        if key not in tp or key not in st:
            continue
        name = f"\\textbf{{{disp}}}" if "ours" in disp else disp
        body.append(" & ".join([
            name,
            cell(tp[key]["auc"], tp_auc_best),
            cell(tp[key]["ece"], tp_ece_best, lower_better=True),
            cell(st[key]["auc"], st_auc_best),
            cell(st[key]["ece"], st_ece_best, lower_better=True)]) + r" \\")
    tex = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Second-cohort validation on \textbf{TCGA-LUAD} (UNI2-h "
        r"features, $5$-fold $\times$ $5$-seed leakage-safe CV; mean$\pm$std; key "
        r"methods). \textbf{TP53} is molecular (genomics strong); \textbf{stage} "
        r"(early vs.\ II$+$) is anatomic and \emph{not} genomically saturated. On "
        r"TP53 the CoRE variant is best on both AUC and ECE; on stage every "
        r"multimodal method far exceeds the weak gene-only baseline. Best AUC/ECE "
        r"per column in bold.}",
        r"\label{tab:luad_combined}",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"& \multicolumn{2}{c}{TP53 mutation ($n{=}460$)} & "
        r"\multicolumn{2}{c}{Stage early/II$+$ ($n{=}393$)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Method & AUC & ECE & AUC & ECE \\", r"\midrule",
    ] + body + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(PAPER, "w").write("\n".join(tex) + "\n")
    print("wrote", PAPER)


if __name__ == "__main__":
    main()
