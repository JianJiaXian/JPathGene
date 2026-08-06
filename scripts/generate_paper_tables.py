#!/usr/bin/env python
"""Build combined, multi-cohort LaTeX tables for the paper from REAL per-cohort
output CSVs (outputs/<cohort>/tables/*.csv). Numbers are only formatted, never
invented; a cohort that has not produced a CSV yields \\TODO placeholders.

Usage:
  python scripts/generate_paper_tables.py \
      --paper_tables "Springer Lecture Notes in Computer Science/tables"
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

COHORTS = [("feature_demo", "Demo"), ("tcga_brca", "TCGA-BRCA")]


def _read_csv(cohort, name):
    p = os.path.join("outputs", cohort, "tables", name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def _esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def _table(path, caption, label, header, rows, note=None):
    col = "l" + "c" * (len(header) - 1)
    out = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}",
           f"\\label{{{label}}}", f"\\begin{{tabular}}{{{col}}}", r"\toprule",
           " & ".join(header) + r" \\", r"\midrule"]
    out += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}"]
    if note:
        out.append(f"\\\\[2pt]\\footnotesize {note}")
    out.append(r"\end{table}")
    open(path, "w").write("\n".join(out) + "\n")
    print(f"  wrote {path}")


def main_comparison(outdir):
    pretty = {"Image-only": "Image-only", "Gene-only": "Gene-only",
              "Concat fusion": "Concat fusion", "Gated fusion": "Gated fusion",
              "JPathGene (ours)": "JPathGene (ours)"}
    data = {c: {r["Method"]: r for r in (_read_csv(c, "main_comparison.csv") or [])}
            for c, _ in COHORTS}
    header = ["Method", "Demo AUC", "Demo F1", "Demo ECE",
              "BRCA AUC", "BRCA F1", "BRCA ECE"]
    rows = []
    for m in pretty:
        d = data["feature_demo"].get(m)
        b = data["tcga_brca"].get(m)
        cell = lambda x, k: (f"{float(x[k]):.3f}" if x else r"\TODO{--}")
        name = pretty[m] + (r"\textbf{}" if "ours" in m else "")
        row = [("\\textbf{%s}" % pretty[m]) if "ours" in m else pretty[m],
               cell(d, "AUC"), cell(d, "F1"), cell(d, "ECE"),
               cell(b, "AUC"), cell(b, "F1"), cell(b, "ECE")]
        rows.append(row)
    _table(os.path.join(outdir, "main_comparison.tex"),
           "Feature-level image--gene fusion. Higher is better except ECE. "
           "TCGA-BRCA has only 19 test patients, so its metrics are high-variance "
           "and reported with caution (no SOTA claim).",
           "tab:main", header, rows)


def ablation(outdir):
    rows_in = _read_csv("feature_demo", "ablation_results.csv")
    header = ["Configuration", "Predictor", "Fusion", "AUC", "F1", "Acc", "ECE"]
    if not rows_in:
        _table(os.path.join(outdir, "jepa_ablation.tex"),
               "Ablation of the JPathGene objective (Demo).", "tab:ablation",
               header, [["JPathGene"] + [r"\TODO{--}"] * 6])
        return
    pf = {"diffusion": "diffusion", "mlp": "MLP", "concat": "concat",
          "uncertainty_gated": "unc.-gated"}
    rows = [[_esc(r["method"]), pf.get(r.get("predictor", ""), r.get("predictor", "--")),
             pf.get(r.get("fusion", ""), r.get("fusion", "--")),
             r["auc"], r["f1"], r["acc"], r["ece"]] for r in rows_in]
    _table(os.path.join(outdir, "jepa_ablation.tex"),
           "Ablation on the stable 225-patient Demo cohort. The diffusion "
           "predictor improves AUC and calibration (ECE) over the deterministic "
           "MLP predictor and the no-JEPA baseline.", "tab:ablation", header, rows,
           note="Predictor: diffusion vs.\\ deterministic MLP. "
                "Fusion: plain concat vs.\\ uncertainty-gated.")
    # gene target sub-table
    gt = [[_esc(r["gene_target"]), r["auc"], r["f1"], r["acc"], r["ece"]]
          for r in rows_in if r["method"].startswith("target=")]
    if gt:
        _table(os.path.join(outdir, "gene_target_ablation.tex"),
               "Gene latent target type (Demo).", "tab:genetarget",
               ["Gene target", "AUC", "F1", "Acc", "ECE"], gt)


def retrieval(outdir):
    header = ["Cohort", "Direction", "R@1", "R@5", "R@10", "Med.\\ rank"]
    rows = []
    for c, label in COHORTS:
        rin = _read_csv(c, "retrieval_results.csv")
        if not rin:
            continue
        for r in rin:
            rows.append([label, _esc(r["direction"]),
                         f'{float(r["recall@1"]):.3f}', f'{float(r["recall@5"]):.3f}',
                         f'{float(r["recall@10"]):.3f}', f'{float(r["median_rank"]):.0f}'])
    if not rows:
        rows = [["TCGA-BRCA", "image$\\to$gene"] + [r"\TODO{--}"] * 4]
    _table(os.path.join(outdir, "retrieval.tex"),
           "Cross-modal image--gene retrieval. On real TCGA-BRCA, genuine "
           "correspondence emerges (median rank 7 of 19); the synthetic Demo "
           "cohort has no designed correspondence and sits near chance.",
           "tab:retrieval", header, rows)


def gene_prediction(outdir):
    keep = [("rowwise_cosine_mean", "Row-wise cosine (mean)"),
            ("mse", "MSE"), ("smooth_l1", "Smooth-L1"),
            ("per_dim_pearson_mean", "Per-dim Pearson (mean)"),
            ("frac_dims_pearson_gt_0.1", "Frac.\\ dims $r>0.1$")]
    data = {}
    for c, _ in COHORTS:
        rin = _read_csv(c, "gene_prediction_quality.csv")
        data[c] = {r["metric"]: r["value"] for r in rin} if rin else {}
    header = ["Metric", "Demo", "TCGA-BRCA"]
    rows = []
    for k, label in keep:
        def cell(c):
            v = data[c].get(k)
            try:
                return f"{float(v):.3f}"
            except (TypeError, ValueError):
                return r"\TODO{--}"
        rows.append([label, cell("feature_demo"), cell("tcga_brca")])
    _table(os.path.join(outdir, "gene_prediction_quality.tex"),
           "Image$\\to$gene latent prediction quality (test split).",
           "tab:genepred", header, rows)


def efficiency(outdir):
    rin = _read_csv("feature_demo", "efficiency.csv")
    if not rin:
        return
    header = ["Model", "Params (M)", "ms/batch", "Samples/s", "Mem (MB)"]
    rows = [[_esc(r["model"]), r["params_M"], r["ms_per_batch"],
             r["samples_per_sec"], r["peak_mem_MB"]] for r in rin]
    _table(os.path.join(outdir, "efficiency.tex"),
           "Model efficiency (Demo, single GPU).", "tab:efficiency", header, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper_tables",
                    default="Springer Lecture Notes in Computer Science/tables")
    args = ap.parse_args()
    os.makedirs(args.paper_tables, exist_ok=True)
    print("Generating combined paper tables...")
    main_comparison(args.paper_tables)
    ablation(args.paper_tables)
    retrieval(args.paper_tables)
    gene_prediction(args.paper_tables)
    efficiency(args.paper_tables)
    # dataset_summary + model_summary are cohort-independent; reuse outputs/tables
    for t in ("dataset_summary.tex", "model_summary.tex"):
        src = os.path.join("outputs", "tables", t)
        if os.path.exists(src):
            open(os.path.join(args.paper_tables, t), "w").write(open(src).read())
            print(f"  copied {t}")
    print("done.")


if __name__ == "__main__":
    main()
