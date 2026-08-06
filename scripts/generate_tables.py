#!/usr/bin/env python
"""Build CSV + LaTeX tables from REAL analysis artifacts.

Every table is emitted as both .csv and .tex under outputs/tables/. When the
underlying results do not yet exist, a LaTeX table with clearly-marked
\\TODO{...} placeholders is written instead of fabricated numbers.

Tables:
  1. main_comparison       (baselines vs JPathGene)
  2. jepa_ablation
  3. gene_target_ablation
  4. retrieval
  5. gene_prediction_quality
  6. dataset_summary
  7. model_summary
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.io import OUT_ROOT as _OUT  # namespaced output root

TAB = os.path.join(_OUT, "tables")
ANA = os.path.join(_OUT, "analysis")


def _w_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _latex(path, caption, label, header, rows, note=None):
    col = "l" + "c" * (len(header) - 1)
    lines = [r"\begin{table}[t]", r"\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{col}}}", r"\toprule",
             " & ".join(_esc(h) for h in header) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(str(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(f"\\\\[2pt]\\footnotesize {note}")
    lines.append(r"\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def _esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("->", r"$\to$")


def _todo_latex(path, caption, label, header, note):
    rows = [[_esc(header[0]) and "JPathGene"] +
            [r"\TODO{--}" for _ in header[1:]]]
    _latex(path, caption, label, header, rows, note=note)


# ---------------------------------------------------------------- main table
def main_comparison():
    order = ["image_only", "gene_only", "early_fusion", "late_fusion",
             "concat_fusion", "gated_fusion", "contrastive", "pathgene_jepa"]
    pretty = {"image_only": "Image-only", "gene_only": "Gene-only",
              "early_fusion": "Early fusion", "late_fusion": "Late fusion",
              "concat_fusion": "Concat fusion", "gated_fusion": "Gated fusion",
              "contrastive": "Contrastive align.",
              "pathgene_jepa": "JPathGene (ours)"}
    found = {}
    for jp in glob.glob(os.path.join(ANA, "*_test.json")):
        try:
            d = json.load(open(jp))
            m = d.get("test_metrics") or d.get("metrics")
            name = d.get("model")
            if name and m:
                found[name] = m
        except Exception:
            continue
    header = ["Method", "AUC", "F1", "Acc", "ECE"]
    rows = []
    for name in order:
        if name in found:
            m = found[name]
            rows.append([pretty[name], f"{m['auc']:.3f}", f"{m['f1']:.3f}",
                         f"{m['accuracy']:.3f}", f"{m['ece']:.3f}"])
    cap = ("Feature-level image--gene fusion comparison. Higher is better except "
           "ECE (lower is better).")
    if rows:
        _w_csv(os.path.join(TAB, "main_comparison.csv"), header, rows)
        _latex(os.path.join(TAB, "main_comparison.tex"), cap,
               "tab:main", header, rows)
    else:
        _todo_latex(os.path.join(TAB, "main_comparison.tex"), cap, "tab:main",
                    header, "Run train.py / run\\_downstream.py to populate.")


# ---------------------------------------------------------------- ablations
def _read_ablation():
    p = os.path.join(TAB, "ablation_results.csv")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def jepa_ablation():
    rows_in = _read_ablation()
    header = ["Method", "Img->Gene", "Gene->Img", "Contrastive", "Gene target",
              "AUC", "F1", "Acc", "ECE"]
    cap = "Ablation of the JPathGene objective."
    if not rows_in:
        _todo_latex(os.path.join(TAB, "jepa_ablation.tex"), cap, "tab:ablation",
                    header, "Run run\\_ablation.py to populate.")
        return
    yn = lambda b: r"\checkmark" if str(b).lower() == "true" else "--"
    rows = [[_esc(r["method"]), yn(r["img2gene_jepa"]), yn(r["gene2img_jepa"]),
             yn(r["contrastive"]), _esc(r["gene_target"]), r["auc"], r["f1"],
             r["acc"], r["ece"]] for r in rows_in]
    _latex(os.path.join(TAB, "jepa_ablation.tex"), cap, "tab:ablation",
           header, rows)
    _w_csv(os.path.join(TAB, "jepa_ablation.csv"),
           ["method", "img2gene", "gene2img", "contrastive", "gene_target",
            "auc", "f1", "acc", "ece"],
           [[r["method"], r["img2gene_jepa"], r["gene2img_jepa"], r["contrastive"],
             r["gene_target"], r["auc"], r["f1"], r["acc"], r["ece"]]
            for r in rows_in])


def gene_target_ablation():
    rows_in = _read_ablation()
    header = ["Gene latent target", "AUC", "F1", "Acc", "ECE"]
    cap = "Effect of the gene latent target type."
    if not rows_in:
        _todo_latex(os.path.join(TAB, "gene_target_ablation.tex"), cap,
                    "tab:genetarget", header, "Run run\\_ablation.py to populate.")
        return
    rows = [[_esc(r["gene_target"]), r["auc"], r["f1"], r["acc"], r["ece"]]
            for r in rows_in if r["method"].startswith("target=")]
    if rows:
        _latex(os.path.join(TAB, "gene_target_ablation.tex"), cap,
               "tab:genetarget", header, rows)
    else:
        _todo_latex(os.path.join(TAB, "gene_target_ablation.tex"), cap,
                    "tab:genetarget", header, "Run run\\_ablation.py to populate.")


# ---------------------------------------------------------------- retrieval
def retrieval():
    p = os.path.join(TAB, "retrieval_results.csv")
    header = ["Direction", "Recall@1", "Recall@5", "Recall@10", "Median rank"]
    cap = "Image--gene cross-modal retrieval."
    if not os.path.exists(p):
        _todo_latex(os.path.join(TAB, "retrieval.tex"), cap, "tab:retrieval",
                    header, "Run run\\_alignment\\_analysis.py to populate.")
        return
    with open(p) as f:
        rin = list(csv.DictReader(f))
    rows = [[_esc(r["direction"]), f'{float(r["recall@1"]):.3f}',
             f'{float(r["recall@5"]):.3f}', f'{float(r["recall@10"]):.3f}',
             f'{float(r["median_rank"]):.1f}'] for r in rin]
    _latex(os.path.join(TAB, "retrieval.tex"), cap, "tab:retrieval", header, rows)


def gene_prediction_quality():
    p = os.path.join(TAB, "gene_prediction_quality.csv")
    header = ["Metric", "Value"]
    cap = "Gene latent prediction quality (image $\\to$ gene)."
    if not os.path.exists(p):
        _todo_latex(os.path.join(TAB, "gene_prediction_quality.tex"), cap,
                    "tab:genepred", header,
                    "Run run\\_gene\\_prediction\\_analysis.py to populate.")
        return
    keep = {"rowwise_cosine_mean": "Row-wise cosine (mean)",
            "mse": "MSE", "smooth_l1": "Smooth-L1",
            "per_dim_pearson_mean": "Per-dim Pearson (mean)",
            "frac_dims_pearson_gt_0.1": "Frac. dims $r>0.1$"}
    with open(p) as f:
        d = {r[0]: r[1] for r in csv.reader(f)}
    rows = []
    for k, label in keep.items():
        if k in d:
            try:
                rows.append([label, f"{float(d[k]):.3f}"])
            except ValueError:
                rows.append([label, _esc(d[k])])
    _latex(os.path.join(TAB, "gene_prediction_quality.tex"), cap,
           "tab:genepred", header, rows)


# ---------------------------------------------------------------- dataset/model
def dataset_summary(configs):
    header = ["Dataset", "Patients (paired)", "Image dim", "Gene dim",
              "Classes", "Train/Val/Test"]
    rows = []
    try:
        from datasets import build_datasets
        from utils.io import load_config
        for cfgp in configs:
            if not os.path.exists(cfgp):
                continue
            cfg = load_config(cfgp)
            if not os.path.isdir(cfg["data"]["root"]):
                continue
            try:
                _, meta = build_datasets(cfg)
                sc = meta["split_counts"]
                rows.append([_esc(cfg["data"]["name"]), meta["n_total"],
                             meta["image_dim"], meta["gene_dim"],
                             meta["num_classes"],
                             f"{sc['train']}/{sc['val']}/{sc['test']}"])
            except Exception as e:  # noqa
                print(f"  (dataset {cfgp} skipped: {e})")
    except Exception as e:  # noqa
        print(f"  (dataset summary unavailable: {e})")
    cap = "Feature-level datasets (patients with paired histology and genomics)."
    if rows:
        _w_csv(os.path.join(TAB, "dataset_summary.csv"), header, rows)
        _latex(os.path.join(TAB, "dataset_summary.tex"), cap, "tab:datasets",
               header, rows)
    else:
        _todo_latex(os.path.join(TAB, "dataset_summary.tex"), cap, "tab:datasets",
                    header, "Place dataset CSVs (see data/README.md) to populate.")


def model_summary():
    header = ["Component", "Architecture", "Output dim"]
    rows = [
        ["Image encoder", "FeatureMLP (+opt. MIL attn.)", "128"],
        ["Gene encoder", "MLP (opt. TabTransformer)", "128"],
        ["Image$\\to$Gene predictor", "cond. latent diffusion (FiLM denoiser)",
         "gene target dim"],
        ["Gene$\\to$Image predictor", "cond. latent diffusion (FiLM denoiser)",
         "128"],
        ["Target encoders", "EMA copy / stop-grad / PCA", "128 / d"],
        ["Fusion head", "uncertainty-gated attention", "num classes"],
    ]
    _w_csv(os.path.join(TAB, "model_summary.csv"),
           ["component", "architecture", "output_dim"], rows)
    _latex(os.path.join(TAB, "model_summary.tex"),
           "JPathGene components.", "tab:model", header, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*",
                    default=["configs/feature_csv.yaml", "configs/tcga_brca.yaml"])
    args = ap.parse_args()
    os.makedirs(TAB, exist_ok=True)
    print("Generating tables...")
    main_comparison()
    jepa_ablation()
    gene_target_ablation()
    retrieval()
    gene_prediction_quality()
    dataset_summary(args.configs)
    model_summary()
    print("done.")


if __name__ == "__main__":
    main()
