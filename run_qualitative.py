#!/usr/bin/env python
"""Collect per-patient out-of-fold predictions from several methods so we can
show qualitative cases where image-only (and other baselines) are WRONG but
JPathGene is CORRECT. Saves outputs/<cohort>/analysis/qualitative_preds.csv.

  python run_qualitative.py --config configs/tcga_brca_uni.yaml --folds 5 --seed 42
"""
import argparse
import csv
import os

import numpy as np

from datasets.image_gene_feature_dataset import build_datasets
from run_crossval_compare import (_apply_flags, _apply_override, _make_fold_splits,
                                  _oof, _train_baseline, _train_pathgene)
from models.baselines import build_baseline
from models.pathgene_jepa import build_pathgene_jepa
from utils.engine import make_loaders, move_batch
from utils.io import ensure_dir, get_device, load_config
from utils.logger import Logger, banner
from utils.seed import set_seed
from utils.io import OUT_ROOT as _OUT

import torch

METHODS = [
    ("image_only", "baseline", dict(name="image_only")),
    ("gene_only", "baseline", dict(name="gene_only")),
    ("concat_fusion", "baseline", dict(name="concat_fusion")),
    ("JPathGene (ours)", "pathgene",
     dict(predictor="mlp", target="learned", fusion="concat", anchor=True)),
]


@torch.no_grad()
def _oof_perpatient(model, loader, device, nc, is_pg):
    model.eval()
    out = {}
    from utils.engine import logits_to_prob
    for batch in loader:
        b = move_batch(batch, device)
        if is_pg:
            o = model(b["image"], b["gene"],
                      gene_pca_target=b.get("gene_pca_target"), mode="downstream")
        else:
            o = model(b["image"], b["gene"])
        prob = logits_to_prob(o["fusion_logits"], nc)
        for i, pid in enumerate(b["patient_id"]):
            out[pid] = (float(prob[i, 1]), int(b["label"][i].item()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tcga_brca_uni.yaml")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 1, 7, 11, 23])
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "cuda"))
    ensure_dir(os.path.join(_OUT, "analysis"))
    log = Logger("qualitative", os.path.join(_OUT, "logs", "qualitative.log"))
    banner(f"qualitative per-patient predictions :: {device}")

    ds, meta = build_datasets(cfg)
    ids, labels = [], []
    for s in ("train", "val", "test"):
        ids += list(ds[s].ids); labels += list(ds[s].label)
    ids, labels = np.array(ids), np.array(labels)

    # preds[disp][pid] = list of OOF probs (one per seed); truth[pid] = label
    preds = {disp: {} for disp, _, _ in METHODS}
    truth = {}
    pooled = []  # (pid, seed, label, {disp: prob}) for reliability diagrams
    for disp, kind, o in METHODS:
        c = _apply_override(cfg, kind, o)
        for seed in args.seeds:
            c["seed"] = seed
            assignments = _make_fold_splits(ids, labels, args.folds, seed)
            for override in assignments:
                set_seed(seed)
                datasets, m = build_datasets(c, override_splits=override)
                loaders = make_loaders(datasets, c)
                nc = m["num_classes"]
                if kind == "pathgene":
                    model = build_pathgene_jepa(c, m).to(device)
                    model = _apply_flags(model, o)
                    model = _train_pathgene(model, loaders, c, device, args.epochs)
                    pp = _oof_perpatient(model, loaders["test"], device, nc, True)
                else:
                    model = build_baseline(o["name"], c, m).to(device)
                    clam = c.get("loss", {}).get("lambda_align", 0.1) \
                        if o["name"] == "contrastive" else 0.0
                    model = _train_baseline(model, loaders, c, device, args.epochs, clam)
                    pp = _oof_perpatient(model, loaders["test"], device, nc, False)
                for pid, (p, y) in pp.items():
                    preds[disp].setdefault(pid, []).append(p)
                    truth[pid] = y
                    pooled.append((pid, seed, y, disp, p))
        log.info(f"  collected OOF for {disp} over {len(args.seeds)} seeds")

    # per-patient MEAN prob (for the case-study figure)
    cols = ["patient_id", "label"] + [d for d, _, _ in METHODS]
    path = os.path.join(_OUT, "analysis", "qualitative_preds.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for pid in sorted(truth):
            row = [pid, truth[pid]] + [
                round(float(np.mean(preds[d].get(pid, [float("nan")]))), 4)
                for d, _, _ in METHODS]
            w.writerow(row)
    # pooled per-(patient,seed) predictions (for reliability diagrams)
    pp_path = os.path.join(_OUT, "analysis", "qualitative_preds_pooled.csv")
    bypid = {}
    for pid, seed, y, disp, p in pooled:
        bypid.setdefault((pid, seed), {"label": y})[disp] = p
    with open(pp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "seed", "label"] + [d for d, _, _ in METHODS])
        for (pid, seed), d in bypid.items():
            if all(m in d for m, _, _ in METHODS):
                w.writerow([pid, seed, d["label"]] +
                           [round(d[m], 4) for m, _, _ in METHODS])
    # report cases: image-only wrong, ours correct (using per-patient mean prob)
    def _mean(d, pid):
        v = d.get(pid, [0.5])
        return float(np.mean(v))
    n_win = 0
    for pid in truth:
        y = truth[pid]
        img_wrong = (_mean(preds["image_only"], pid) > 0.5) != bool(y)
        ours_right = (_mean(preds["JPathGene (ours)"], pid) > 0.5) == bool(y)
        if img_wrong and ours_right:
            n_win += 1
    banner(f"saved {path}; {n_win} patients: image-only WRONG & ours CORRECT")


if __name__ == "__main__":
    main()
