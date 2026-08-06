#!/usr/bin/env python
"""Comprehensive, leakage-safe CV comparison of many multimodal methods on the
same protocol (stratified K-fold x seeds, train-only scalers/PCA per fold,
out-of-fold pooled metrics). Reports mean +/- std AUC / F1 / Acc / ECE so every
method is compared fairly. All methods see the SAME gene input (full panel); only
the fusion/predictor/target differ.

  python run_crossval_compare.py --config configs/tcga_brca_pathway.yaml \
      --folds 5 --seeds 42 1 7 --epochs 80 --exp_name brca_compare
"""
import argparse
import copy
import os

import numpy as np
import torch

from datasets.image_gene_feature_dataset import build_datasets
from models.baselines import build_baseline
from models.contrastive import info_nce
from models.losses import PathGeneLoss, classification_loss
from models.pathgene_jepa import build_pathgene_jepa
from utils.engine import (build_optimizer, build_scheduler, evaluate_split,
                          logits_to_prob, make_loaders, move_batch)
from utils.io import ensure_dir, get_device, load_config, save_json
from utils.logger import Logger, banner
from utils.metrics import expected_calibration_error
from utils.seed import set_seed
from utils.io import OUT_ROOT as _OUT


def ov(**kw):
    return kw


# (display name, kind, config-override dict)
METHODS = [
    ("Image-only", "baseline", ov(name="image_only")),
    ("Gene-only", "baseline", ov(name="gene_only")),
    ("Early fusion", "baseline", ov(name="early_fusion")),
    ("Late fusion", "baseline", ov(name="late_fusion")),
    ("Concat fusion", "baseline", ov(name="concat_fusion")),
    ("Gated fusion", "baseline", ov(name="gated_fusion")),
    ("Cross-attention", "baseline", ov(name="cross_attention")),
    ("TabTransformer concat", "baseline",
     ov(name="concat_fusion", gene_encoder_type="tabtransformer")),
    ("Contrastive (CLIP-style)", "baseline", ov(name="contrastive")),
    ("JPathGene (MLP pred.)", "pathgene",
     ov(predictor="mlp", target="learned", fusion="concat")),
    ("JPathGene (diffusion, PCA)", "pathgene",
     ov(predictor="diffusion", target="pca", fusion="uncertainty_gated")),
    ("JPathGene (full: diff.+pathway)", "pathgene",
     ov(predictor="diffusion", target="pathway", fusion="uncertainty_gated")),
    ("JPathGene (gene-anchored, ours)", "pathgene",
     ov(predictor="mlp", target="learned", fusion="concat", anchor=True)),
    ("JPathGene (CoRE, ours)", "pathgene",
     ov(predictor="diffusion", target="learned", fusion="core")),
    ("JPathGene (full, ours)", "pathgene",
     ov(predictor="diffusion", target="learned", fusion="concat", anchor=True,
        i2g=True, g2i=True, masked=True, contrastive=False)),
]


# Component-incremental ablation of our method (all diffusion predictor, full
# genes), each row adding one module to show its marginal contribution.
ABLATION = [
    ("Fusion only (no JEPA)", "pathgene",
     ov(i2g=False, g2i=False, masked=False, contrastive=False,
        target="learned", fusion="concat")),
    ("+ Image$\\to$Gene diffusion JEPA", "pathgene",
     ov(i2g=True, g2i=False, masked=False, contrastive=False,
        target="learned", fusion="concat")),
    ("+ bidirectional JEPA", "pathgene",
     ov(i2g=True, g2i=True, masked=False, contrastive=False,
        target="learned", fusion="concat")),
    ("+ masked I-JEPA", "pathgene",
     ov(i2g=True, g2i=True, masked=True, contrastive=False,
        target="learned", fusion="concat")),
    ("+ contrastive alignment", "pathgene",
     ov(i2g=True, g2i=True, masked=True, contrastive=True,
        target="learned", fusion="concat")),
    ("+ pathway target", "pathgene",
     ov(i2g=True, g2i=True, masked=True, contrastive=True,
        target="pathway", fusion="concat")),
    ("+ uncertainty-gated fusion (full)", "pathgene",
     ov(i2g=True, g2i=True, masked=True, contrastive=True,
        target="pathway", fusion="uncertainty_gated")),
]


# Full-factorial ablation: all 2^4 on/off combinations of the four toggleable
# modules (cross-modal JEPA, masked I-JEPA, contrastive, gene-anchor). The method
# name encodes the bits as "FAC:JMCA=abcd" so the table generator can parse them.
import itertools as _it
FACTORIAL = []
for _A, _B, _C, _D in _it.product([0, 1], repeat=4):
    FACTORIAL.append((
        f"FAC:JMCA={_A}{_B}{_C}{_D}", "pathgene",
        dict(i2g=bool(_A), g2i=bool(_A), masked=bool(_B), contrastive=bool(_C),
             anchor=bool(_D), predictor="diffusion", target="learned",
             fusion="concat")))


def _apply_override(cfg, kind, o):
    c = copy.deepcopy(cfg)
    if "gene_encoder_type" in o:
        c["model"]["gene_encoder"]["type"] = o["gene_encoder_type"]
    if kind == "pathgene":
        c["model"]["predictor"]["type"] = o.get("predictor", "diffusion")
        c["model"]["target"]["gene_target_type"] = o.get("target", "pathway")
        c["model"]["fusion"]["mode"] = o.get("fusion", "uncertainty_gated")
        c["model"]["fusion"]["gene_anchor"] = o.get("anchor", False)
        c["loss"]["use_contrastive"] = o.get("contrastive", True)
    return c


def _apply_flags(model, o):
    """Toggle JEPA components on a built PathGeneJEPA (for the ablation)."""
    model.use_img2gene = o.get("i2g", True)
    model.use_gene2img = o.get("g2i", True)
    model.use_contrastive = o.get("contrastive", True)
    if "masked" in o:
        model.use_masked_jepa = o["masked"] and model.predictor_type == "diffusion"
    return model


def _make_fold_splits(ids, labels, folds, seed):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rng = np.random.RandomState(seed)
    out = []
    for tr_idx, te_idx in skf.split(ids, labels):
        tr_idx = list(tr_idx)
        rng.shuffle(tr_idx)
        n_val = max(2, int(0.15 * len(tr_idx)))
        val = set(tr_idx[:n_val])
        te = set(te_idx)
        out.append({ids[i]: ("test" if i in te else "val" if i in val else "train")
                    for i in range(len(ids))})
    return out


def _train_baseline(model, loaders, cfg, device, epochs, contrastive_lambda):
    nc = cfg["task"]["num_classes"]
    task = cfg["task"]["type"]
    opt = build_optimizer(model, cfg["train"]["lr"], cfg["train"]["weight_decay"])
    sched = build_scheduler(opt, epochs, cfg["train"].get("warmup_epochs", 0))
    temp = cfg.get("loss", {}).get("contrastive_temp", 0.1)
    best, best_state, stale = -np.inf, None, 0
    patience = cfg["train"].get("patience", 30)
    for ep in range(epochs):
        model.train()
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            out = model(batch["image"], batch["gene"])
            loss = classification_loss(out["fusion_logits"], batch["label"], task)
            if contrastive_lambda > 0 and out.get("img_proj") is not None \
                    and batch["image"].size(0) > 1:
                loss = loss + contrastive_lambda * info_nce(
                    out["img_proj"], out["gene_proj"], temp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if sched:
            sched.step()
        m, _ = evaluate_split(model, loaders["val"], device, nc)
        score = m["auc"] if not np.isnan(m["auc"]) else m["f1"]
        if score > best:
            best, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def _train_pathgene(model, loaders, cfg, device, epochs):
    nc = cfg["task"]["num_classes"]
    criterion = PathGeneLoss(cfg)
    opt = build_optimizer(model, cfg["train"]["lr"], cfg["train"]["weight_decay"])
    sched = build_scheduler(opt, epochs, cfg["train"].get("warmup_epochs", 0))
    best, best_state, stale = -np.inf, None, 0
    patience = cfg["train"].get("patience", 30)
    for ep in range(epochs):
        model.train()
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            out = model(batch["image"], batch["gene"],
                        gene_pca_target=batch.get("gene_pca_target"),
                        mode="pretrain_jepa")
            loss, _ = criterion(out, batch["label"], include_cls=True,
                                include_jepa=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.update_ema()
        if sched:
            sched.step()
        m, _ = evaluate_split(model, loaders["val"], device, nc)
        score = m["auc"] if not np.isnan(m["auc"]) else m["f1"]
        if score > best:
            best, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _oof(model, loader, device, nc, is_pathgene):
    model.eval()
    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        if is_pathgene:
            out = model(batch["image"], batch["gene"],
                        gene_pca_target=batch.get("gene_pca_target"),
                        mode="downstream")
        else:
            out = model(batch["image"], batch["gene"])
        prob = logits_to_prob(out["fusion_logits"], nc)
        for i in range(prob.shape[0]):
            rows.append((float(prob[i, 1]), int(prob[i].argmax()),
                         int(batch["label"][i].item())))
    return rows


def _metrics(oof):
    p1 = np.array([r[0] for r in oof])
    pred = np.array([r[1] for r in oof])
    y = np.array([r[2] for r in oof])
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
    P = np.stack([1 - p1, p1], 1)
    try:
        auc = float(roc_auc_score(y, p1))
    except Exception:
        auc = float("nan")
    return dict(auc=auc, f1=float(f1_score(y, pred, zero_division=0)),
                acc=float(accuracy_score(y, pred)),
                ece=float(expected_calibration_error(y, P)))


def run_method(disp, kind, o, cfg, ids, labels, device, folds, seeds, epochs, log):
    c = _apply_override(cfg, kind, o)
    per_seed = []
    for seed in seeds:
        c["seed"] = seed
        set_seed(seed)
        assignments = _make_fold_splits(ids, labels, folds, seed)
        oof = []
        for override in assignments:
            datasets, meta = build_datasets(c, override_splits=override)
            loaders = make_loaders(datasets, c)
            nc = meta["num_classes"]
            if kind == "pathgene":
                model = build_pathgene_jepa(c, meta).to(device)
                model = _apply_flags(model, o)
                model = _train_pathgene(model, loaders, c, device, epochs)
                oof += _oof(model, loaders["test"], device, nc, True)
            else:
                model = build_baseline(o["name"], c, meta).to(device)
                clam = c.get("loss", {}).get("lambda_align", 0.1) \
                    if o["name"] == "contrastive" else 0.0
                model = _train_baseline(model, loaders, c, device, epochs, clam)
                oof += _oof(model, loaders["test"], device, nc, False)
        per_seed.append(_metrics(oof))
    agg = {k: [float(np.nanmean([s[k] for s in per_seed])),
               float(np.nanstd([s[k] for s in per_seed]))]
           for k in ("auc", "f1", "acc", "ece")}
    log.info(f"  {disp:38s} AUC {agg['auc'][0]:.3f}+/-{agg['auc'][1]:.3f}  "
             f"F1 {agg['f1'][0]:.3f}  ECE {agg['ece'][0]:.3f}")
    # persist per-seed values for paired significance tests
    persd = os.path.join(_OUT, "tables", "per_seed.csv")
    os.makedirs(os.path.dirname(persd), exist_ok=True)
    write_header = not os.path.exists(persd)
    import csv as _csv
    with open(persd, "a", newline="") as f:
        w = _csv.writer(f)
        if write_header:
            w.writerow(["method", "seed", "auc", "f1", "acc", "ece"])
        for sd, m in zip(seeds, per_seed):
            w.writerow([disp, sd, m["auc"], m["f1"], m["acc"], m["ece"]])
    return {"method": disp, "kind": kind, **agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tcga_brca_pathway.yaml")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 1, 7])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--exp_name", default="brca_compare")
    ap.add_argument("--ablation", action="store_true",
                    help="run the component-incremental ablation instead of the "
                         "cross-method comparison")
    ap.add_argument("--factorial", action="store_true",
                    help="run the full 2^4 factorial ablation over the modules")
    ap.add_argument("--only", nargs="*", default=None,
                    help="substrings to filter methods (e.g. Gene-only CoRE)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "cuda"))
    ensure_dir(os.path.join(_OUT, "tables"))
    ensure_dir(os.path.join(_OUT, "logs"))
    log = Logger(args.exp_name, os.path.join(_OUT, "logs", f"{args.exp_name}.log"))
    banner(f"CV method comparison :: {args.config} :: {device}")

    ds, meta = build_datasets(cfg)
    ids, labels = [], []
    for s in ("train", "val", "test"):
        ids += list(ds[s].ids); labels += list(ds[s].label)
    ids, labels = np.array(ids), np.array(labels)
    log.info(f"{len(ids)} patients, {args.folds}-fold x {len(args.seeds)} seeds")

    method_list = (FACTORIAL if args.factorial else
                   ABLATION if args.ablation else METHODS)
    if args.only:
        method_list = [m for m in method_list
                       if any(s.lower() in m[0].lower() for s in args.only)]
    rows = []
    for disp, kind, o in method_list:
        try:
            rows.append(run_method(disp, kind, o, cfg, ids, labels, device,
                                   args.folds, args.seeds, args.epochs, log))
        except Exception as e:  # noqa
            log.info(f"  {disp}: FAILED ({e})")

    import csv
    csv_path = os.path.join(_OUT, "tables", f"{args.exp_name}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "kind", "auc", "auc_std", "f1", "f1_std",
                    "acc", "acc_std", "ece", "ece_std"])
        for r in rows:
            w.writerow([r["method"], r["kind"], r["auc"][0], r["auc"][1],
                        r["f1"][0], r["f1"][1], r["acc"][0], r["acc"][1],
                        r["ece"][0], r["ece"][1]])
    save_json(rows, os.path.join(_OUT, "tables", f"{args.exp_name}.json"))
    banner(f"saved {csv_path}")


if __name__ == "__main__":
    main()
