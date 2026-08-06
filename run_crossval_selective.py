#!/usr/bin/env python
"""Leakage-safe K-fold cross-validation with out-of-fold (OOF) pooling, used to
(a) give multi-fold statistical rigor and (b) test the headline claim that the
diffusion predictor's uncertainty enables *trustworthy cross-modal substitution*:
when histology can stand in for genomics and when the model should abstain.

For every patient we obtain a held-out: downstream class probability, label,
image->gene prediction cosine, and per-patient diffusion uncertainty. We then
measure calibration (ECE), selective classification (risk-coverage by confidence)
and, crucially, whether uncertainty predicts gene-prediction quality
(uncertainty vs. cosine correlation; low- vs high-uncertainty cosine).

  python run_crossval_selective.py --config configs/tcga_brca.yaml \
      --folds 5 --epochs 80 --exp_name brca_cv
"""
import argparse
import copy
import os

import numpy as np
import torch

from datasets.image_gene_feature_dataset import build_datasets
from models.losses import PathGeneLoss
from models.pathgene_jepa import build_pathgene_jepa
from utils.engine import (build_optimizer, build_scheduler, evaluate_split,
                          logits_to_prob, make_loaders, move_batch)
from utils.gene_utils import rowwise_cosine
from utils.io import ensure_dir, get_device, load_config, save_json
from utils.logger import Logger, banner
from utils.metrics import expected_calibration_error
from utils.seed import set_seed
from utils.io import OUT_ROOT as _OUT


def _all_ids_labels(cfg):
    ds, meta = build_datasets(cfg)  # any split; we only need ids+labels+pids
    ids, labels = [], []
    for s in ("train", "val", "test"):
        ids += list(ds[s].ids)
        labels += list(ds[s].label)
    return np.array(ids), np.array(labels), meta


def _make_fold_splits(ids, labels, folds, seed):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rng = np.random.RandomState(seed)
    assignments = []
    for tr_idx, te_idx in skf.split(ids, labels):
        # carve a small stratified val from the train portion
        tr_idx = list(tr_idx)
        rng.shuffle(tr_idx)
        n_val = max(2, int(0.15 * len(tr_idx)))
        val_idx = set(tr_idx[:n_val])
        override = {}
        for i in range(len(ids)):
            if i in set(te_idx):
                override[ids[i]] = "test"
            elif i in val_idx:
                override[ids[i]] = "val"
            else:
                override[ids[i]] = "train"
        assignments.append(override)
    return assignments


def train_fold(cfg, override, device, epochs):
    set_seed(cfg.get("seed", 42))
    datasets, meta = build_datasets(cfg, override_splits=override)
    loaders = make_loaders(datasets, cfg)
    model = build_pathgene_jepa(cfg, meta).to(device)
    criterion = PathGeneLoss(cfg)
    opt = build_optimizer(model, cfg["train"]["lr"], cfg["train"]["weight_decay"])
    sched = build_scheduler(opt, epochs, cfg["train"].get("warmup_epochs", 0))
    nc = meta["num_classes"]
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
    return model, loaders, meta


@torch.no_grad()
def collect_oof(model, loader, device, nc):
    model.eval()
    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["image"], batch["gene"],
                    gene_pca_target=batch.get("gene_pca_target"), mode="analysis")
        prob = logits_to_prob(out["fusion_logits"], nc)  # (B, C)
        gcos = rowwise_cosine(out["pred_gene_full"].cpu().numpy(),
                              out["target_gene_latent"].cpu().numpy())
        unc = out.get("gene_uncertainty")
        unc = (unc.cpu().numpy() if unc is not None
               else np.zeros(prob.shape[0]))
        pg = out["pred_gene_full"].cpu().numpy()
        tg = out["target_gene_latent"].cpu().numpy()
        for i in range(prob.shape[0]):
            rows.append(dict(pid=batch["patient_id"][i],
                             prob1=float(prob[i, 1]),
                             pred=int(prob[i].argmax()),
                             label=int(batch["label"][i].item()),
                             gene_cos=float(gcos[i]),
                             uncertainty=float(unc[i]),
                             pred_gene=pg[i], true_gene=tg[i]))
    return rows


def selective_analysis(oof, pathway_names=None):
    y = np.array([r["label"] for r in oof])
    pred = np.array([r["pred"] for r in oof])
    p1 = np.array([r["prob1"] for r in oof])
    conf = np.maximum(p1, 1 - p1)
    unc = np.array([r["uncertainty"] for r in oof])
    cos = np.array([r["gene_cos"] for r in oof])
    correct = (pred == y).astype(float)

    P = np.stack([1 - p1, p1], 1)
    res = {"n": len(y), "ece": expected_calibration_error(y, P),
           "accuracy": float(correct.mean())}
    try:
        from sklearn.metrics import roc_auc_score
        res["auc"] = float(roc_auc_score(y, p1))
    except Exception:
        res["auc"] = float("nan")

    # per-target (per-pathway) correlation across patients: does predicted
    # pathway activity track the true activity? (the biological cross-modal test)
    try:
        PG = np.stack([r["pred_gene"] for r in oof])
        TG = np.stack([r["true_gene"] for r in oof])
        per_dim = []
        for j in range(PG.shape[1]):
            if PG[:, j].std() > 1e-8 and TG[:, j].std() > 1e-8:
                per_dim.append(float(np.corrcoef(PG[:, j], TG[:, j])[0, 1]))
            else:
                per_dim.append(float("nan"))
        names = pathway_names if (pathway_names and len(pathway_names) == len(per_dim)) \
            else [f"dim{j}" for j in range(len(per_dim))]
        res["per_target_pearson"] = {n: v for n, v in zip(names, per_dim)}
        res["per_target_pearson_mean"] = float(np.nanmean(per_dim))
    except Exception:
        pass

    # does uncertainty predict gene-prediction quality? (key claim)
    if unc.std() > 1e-8:
        res["corr_uncertainty_genecos"] = float(np.corrcoef(unc, cos)[0, 1])
        order = np.argsort(unc)
        half = len(unc) // 2
        res["genecos_low_uncertainty"] = float(cos[order[:half]].mean())
        res["genecos_high_uncertainty"] = float(cos[order[half:]].mean())

    # risk-coverage by confidence (selective classification)
    rc = []
    order = np.argsort(-conf)
    for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        k = max(1, int(cov * len(y)))
        idx = order[:k]
        rc.append({"coverage": cov, "accuracy": float(correct[idx].mean())})
    res["risk_coverage_confidence"] = rc

    # selective gene-substitution: abstain on most-uncertain, cosine of the rest
    rcg = []
    order_u = np.argsort(unc)
    for cov in [1.0, 0.8, 0.6, 0.4, 0.2]:
        k = max(1, int(cov * len(unc)))
        rcg.append({"coverage": cov,
                    "gene_cos": float(cos[order_u[:k]].mean())})
    res["substitution_coverage"] = rcg
    return res, dict(unc=unc, cos=cos, conf=conf, correct=correct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42])
    ap.add_argument("--exp_name", default="cv_selective")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "cuda"))
    ensure_dir(os.path.join(_OUT, "analysis"))
    ensure_dir(os.path.join(_OUT, "tables"))
    ensure_dir(os.path.join(_OUT, "embeddings"))
    log = Logger(args.exp_name, os.path.join(_OUT, "logs", f"{args.exp_name}.log"))
    banner(f"CV + selective prediction :: {args.config} :: {device}")

    ids, labels, meta0 = _all_ids_labels(cfg)
    pathway_names = meta0.get("pathway_names", [])
    log.info(f"{len(ids)} patients, {args.folds}-fold CV, seeds={args.seeds}, "
             f"targets={pathway_names or 'latent'}")

    per_seed = []
    pooled_arrays = []
    for seed in args.seeds:
        cfg["seed"] = seed
        assignments = _make_fold_splits(ids, labels, args.folds, seed)
        oof = []
        for f, override in enumerate(assignments):
            model, loaders, meta = train_fold(cfg, override, device, args.epochs)
            fold_rows = collect_oof(model, loaders["test"], device,
                                    meta["num_classes"])
            oof += fold_rows
            log.info(f"  seed {seed} fold {f}: {len(fold_rows)} held-out patients")
        res, arrs = selective_analysis(oof, pathway_names)
        per_seed.append(res)
        pooled_arrays.append(arrs)
        log.info(f"seed {seed}: auc={res['auc']:.3f} ece={res['ece']:.3f} "
                 f"per-pathway Pearson={res.get('per_target_pearson', {})}")

    # aggregate across seeds
    def agg(key):
        vals = [r[key] for r in per_seed if key in r and not
                (isinstance(r[key], float) and np.isnan(r[key]))]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), 0.0)

    # aggregate per-pathway Pearson across seeds
    pathway_agg = {}
    if per_seed and per_seed[0].get("per_target_pearson"):
        for name in per_seed[0]["per_target_pearson"]:
            vals = [r["per_target_pearson"][name] for r in per_seed
                    if not np.isnan(r["per_target_pearson"].get(name, float("nan")))]
            if vals:
                pathway_agg[name] = [float(np.mean(vals)), float(np.std(vals))]

    summary = {"config": args.config, "folds": args.folds, "seeds": args.seeds,
               "auc_mean_std": agg("auc"), "ece_mean_std": agg("ece"),
               "per_pathway_pearson_mean_std": pathway_agg,
               "per_target_pearson_mean": agg("per_target_pearson_mean"),
               "corr_unc_genecos_mean_std": agg("corr_uncertainty_genecos"),
               "genecos_low_unc_mean_std": agg("genecos_low_uncertainty"),
               "genecos_high_unc_mean_std": agg("genecos_high_uncertainty"),
               "per_seed": per_seed}
    save_json(summary, os.path.join(_OUT, "analysis", f"{args.exp_name}_summary.json"))
    np.savez(os.path.join(_OUT, "embeddings", f"{args.exp_name}_oof.npz"),
             unc=np.concatenate([a["unc"] for a in pooled_arrays]),
             cos=np.concatenate([a["cos"] for a in pooled_arrays]),
             conf=np.concatenate([a["conf"] for a in pooled_arrays]),
             correct=np.concatenate([a["correct"] for a in pooled_arrays]))
    banner(f"AUC {summary['auc_mean_std']} | ECE {summary['ece_mean_std']} | "
           f"corr(unc,genecos) {summary['corr_unc_genecos_mean_std']}")


if __name__ == "__main__":
    main()
