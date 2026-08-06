#!/usr/bin/env python
"""Ablation study for JPathGene.

Each configuration trains JPathGene jointly (selected JEPA terms +
classification) and reports test AUC/F1/Acc/ECE. Configurations:

  1. fusion only (no JEPA)
  2. image->gene JEPA only
  3. gene->image JEPA only
  4. bidirectional JEPA
  5. bidirectional JEPA + contrastive alignment
  6. learned gene latent target  (bidir + contrastive)
  7. PCA gene latent target       (bidir + contrastive)
  8. pathway target               (optional; skipped if unavailable)

Writes outputs/tables/ablation_results.csv and .json.
"""
import argparse
import copy
import os

import numpy as np
import torch

from datasets import build_datasets
from models.losses import PathGeneLoss
from models.pathgene_jepa import build_pathgene_jepa
from utils.engine import (build_optimizer, build_scheduler, evaluate_split,
                          make_loaders, move_batch)
from utils.io import ensure_dir, get_device, load_config, save_json
from utils.logger import Logger, banner
from utils.metrics import compute_metrics
from utils.seed import set_seed
from utils.io import OUT_ROOT as _OUT  # namespaced output root

def _ab(name, i2g=True, g2i=True, ctr=True, target="learned",
        predictor="diffusion", fusion="uncertainty_gated"):
    return dict(name=name, i2g=i2g, g2i=g2i, ctr=ctr, target=target,
                predictor=predictor, fusion=fusion)


# The ablation tells the Diffusion-JEPA story: (a) does cross-modal JEPA help at
# all, (b) does a DIFFUSION predictor beat a deterministic MLP predictor, (c)
# does uncertainty-gated fusion beat plain concat, (d) target type / directions.
ABLATIONS = [
    _ab("fusion_only", i2g=False, g2i=False, ctr=False, fusion="concat"),
    _ab("mlp_predictor", predictor="mlp", fusion="concat"),
    _ab("diffusion_predictor", predictor="diffusion", fusion="concat"),
    _ab("diffusion+uncertainty_gate", predictor="diffusion",
        fusion="uncertainty_gated"),
    _ab("img2gene_only", g2i=False, ctr=False),
    _ab("gene2img_only", i2g=False, ctr=False),
    _ab("target=learned", target="learned"),
    _ab("target=pca", target="pca"),
]


def train_eval(cfg, ab, device, epochs, log):
    set_seed(cfg.get("seed", 42))
    c = copy.deepcopy(cfg)
    c["model"]["target"]["gene_target_type"] = ab["target"]
    c["model"]["predictor"]["type"] = ab.get("predictor", "diffusion")
    c["model"]["fusion"]["mode"] = ab.get("fusion", "uncertainty_gated")
    c["loss"]["use_contrastive"] = ab["ctr"]
    if ab["target"] == "pca":
        c["data"]["gene_pca"]["enabled"] = True

    datasets, meta = build_datasets(c)
    loaders = make_loaders(datasets, c)
    model = build_pathgene_jepa(c, meta).to(device)
    model.use_img2gene = ab["i2g"]
    model.use_gene2img = ab["g2i"]
    model.use_contrastive = ab["ctr"]
    criterion = PathGeneLoss(c)

    opt = build_optimizer(model, c["train"]["lr"], c["train"]["weight_decay"])
    sched = build_scheduler(opt, epochs, c["train"].get("warmup_epochs", 0))
    num_classes = meta["num_classes"]
    best_auc, best_state, stale = -np.inf, None, 0
    patience = c["train"].get("patience", 30)

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
        m, _ = evaluate_split(model, loaders["val"], device, num_classes)
        score = m["auc"] if not np.isnan(m["auc"]) else m["f1"]
        if score > best_auc:
            best_auc = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_m, _ = evaluate_split(model, loaders["test"], device, num_classes)
    log.info(f"  {ab['name']:>20s} | test_auc {test_m['auc']:.4f} "
             f"f1 {test_m['f1']:.4f} acc {test_m['accuracy']:.4f} "
             f"ece {test_m['ece']:.4f}")
    return test_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp_name", default="pathgene_ablation")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "cuda"))
    ensure_dir(os.path.join(_OUT, "tables"))
    log = Logger(args.exp_name, os.path.join(os.path.join(_OUT, "logs"), f"{args.exp_name}.log"))
    banner(f"JPathGene ablation :: {device}")
    epochs = args.epochs or cfg["train"]["epochs"]

    rows = []
    for ab in ABLATIONS:
        m = train_eval(cfg, ab, device, epochs, log)
        rows.append({
            "method": ab["name"],
            "predictor": ab.get("predictor", "diffusion"),
            "fusion": ab.get("fusion", "uncertainty_gated"),
            "img2gene_jepa": ab["i2g"], "gene2img_jepa": ab["g2i"],
            "contrastive": ab["ctr"], "gene_target": ab["target"],
            "auc": round(m["auc"], 4), "f1": round(m["f1"], 4),
            "acc": round(m["accuracy"], 4), "ece": round(m["ece"], 4)})

    import csv as csvmod
    csv_path = os.path.join(os.path.join(_OUT, "tables"), "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    save_json(rows, os.path.join(os.path.join(_OUT, "tables"), "ablation_results.json"))
    banner(f"saved {csv_path}")


if __name__ == "__main__":
    main()
