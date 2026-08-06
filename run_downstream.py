#!/usr/bin/env python
"""Downstream training from a pretrained JPathGene checkpoint.

Loads JEPA-pretrained encoders, then trains the fusion classification head.
  --mode linear_probe : freeze encoders, train only the fusion head.
  --mode finetune     : train the whole model end-to-end at a smaller LR.
Saves the best-val checkpoint to outputs/checkpoints/best_pathgene_downstream.pth
"""
import argparse
import os

import numpy as np
import torch

from datasets import build_datasets
from models.pathgene_jepa import build_pathgene_jepa
from train import train_classifier
from utils.engine import evaluate_split, make_loaders
from utils.io import ensure_dir, get_device, load_config, save_json
from utils.logger import Logger, banner
from utils.seed import set_seed
from utils.io import OUT_ROOT as _OUT  # namespaced output root

CKPT_DIR = os.path.join(_OUT, "checkpoints")
LOG_DIR = os.path.join(_OUT, "logs")
PRED_DIR = os.path.join(_OUT, "predictions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pretrained", default=os.path.join(_OUT, "checkpoints/best_pathgene_jepa.pth"))
    ap.add_argument("--exp_name", default="pathgene_downstream")
    ap.add_argument("--mode", default=None, choices=[None, "linear_probe", "finetune"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "cuda"))
    for d in (CKPT_DIR, LOG_DIR, PRED_DIR, os.path.join(_OUT, "analysis")):
        ensure_dir(d)
    log = Logger(args.exp_name, os.path.join(LOG_DIR, f"{args.exp_name}.log"))
    mode = args.mode or cfg.get("downstream", {}).get("mode", "finetune")
    banner(f"JPathGene downstream ({mode}) :: {args.exp_name} :: {device}")

    datasets, meta = build_datasets(cfg)
    loaders = make_loaders(datasets, cfg)
    model = build_pathgene_jepa(cfg, meta).to(device)

    if os.path.exists(args.pretrained):
        state = torch.load(args.pretrained, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        log.info(f"loaded pretrained {args.pretrained} "
                 f"(missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        log.info(f"WARNING: pretrained checkpoint {args.pretrained} not found; "
                 f"training downstream from scratch.")

    if mode == "linear_probe":
        for name, p in model.named_parameters():
            if not name.startswith("fusion"):
                p.requires_grad_(False)
        log.info("linear probe: encoders frozen, training fusion head only")

    dcfg = cfg.get("downstream", {})
    best_path = train_classifier(
        model, loaders, cfg, device, args.exp_name,
        epochs=dcfg.get("epochs", cfg["train"]["epochs"]),
        lr=dcfg.get("lr", cfg["train"]["lr"]), log=log, meta=meta)

    # rename best to the canonical downstream checkpoint path
    canonical = os.path.join(CKPT_DIR, "best_pathgene_downstream.pth")
    if best_path != canonical and os.path.exists(best_path):
        import shutil
        shutil.copyfile(best_path, canonical)

    state = torch.load(canonical, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    metrics, (y, p, ids) = evaluate_split(model, loaders["test"], device,
                                          meta["num_classes"])
    log.info(f"TEST pathgene_jepa: {metrics}")
    save_json({"model": "pathgene_jepa", "mode": mode, "exp_name": args.exp_name,
               "test_metrics": metrics, "meta": meta},
              os.path.join(os.path.join(_OUT, "analysis"), f"{args.exp_name}_test.json"))
    np.savez(os.path.join(PRED_DIR, f"{args.exp_name}_preds.npz"),
             y_true=y, y_prob=p, patient_id=np.array(ids))
    banner(f"done. test_auc={metrics['auc']:.4f} f1={metrics['f1']:.4f} "
           f"checkpoint={canonical}")


if __name__ == "__main__":
    main()
