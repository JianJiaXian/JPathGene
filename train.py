#!/usr/bin/env python
"""Unified supervised trainer for baselines (and the fine-tuning backbone).

    python train.py --config configs/feature_csv.yaml --model image_only --exp_name image_only
    python train.py --config configs/feature_csv.yaml --model concat_fusion --exp_name concat_fusion

Models: image_only | gene_only | early_fusion | late_fusion | concat_fusion |
        gated_fusion | contrastive | pathgene_jepa
"""
import argparse
import os

import numpy as np
import torch

from datasets import build_datasets
from models.baselines import build_baseline
from models.contrastive import info_nce
from models.losses import classification_loss
from utils.engine import (build_optimizer, build_scheduler, evaluate_split,
                          make_loaders, move_batch, save_checkpoint)
from utils.io import ensure_dir, get_device, load_config, save_json
from utils.logger import Logger, MetricCSVLogger, banner
from utils.seed import set_seed
from utils.io import OUT_ROOT as _OUT  # namespaced output root

CKPT_DIR = os.path.join(_OUT, "checkpoints")
LOG_DIR = os.path.join(_OUT, "logs")
PRED_DIR = os.path.join(_OUT, "predictions")


def train_classifier(model, loaders, cfg, device, exp_name, epochs=None, lr=None,
                     log=None, contrastive_lambda=0.0, meta=None):
    task = cfg.get("task", {}).get("type", "binary")
    num_classes = cfg["task"]["num_classes"]
    epochs = epochs or cfg["train"]["epochs"]
    lr = lr or cfg["train"]["lr"]
    opt = build_optimizer(model, lr, cfg["train"]["weight_decay"])
    sched = build_scheduler(opt, epochs, cfg["train"].get("warmup_epochs", 0),
                            cfg["train"].get("scheduler", "cosine"))
    use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    temp = cfg.get("loss", {}).get("contrastive_temp", 0.1)

    csv = MetricCSVLogger(os.path.join(LOG_DIR, f"{exp_name}_metrics.csv"),
                          ["epoch", "train_loss", "val_auc", "val_f1", "val_acc",
                           "lr"])
    best, stale = -np.inf, 0
    patience = cfg["train"].get("patience", 30)
    best_path = os.path.join(CKPT_DIR, f"best_{exp_name}.pth")
    is_pathgene = model.__class__.__name__ == "PathGeneJEPA"

    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            bs = batch["image"].size(0)
            with torch.autocast("cuda", enabled=scaler is not None):
                if is_pathgene:
                    out = model(batch["image"], batch["gene"],
                                gene_pca_target=batch.get("gene_pca_target"),
                                mode="downstream")
                else:
                    out = model(batch["image"], batch["gene"])
                loss = classification_loss(out["fusion_logits"], batch["label"],
                                           task)
                if contrastive_lambda > 0 and out.get("img_proj") is not None \
                        and bs > 1:
                    loss = loss + contrastive_lambda * info_nce(
                        out["img_proj"], out["gene_proj"], temp)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            tot += float(loss.detach()) * bs
            n += bs
        if sched:
            sched.step()
        train_loss = tot / max(1, n)
        m, _ = evaluate_split(model, loaders["val"], device, num_classes)
        csv.log({"epoch": ep, "train_loss": train_loss, "val_auc": m["auc"],
                 "val_f1": m["f1"], "val_acc": m["accuracy"],
                 "lr": opt.param_groups[0]["lr"]})
        if log:
            log.info(f"ep {ep:03d} | loss {train_loss:.4f} | "
                     f"val_auc {m['auc']:.4f} f1 {m['f1']:.4f} acc {m['accuracy']:.4f}")
        score = m["auc"] if not np.isnan(m["auc"]) else m["f1"]
        if score > best:
            best, stale = score, 0
            save_checkpoint(best_path, model, cfg, meta or {},
                            extra={"epoch": ep, "val": m})
        else:
            stale += 1
            if stale >= patience:
                if log:
                    log.info(f"early stop at {ep}")
                break
    return best_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "cuda"))
    for d in (CKPT_DIR, LOG_DIR, PRED_DIR, os.path.join(_OUT, "analysis")):
        ensure_dir(d)
    log = Logger(args.exp_name, os.path.join(LOG_DIR, f"{args.exp_name}.log"))
    banner(f"train {args.model} :: {args.exp_name} :: device={device}")

    datasets, meta = build_datasets(cfg)
    log.info(f"meta: {meta}")
    loaders = make_loaders(datasets, cfg)
    model = build_baseline(args.model, cfg, meta).to(device)

    clam = cfg.get("loss", {}).get("lambda_align", 0.1) if args.model == "contrastive" else 0.0
    best_path = train_classifier(model, loaders, cfg, device, args.exp_name,
                                 epochs=args.epochs, log=log,
                                 contrastive_lambda=clam, meta=meta)

    # final test evaluation from best checkpoint
    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    metrics, (y, p, ids) = evaluate_split(model, loaders["test"], device,
                                          meta["num_classes"])
    log.info(f"TEST {args.model}: {metrics}")
    save_json({"model": args.model, "exp_name": args.exp_name,
               "test_metrics": metrics, "meta": meta},
              os.path.join(os.path.join(_OUT, "analysis"), f"{args.exp_name}_test.json"))
    np.savez(os.path.join(PRED_DIR, f"{args.exp_name}_preds.npz"),
             y_true=y, y_prob=p, patient_id=np.array(ids))
    banner(f"done. test_auc={metrics['auc']:.4f} f1={metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
