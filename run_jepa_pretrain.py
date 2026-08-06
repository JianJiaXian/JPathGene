#!/usr/bin/env python
"""JEPA-style pretraining of JPathGene.

Trains the cross-modal latent-prediction objective (image->gene, gene->image)
plus optional contrastive alignment and a joint classification term, logging
every loss component separately. Saves the best checkpoint to
outputs/checkpoints/best_pathgene_jepa.pth.
"""
import argparse
import os

import numpy as np
import torch

from datasets import build_datasets
from models.losses import PathGeneLoss
from models.pathgene_jepa import build_pathgene_jepa
from utils.engine import (build_optimizer, build_scheduler, evaluate_split,
                          make_loaders, move_batch, save_checkpoint)
from utils.io import ensure_dir, get_device, load_config, save_json
from utils.logger import Logger, MetricCSVLogger, banner
from utils.seed import set_seed
from utils.io import OUT_ROOT as _OUT  # namespaced output root

CKPT_DIR = os.path.join(_OUT, "checkpoints")
LOG_DIR = os.path.join(_OUT, "logs")


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              include_cls=True):
    train = optimizer is not None
    model.train(train)
    agg = {}
    n = 0
    for batch in loader:
        batch = move_batch(batch, device)
        bs = batch["image"].size(0)
        with torch.set_grad_enabled(train), \
                torch.autocast("cuda", enabled=scaler is not None):
            out = model(batch["image"], batch["gene"],
                        gene_pca_target=batch.get("gene_pca_target"),
                        mode="pretrain_jepa")
            loss, terms = criterion(out, batch["label"],
                                    include_cls=include_cls, include_jepa=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            model.update_ema()
        for k, v in terms.items():
            agg[k] = agg.get(k, 0.0) + v * bs
        n += bs
    return {k: v / max(1, n) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp_name", default="pathgene_jepa_pretrain")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "cuda"))
    ensure_dir(CKPT_DIR)
    ensure_dir(LOG_DIR)
    log = Logger(args.exp_name, os.path.join(LOG_DIR, f"{args.exp_name}.log"))
    banner(f"JPathGene pretraining :: {args.exp_name} :: device={device}")

    datasets, meta = build_datasets(cfg)
    log.info(f"meta: {meta}")
    loaders = make_loaders(datasets, cfg)

    model = build_pathgene_jepa(cfg, meta).to(device)
    criterion = PathGeneLoss(cfg)
    epochs = args.epochs or cfg["train"]["epochs"]
    opt = build_optimizer(model, cfg["train"]["lr"], cfg["train"]["weight_decay"])
    sched = build_scheduler(opt, epochs, cfg["train"].get("warmup_epochs", 0),
                            cfg["train"].get("scheduler", "cosine"))
    use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    include_cls = cfg.get("loss", {}).get("lambda_cls", 1.0) > 0
    num_classes = meta["num_classes"]
    csv = MetricCSVLogger(
        os.path.join(LOG_DIR, f"{args.exp_name}_metrics.csv"),
        ["epoch", "train_total", "train_img_to_gene", "train_gene_to_img",
         "train_contrastive", "train_classification", "val_total", "val_auc",
         "val_f1", "lr"])

    best = -np.inf  # by val AUC if cls, else by -val_total
    best_path = os.path.join(CKPT_DIR, "best_pathgene_jepa.pth")
    patience = cfg["train"].get("patience", 30)
    stale = 0

    for ep in range(epochs):
        tr = run_epoch(model, loaders["train"], criterion, device, opt, scaler,
                       include_cls)
        va = run_epoch(model, loaders["val"], criterion, device, None, None,
                       include_cls)
        if sched:
            sched.step()
        lr = opt.param_groups[0]["lr"]

        val_auc, val_f1 = float("nan"), float("nan")
        if include_cls:
            m, _ = evaluate_split(model, loaders["val"], device, num_classes)
            val_auc, val_f1 = m["auc"], m["f1"]
            score = val_auc if not np.isnan(val_auc) else -va["total"]
        else:
            score = -va["total"]

        csv.log({"epoch": ep, "train_total": tr["total"],
                 "train_img_to_gene": tr["img_to_gene"],
                 "train_gene_to_img": tr["gene_to_img"],
                 "train_contrastive": tr["contrastive"],
                 "train_classification": tr["classification"],
                 "val_total": va["total"], "val_auc": val_auc,
                 "val_f1": val_f1, "lr": lr})
        log.info(f"ep {ep:03d} | tr_total {tr['total']:.4f} "
                 f"i2g {tr['img_to_gene']:.4f} g2i {tr['gene_to_img']:.4f} "
                 f"ctr {tr['contrastive']:.4f} cls {tr['classification']:.4f} "
                 f"| val_total {va['total']:.4f} val_auc {val_auc:.4f}")

        if score > best:
            best = score
            stale = 0
            save_checkpoint(best_path, model, cfg, meta,
                            extra={"epoch": ep, "score": float(score)})
            log.info(f"  -> saved best ({score:.4f}) to {best_path}")
        else:
            stale += 1
            if stale >= patience:
                log.info(f"early stop at epoch {ep}")
                break

    save_json({"best_score": float(best), "epochs": epochs, "meta": meta},
              os.path.join(os.path.join(_OUT, "analysis"), f"{args.exp_name}_summary.json"))
    banner(f"done. best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
