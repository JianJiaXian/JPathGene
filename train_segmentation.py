#!/usr/bin/env python
"""Train the gene-conditioned U-Net for TCGA-LGG radiogenomic segmentation.

Gene modes:
  none : image-only U-Net (baseline)
  true : FiLM-condition on the real genomic profile (needs genetics at test)
  jepa : FiLM-condition on the image-imputed genetic latent; an image->gene
         JEPA loss aligns the imputed latent with the real one, so segmentation
         is genomics-guided yet genomics-OPTIONAL at test time.

  python train_segmentation.py --config configs/tcga_lgg_seg.yaml \
      --gene_mode jepa --exp_name lgg_jepa
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from datasets.lgg_radiogenomics_dataset import build_lgg_datasets
from models.segmentation import GeneConditionedUNet, dice_iou_per, seg_loss
from models.seg_backbones import AttentionUNet, TransUNet
from models.diffusion_seg import DiffusionSeg
from utils.engine import make_loaders, move_batch


def build_seg_model(arch, meta, cfg):
    gd = max(1, meta["gene_dim"])
    gl = cfg["model"].get("gene_latent", 128)
    base = cfg["model"].get("base", 32)
    drop = cfg["model"].get("dropout", 0.2)
    arch = arch.lower()
    if arch == "unet":
        return GeneConditionedUNet(3, 1, gd, gl, base, drop)
    if arch == "attunet":
        return AttentionUNet(3, 1, gd, gl, base, drop)
    if arch == "transunet":
        return TransUNet(3, 1, gd, gl, base, drop,
                         image_size=cfg["data"].get("image_size", 256))
    if arch == "diffseg":
        return DiffusionSeg(3, 1, gd, gl, base, drop,
                            sample_steps=cfg["model"].get("sample_steps", 10))
    raise ValueError(f"unknown arch '{arch}'")
from utils.io import ensure_dir, get_device, load_config, save_json
from utils.logger import Logger, MetricCSVLogger, banner
from utils.seed import set_seed
from utils.io import OUT_ROOT as _OUT


def evaluate(model, loader, device, gene_mode, ensemble=1):
    """Dice/IoU averaged over TUMOR-BEARING slices only (the meaningful metric for
    LGG, where most slices are tumor-free and an empty mask would otherwise score
    ~0.68). Also reports the all-slice average for reference."""
    model.eval()
    is_diff = getattr(model, "is_diffusion", False)
    dt, it, nt = 0.0, 0.0, 0      # tumor slices
    da, ia, na = 0.0, 0.0, 0      # all slices
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            if is_diff:
                out = model.segment(batch["image"], batch["gene"],
                                    gene_mode=gene_mode, n_ensemble=ensemble)
            else:
                out = model(batch["image"], batch["gene"], gene_mode=gene_mode)
            d, i, has = dice_iou_per(out["logits"], batch["mask"])
            da += d.sum().item(); ia += i.sum().item(); na += d.numel()
            if has.any():
                dt += d[has].sum().item(); it += i[has].sum().item()
                nt += int(has.sum().item())
    return {"dice": dt / max(1, nt), "iou": it / max(1, nt),
            "dice_allslices": da / max(1, na), "iou_allslices": ia / max(1, na)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp_name", default="lgg_seg")
    ap.add_argument("--arch", default="unet",
                    choices=["unet", "attunet", "transunet", "diffseg"])
    ap.add_argument("--gene_mode", default=None, choices=[None, "none", "true", "jepa"])
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "cuda"))
    for d in ("checkpoints", "logs", "analysis", "predictions", "tables"):
        ensure_dir(os.path.join(_OUT, d))
    gene_mode = args.gene_mode or cfg["train"].get("gene_mode", "jepa")
    log = Logger(args.exp_name, os.path.join(_OUT, "logs", f"{args.exp_name}.log"))
    banner(f"LGG seg :: arch={args.arch} gene_mode={gene_mode} :: "
           f"{args.exp_name} :: {device}")

    datasets, meta = build_lgg_datasets(cfg)
    log.info(f"meta: {meta}")
    loaders = make_loaders(datasets, cfg)

    model = build_seg_model(args.arch, meta, cfg).to(device)
    is_diff = getattr(model, "is_diffusion", False)

    epochs = args.epochs or cfg["train"]["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    lam_jepa = cfg["train"].get("lambda_jepa", 1.0) if gene_mode == "jepa" else 0.0

    csv = MetricCSVLogger(os.path.join(_OUT, "logs", f"{args.exp_name}_metrics.csv"),
                          ["epoch", "train_loss", "val_dice", "val_iou", "lr"])
    best, stale = -np.inf, 0
    patience = cfg["train"].get("patience", 12)
    best_path = os.path.join(_OUT, "checkpoints", f"best_{args.exp_name}.pth")

    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            with torch.autocast("cuda", enabled=scaler is not None):
                if is_diff:
                    loss, gene_hat, _ = model.loss(
                        batch["image"], batch["mask"], batch["gene"], gene_mode)
                else:
                    out = model(batch["image"], batch["gene"], gene_mode=gene_mode)
                    loss = seg_loss(out["logits"], batch["mask"])
                    gene_hat = out["gene_hat"]
                if lam_jepa > 0:
                    has_gene = (batch["gene"].abs().sum(1) > 0)
                    if has_gene.any():
                        tgt = model.gene_encoder(batch["gene"]).detach()
                        jl = F.mse_loss(gene_hat[has_gene], tgt[has_gene])
                        loss = loss + lam_jepa * jl
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            tot += float(loss.detach()) * batch["image"].size(0)
            n += batch["image"].size(0)
        sched.step()
        val = evaluate(model, loaders["val"], device, gene_mode)
        csv.log({"epoch": ep, "train_loss": tot / max(1, n),
                 "val_dice": val["dice"], "val_iou": val["iou"],
                 "lr": opt.param_groups[0]["lr"]})
        log.info(f"ep {ep:03d} | loss {tot/max(1,n):.4f} | "
                 f"val_dice {val['dice']:.4f} iou {val['iou']:.4f}")
        if val["dice"] > best:
            best, stale = val["dice"], 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "meta": meta,
                        "gene_mode": gene_mode}, best_path)
        else:
            stale += 1
            if stale >= patience:
                log.info(f"early stop at {ep}")
                break

    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    # ensemble multiple DDIM samples at test for the diffusion segmenter (STAPLE-like)
    test_ens = getattr(model, "eval_ensemble", 1) if is_diff else 1
    test = evaluate(model, loaders["test"], device, gene_mode, ensemble=test_ens)
    log.info(f"TEST gene_mode={gene_mode} (ensemble={test_ens}): {test}")
    save_json({"exp_name": args.exp_name, "gene_mode": gene_mode,
               "test": test, "meta": meta},
              os.path.join(_OUT, "analysis", f"{args.exp_name}_test.json"))
    banner(f"done. test_dice={test['dice']:.4f} iou={test['iou']:.4f}")


if __name__ == "__main__":
    main()
