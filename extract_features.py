#!/usr/bin/env python
"""Dump JPathGene latent embeddings for all splits.

For each patient writes: image_latent, gene_latent, predicted_gene_latent,
predicted_image_latent, target_gene_latent. Useful for downstream linear probes
and for the UMAP / scatter figures.

NOTE on raw WSI features: this first version operates on PRE-EXTRACTED image
features (image_features.csv). Extracting patch features from raw whole-slide
images (e.g. with a frozen pathology foundation encoder + MIL) is left as future
work; the FeatureMLP/MIL hooks in models/image_encoder.py are ready for it.
"""
import argparse
import os

import numpy as np
import torch

from datasets import build_datasets
from models.pathgene_jepa import build_pathgene_jepa
from utils.engine import make_loaders, move_batch
from utils.io import ensure_dir, get_device, load_config
from utils.logger import banner
from utils.io import OUT_ROOT as _OUT  # namespaced output root


@torch.no_grad()
def dump(model, loader, device):
    model.eval()
    acc = {k: [] for k in ["image_latent", "gene_latent", "pred_gene_full",
                           "pred_img_full", "target_gene_latent"]}
    y, ids = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["image"], batch["gene"],
                    gene_pca_target=batch.get("gene_pca_target"), mode="analysis")
        for k in acc:
            acc[k].append(out[k].cpu().numpy())
        y.append(batch["label"].cpu().numpy())
        ids.extend(batch["patient_id"])
    res = {k: np.concatenate(v) for k, v in acc.items()}
    res["y"] = np.concatenate(y)
    res["ids"] = np.array(ids)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=os.path.join(_OUT, "checkpoints/best_pathgene_jepa.pth"))
    ap.add_argument("--exp_name", default="pathgene_embeddings")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "cuda"))
    ensure_dir(os.path.join(_OUT, "embeddings"))
    banner(f"extract features :: {args.checkpoint} :: {device}")
    datasets, meta = build_datasets(cfg)
    loaders = make_loaders(datasets, cfg)
    model = build_pathgene_jepa(cfg, meta).to(device)
    if os.path.exists(args.checkpoint):
        st = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(st["model"], strict=False)
    else:
        print(f"WARNING: checkpoint {args.checkpoint} missing; random init.")

    for split in ("train", "val", "test"):
        res = dump(model, loaders[split], device)
        path = os.path.join(os.path.join(_OUT, "embeddings"), f"{args.exp_name}_{split}.npz")
        np.savez(path, **res)
        print(f"  saved {path}  (n={len(res['ids'])})")
    banner("extract features done")


if __name__ == "__main__":
    main()
