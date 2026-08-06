#!/usr/bin/env python
"""Gene latent prediction-quality analysis for a trained JPathGene model.

Compares the image-predicted gene latent against the true gene latent target:
  * row-wise cosine similarity (predicted vs true)
  * MSE / Smooth-L1 error
  * per-dimension Pearson correlation
  * PCA-latent correlation summary

Outputs go to outputs/analysis and outputs/embeddings (arrays for the scatter
plot are saved so the figure is generated only from real numbers).
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from datasets import build_datasets
from models.pathgene_jepa import build_pathgene_jepa
from utils.engine import make_loaders, move_batch
from utils.gene_utils import pearson_per_dim, rowwise_cosine
from utils.io import ensure_dir, get_device, load_config, save_json
from utils.logger import banner
from utils.io import OUT_ROOT as _OUT  # namespaced output root


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    pred, true, ids = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["image"], batch["gene"],
                    gene_pca_target=batch.get("gene_pca_target"), mode="analysis")
        pred.append(out["pred_gene_full"].cpu().numpy())
        true.append(out["target_gene_latent"].cpu().numpy())
        ids.extend(batch["patient_id"])
    return np.concatenate(pred), np.concatenate(true), np.array(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=os.path.join(_OUT, "checkpoints/best_pathgene_jepa.pth"))
    ap.add_argument("--exp_name", default="pathgene_gene_prediction")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "cuda"))
    ensure_dir(os.path.join(_OUT, "analysis"))
    ensure_dir(os.path.join(_OUT, "embeddings"))
    banner(f"gene prediction analysis :: {args.checkpoint} :: {device}")

    datasets, meta = build_datasets(cfg)
    loaders = make_loaders(datasets, cfg)
    model = build_pathgene_jepa(cfg, meta).to(device)
    if os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
    else:
        print(f"WARNING: checkpoint {args.checkpoint} missing; using random init.")

    pred, true, ids = extract(model, loaders[args.split], device)
    cos = rowwise_cosine(pred, true)
    pear = pearson_per_dim(pred, true)
    mse = float(np.mean((pred - true) ** 2))
    sl1 = float(F.smooth_l1_loss(torch.tensor(pred), torch.tensor(true)))

    metrics = {
        "n": int(len(ids)),
        "target_dim": int(pred.shape[1]),
        "rowwise_cosine_mean": float(cos.mean()),
        "rowwise_cosine_std": float(cos.std()),
        "mse": mse,
        "smooth_l1": sl1,
        "per_dim_pearson_mean": float(np.nanmean(pear)),
        "per_dim_pearson_median": float(np.nanmedian(pear)),
        "frac_dims_pearson_gt_0.1": float(np.mean(pear > 0.1)),
    }
    for k, v in metrics.items():
        print(f"  {k:28s}: {v}")

    save_json(metrics, os.path.join(os.path.join(_OUT, "analysis"),
                                    f"{args.exp_name}_metrics.json"))
    np.savez(os.path.join(os.path.join(_OUT, "embeddings"), f"{args.exp_name}_arrays.npz"),
             pred=pred, true=true, ids=ids, rowwise_cosine=cos, per_dim_pearson=pear)

    # quality table
    import csv
    ensure_dir(os.path.join(_OUT, "tables"))
    with open(os.path.join(os.path.join(_OUT, "tables"), "gene_prediction_quality.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])
    banner("gene prediction analysis done")


if __name__ == "__main__":
    main()
