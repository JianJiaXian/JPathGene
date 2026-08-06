#!/usr/bin/env python
"""Image-gene alignment analysis for a trained JPathGene model.

Computes:
  1. image->gene retrieval (Recall@1/5/10, median rank)
  2. gene->image retrieval
  3. same-patient vs different-patient cosine similarity distributions
  4. saves image/gene embeddings for UMAP/t-SNE plotting

Outputs go to outputs/analysis and outputs/embeddings.
"""
import argparse
import os

import numpy as np
import torch

from datasets import build_datasets
from models.pathgene_jepa import build_pathgene_jepa
from utils.engine import make_loaders, move_batch
from utils.gene_utils import cosine_sim_matrix, retrieval_recall_at_k
from utils.io import ensure_dir, get_device, load_config, save_json
from utils.logger import banner
from utils.io import OUT_ROOT as _OUT  # namespaced output root


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    img, gene, ipro, gpro, y, ids = [], [], [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["image"], batch["gene"],
                    gene_pca_target=batch.get("gene_pca_target"), mode="analysis")
        img.append(out["image_latent"].cpu().numpy())
        gene.append(out["gene_latent"].cpu().numpy())
        if out.get("img_proj") is not None:
            ipro.append(out["img_proj"].cpu().numpy())
            gpro.append(out["gene_proj"].cpu().numpy())
        y.append(batch["label"].cpu().numpy())
        ids.extend(batch["patient_id"])
    R = dict(image_latent=np.concatenate(img), gene_latent=np.concatenate(gene),
             y=np.concatenate(y), ids=np.array(ids))
    if ipro:
        R["img_proj"] = np.concatenate(ipro)
        R["gene_proj"] = np.concatenate(gpro)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=os.path.join(_OUT, "checkpoints/best_pathgene_jepa.pth"))
    ap.add_argument("--exp_name", default="pathgene_alignment")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "cuda"))
    ensure_dir(os.path.join(_OUT, "analysis"))
    ensure_dir(os.path.join(_OUT, "embeddings"))
    ensure_dir(os.path.join(_OUT, "tables"))
    banner(f"alignment analysis :: {args.checkpoint} :: {device}")

    datasets, meta = build_datasets(cfg)
    loaders = make_loaders(datasets, cfg)
    model = build_pathgene_jepa(cfg, meta).to(device)
    if os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
    else:
        print(f"WARNING: checkpoint {args.checkpoint} missing; using random init.")

    R = extract(model, loaders[args.split], device)
    # use contrastive projections when available, else raw latents
    A = R.get("img_proj", R["image_latent"])
    B = R.get("gene_proj", R["gene_latent"])

    sim = cosine_sim_matrix(A, B)              # rows=image, cols=gene
    i2g = retrieval_recall_at_k(sim, ks=(1, 5, 10))
    g2i = retrieval_recall_at_k(sim.T, ks=(1, 5, 10))

    n = sim.shape[0]
    same = np.diag(sim)
    off = sim[~np.eye(n, dtype=bool)]
    sim_stats = {
        "same_patient_mean": float(same.mean()),
        "same_patient_std": float(same.std()),
        "diff_patient_mean": float(off.mean()),
        "diff_patient_std": float(off.std()),
        "separation": float(same.mean() - off.mean()),
    }

    print("image->gene retrieval:", i2g)
    print("gene->image retrieval:", g2i)
    print("similarity:", sim_stats)

    save_json({"exp_name": args.exp_name, "split": args.split, "n": int(n),
               "image_to_gene_retrieval": i2g, "gene_to_image_retrieval": g2i,
               "similarity": sim_stats},
              os.path.join(os.path.join(_OUT, "analysis"), f"{args.exp_name}_metrics.json"))

    # retrieval table
    import csv
    with open(os.path.join(os.path.join(_OUT, "tables"), "retrieval_results.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["direction", "recall@1", "recall@5", "recall@10", "median_rank"])
        w.writerow(["image->gene", i2g["recall@1"], i2g["recall@5"],
                    i2g["recall@10"], i2g["median_rank"]])
        w.writerow(["gene->image", g2i["recall@1"], g2i["recall@5"],
                    g2i["recall@10"], g2i["median_rank"]])

    np.savez(os.path.join(os.path.join(_OUT, "embeddings"), f"{args.exp_name}_embeddings.npz"),
             image_latent=R["image_latent"], gene_latent=R["gene_latent"],
             y=R["y"], ids=R["ids"], same_sim=same, diff_sim=off)
    banner("alignment analysis done")


if __name__ == "__main__":
    main()
