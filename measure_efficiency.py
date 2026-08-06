#!/usr/bin/env python
"""Measure model efficiency: parameter count, inference latency / throughput,
and (if available) GPU peak memory. Writes outputs/tables/efficiency.csv.
"""
import argparse
import os
import time

import numpy as np
import torch

from datasets import build_datasets
from models.baselines import build_baseline
from utils.engine import make_loaders, move_batch
from utils.io import ensure_dir, get_device, load_config, save_json
from utils.logger import banner
from utils.io import OUT_ROOT as _OUT  # namespaced output root


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def time_inference(model, loader, device, warmup=2, iters=10):
    model.eval()
    batches = list(loader)
    if not batches:
        return float("nan"), float("nan")
    b = move_batch(batches[0], device)
    is_pg = model.__class__.__name__ == "PathGeneJEPA"

    def fwd():
        if is_pg:
            return model(b["image"], b["gene"],
                         gene_pca_target=b.get("gene_pca_target"),
                         mode="downstream")
        return model(b["image"], b["gene"])

    for _ in range(warmup):
        fwd()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fwd()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    bs = b["image"].size(0)
    return dt * 1000.0, bs / dt  # ms/batch, samples/sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--exp_name", default="pathgene_efficiency")
    ap.add_argument("--models", nargs="*",
                    default=["image_only", "gene_only", "concat_fusion",
                             "gated_fusion", "pathgene_jepa"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "cuda"))
    ensure_dir(os.path.join(_OUT, "tables"))
    banner(f"efficiency :: {device}")
    datasets, meta = build_datasets(cfg)
    loaders = make_loaders(datasets, cfg)

    rows = []
    for name in args.models:
        model = build_baseline(name, cfg, meta).to(device)
        if name == "pathgene_jepa" and args.checkpoint and \
                os.path.exists(args.checkpoint):
            st = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(st["model"], strict=False)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        total, trainable = count_params(model)
        ms, sps = time_inference(model, loaders["test"], device)
        mem = (torch.cuda.max_memory_allocated() / 1e6
               if device.type == "cuda" else float("nan"))
        rows.append({"model": name, "params_M": round(total / 1e6, 3),
                     "trainable_M": round(trainable / 1e6, 3),
                     "ms_per_batch": round(ms, 3),
                     "samples_per_sec": round(sps, 1),
                     "peak_mem_MB": round(mem, 1)})
        print(rows[-1])

    import csv
    with open(os.path.join(os.path.join(_OUT, "tables"), "efficiency.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    save_json(rows, os.path.join(os.path.join(_OUT, "tables"), "efficiency.json"))
    banner("efficiency done")


if __name__ == "__main__":
    main()
