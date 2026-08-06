#!/usr/bin/env python
"""Full evaluation of a trained checkpoint on the test split.

Reports AUC (macro for multiclass), accuracy, F1, precision, recall, ECE,
confusion matrix and per-class metrics; writes them to outputs/analysis.
"""
import argparse
import os

import numpy as np
import torch

from datasets import build_datasets
from models.baselines import build_baseline
from utils.engine import make_loaders, predict
from utils.io import ensure_dir, get_device, load_config, save_json
from utils.logger import banner
from utils.metrics import compute_metrics, confusion, per_class_metrics
from utils.io import OUT_ROOT as _OUT  # namespaced output root


def _load_model(checkpoint, cfg, meta, device, model_name):
    model = build_baseline(model_name, cfg, meta).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    try:
        model.load_state_dict(state["model"], strict=True)
    except RuntimeError:
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        print(f"[evaluate] non-strict load (missing={len(missing)}, "
              f"unexpected={len(unexpected)})")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--exp_name", default="eval")
    ap.add_argument("--model", default="pathgene_jepa")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "cuda"))
    ensure_dir(os.path.join(_OUT, "analysis"))
    banner(f"evaluate {args.model} :: {args.checkpoint} :: {device}")

    datasets, meta = build_datasets(cfg)
    loaders = make_loaders(datasets, cfg)
    num_classes = meta["num_classes"]

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    model = _load_model(args.checkpoint, cfg, meta, device, args.model)

    y, p, ids = predict(model, loaders[args.split], device, num_classes)
    metrics = compute_metrics(y, p, num_classes)
    cm = confusion(y, p, num_classes).tolist()
    per_class = per_class_metrics(y, p, num_classes)

    print("=== metrics ===")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print("=== confusion ===")
    for row in cm:
        print("  ", row)

    out = {"exp_name": args.exp_name, "model": args.model, "split": args.split,
           "checkpoint": args.checkpoint, "metrics": metrics,
           "confusion_matrix": cm, "per_class": per_class, "meta": meta}
    path = os.path.join(os.path.join(_OUT, "analysis"), f"{args.exp_name}_metrics.json")
    save_json(out, path)
    np.savez(os.path.join(os.path.join(_OUT, "predictions"), f"{args.exp_name}_preds.npz"),
             y_true=y, y_prob=p, patient_id=np.array(ids))
    banner(f"saved {path}")


if __name__ == "__main__":
    main()
