#!/usr/bin/env python
"""Sanity-check the feature-level image+gene dataset described by a config.

Prints number of patients, image/gene feature dimensions, number of classes,
train/val/test counts, missing-value summary and label distribution. Fails with
a clear, actionable message (pointing at data/README.md) if CSVs are absent.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.io import load_config  # noqa: E402
from utils.logger import banner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    root = d["root"]
    banner(f"Feature dataset check :: {d.get('name')} :: {root}")

    if not os.path.isdir(root):
        print(f"ERROR: dataset root not found: {root}")
        print("The pipeline is implemented and ready; place the expected CSVs as")
        print("described in data/README.md, then re-run this check.")
        sys.exit(2)

    try:
        from datasets import build_datasets
    except Exception as e:  # noqa: BLE001
        print(f"ERROR importing dataset builder: {e}")
        sys.exit(1)

    # raw missing-value summary before merge
    for nm in ["image_features", "genomic_features", "gene_features", "labels",
               "clinical", "splits"]:
        p = os.path.join(root, f"{nm}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            nmiss = int(df.isna().sum().sum())
            print(f"  {nm+'.csv':24s} rows={len(df):5d} cols={df.shape[1]:5d} "
                  f"missing={nmiss}")

    datasets, meta = build_datasets(cfg)
    print("\n=== merged (patients with BOTH image & gene) ===")
    print(f"  total patients      : {meta['n_total']}")
    print(f"  image feature dim    : {meta['image_dim']}")
    print(f"  gene feature dim     : {meta['gene_dim']}")
    print(f"  clinical dim         : {meta['clinical_dim']}")
    print(f"  num classes          : {meta['num_classes']}")
    print(f"  gene PCA target dim  : {meta['gene_pca_target_dim']}")
    print(f"  split counts         : {meta['split_counts']}")
    print(f"  class counts         : {meta['class_counts']}")

    # label distribution per split
    print("\n=== label distribution per split ===")
    for split, ds in datasets.items():
        labels = ds.label
        uniq, cnt = np.unique(labels, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(uniq, cnt)}
        print(f"  {split:6s} n={len(labels):4d}  labels={dist}")

    # a couple of gene symbols for interpretability sanity
    syms = meta.get("gene_symbols", [])[:8]
    if syms:
        print(f"\n  example gene symbols : {syms}")
    banner("dataset check passed")


if __name__ == "__main__":
    main()
