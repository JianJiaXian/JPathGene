#!/usr/bin/env python
"""Download UNI2-h pathology foundation features (W8Yi/tcga-wsi-uni2h-features)
for our TCGA-BRCA patients, mean-pool the patch features to a patient-level
vector, and write a drop-in image_features.csv. This replaces the weak ResNet
features with SOTA pathology-foundation features to test whether stronger
features let the cross-modal method shine.
"""
import csv
import json
import os

import numpy as np

REPO = "W8Yi/tcga-wsi-uni2h-features"
SRC = "data/TCGA_BRCA"
OUT_DIR = "data/TCGA_BRCA_UNI"
FILELIST = "data/_uni_filelist.json"
CACHE = "data/_uni_cache"


def genomics_ids():
    with open(os.path.join(SRC, "genomic_features.csv")) as f:
        r = csv.reader(f)
        next(r)
        return [row[0] for row in r]


def pool_h5(path):
    import h5py
    with h5py.File(path, "r") as h:
        if "features" not in h:
            return None
        X = np.asarray(h["features"][:], dtype=np.float32)  # (1, N, 1536)
    X = np.squeeze(X)                       # -> (N, 1536)
    if X.ndim == 1:
        return X
    return X.mean(axis=0)                    # mean-pool patches -> (1536,)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    from huggingface_hub import hf_hub_download

    files = json.load(open(FILELIST))
    gids = genomics_ids()
    bc_to_full = {}
    for g in gids:
        bc_to_full.setdefault(g[:12], g)

    rows = {}
    dim = None
    for i, rel in enumerate(files):
        bc = "-".join(os.path.basename(rel).split("-")[:3])
        full = bc_to_full.get(bc)
        if full is None:
            continue
        try:
            p = hf_hub_download(REPO, rel, repo_type="dataset", local_dir=CACHE)
            vec = pool_h5(p)
            os.remove(p)  # free space after pooling
        except Exception as e:  # noqa
            print(f"  [{i}] {bc} FAILED: {repr(e)[:100]}", flush=True)
            continue
        if vec is None:
            continue
        rows[full] = vec
        dim = len(vec)
        if i % 10 == 0:
            print(f"  [{i}/{len(files)}] {bc} pooled dim={dim}", flush=True)

    print(f"pooled {len(rows)} patients, dim={dim}")
    cols = ["patient_id"] + [f"uni_{j}" for j in range(dim)]
    with open(os.path.join(OUT_DIR, "image_features.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for pid, vec in rows.items():
            w.writerow([pid] + [f"{v:.5f}" for v in vec])

    # symlink the other modalities (same patients/files)
    for name in ["genomic_features.csv", "labels.csv", "clinical.csv", "splits.csv"]:
        dst = os.path.join(OUT_DIR, name)
        if not os.path.exists(dst):
            os.symlink(os.path.join(SRC, name), dst)
    print(f"wrote {OUT_DIR}/image_features.csv  ({len(rows)} x {dim})")


if __name__ == "__main__":
    main()
