#!/usr/bin/env python
"""LUAD Job B: download UNI2-h features for TCGA-LUAD diagnostic slides, mean-pool
to patient-level 1536-d vectors -> image_features.csv. Resumable (skips patients
already written). Keyed on 12-char patient id to merge with Job A's RNA/labels.
"""
import csv, os
import numpy as np

REPO = "W8Yi/tcga-wsi-uni2h-features"
OUT_DIR = "data/TCGA_LUAD_UNI"
CACHE = "data/_luad_uni_cache"
OUTCSV = os.path.join(OUT_DIR, "image_features.csv")


def pool_h5(path):
    import h5py
    with h5py.File(path, "r") as h:
        if "features" not in h:
            return None
        X = np.asarray(h["features"][:], dtype=np.float32)
    X = np.squeeze(X)
    return X if X.ndim == 1 else X.mean(axis=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception:
        os.system("pip install --user -q huggingface_hub")
        from huggingface_hub import HfApi, hf_hub_download

    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    luad = [f for f in files if f.startswith("TCGA-LUAD/") and f.endswith(".h5")]
    # prefer diagnostic (DX) slides; one slide per patient
    dx = [f for f in luad if "-DX" in os.path.basename(f).upper()] or luad
    per_pat = {}
    for f in sorted(dx):
        bc = os.path.basename(f).split("-")
        pid = "-".join(bc[:3])                  # TCGA-05-4244
        per_pat.setdefault(pid, f)              # first DX slide per patient
    print(f"{len(luad)} LUAD h5; {len(per_pat)} unique patients", flush=True)

    done = set()
    if os.path.exists(OUTCSV):
        with open(OUTCSV) as f:
            r = csv.reader(f)
            next(r, None)
            done = {row[0] for row in r if row}
        print(f"resume: {len(done)} already pooled", flush=True)

    rows, dim = {}, None
    for i, (pid, rel) in enumerate(sorted(per_pat.items())):
        if pid in done:
            continue
        try:
            p = hf_hub_download(REPO, rel, repo_type="dataset", local_dir=CACHE)
            vec = pool_h5(p)
            os.remove(p)
        except Exception as e:  # noqa
            print(f"  [{i}] {pid} FAILED {repr(e)[:90]}", flush=True)
            continue
        if vec is None:
            continue
        rows[pid] = vec
        dim = len(vec)
        if i % 20 == 0:
            print(f"  [{i}/{len(per_pat)}] {pid} dim={dim}", flush=True)

    # append/merge
    header_needed = not os.path.exists(OUTCSV)
    if dim is None and header_needed:
        print("no vectors pooled; abort")
        return
    if dim is None:
        dim = 1536
    mode = "a" if os.path.exists(OUTCSV) else "w"
    with open(OUTCSV, mode, newline="") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["patient_id"] + [f"uni_{j}" for j in range(dim)])
        for pid, vec in rows.items():
            w.writerow([pid] + [f"{v:.5f}" for v in vec])

    total = len(done) + len(rows)
    print(f"wrote {OUTCSV}: +{len(rows)} new, {total} total x {dim}")
    print("JOB_B_DONE")


if __name__ == "__main__":
    main()
