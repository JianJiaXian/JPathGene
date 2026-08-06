"""TCGA-LGG brain-MRI radiogenomics segmentation dataset (Buda et al.).

Each sample is an axial MRI slice (3 channels: pre-contrast / FLAIR / post-
contrast) with a binary FLAIR-abnormality (tumor) mask, plus the patient's
genomic profile derived from the molecular-subtype cluster columns in
``data.csv`` (RNASeq / Methylation / miRNA / CN / RPPA / Oncosign / COC
clusters), one-hot encoded. Splits are by patient to prevent slice leakage.

Item dict:
    image      : FloatTensor (3, H, W)
    mask       : FloatTensor (1, H, W)   in {0,1}
    gene       : FloatTensor (G,)        one-hot genomic clusters
    patient_id : str
    has_tumor  : int
"""
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
    _BASE = Dataset
except ImportError:
    torch = None
    _BASE = object

GENE_COLS = ["RNASeqCluster", "MethylationCluster", "miRNACluster", "CNCluster",
             "RPPACluster", "OncosignCluster", "COCCluster"]


def _find_root(root: str) -> str:
    """Locate the directory that holds patient folders + data.csv."""
    for cand in [root, os.path.join(root, "kaggle_3m"),
                 os.path.join(root, "lgg-mri-segmentation", "kaggle_3m"),
                 os.path.join(root, "lgg-mri-segmentation")]:
        if os.path.isdir(cand) and (
                os.path.exists(os.path.join(cand, "data.csv"))
                or glob.glob(os.path.join(cand, "TCGA_*"))):
            return cand
    # deep search for data.csv
    hits = glob.glob(os.path.join(root, "**", "data.csv"), recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    raise FileNotFoundError(
        f"Could not locate the LGG kaggle_3m directory under {root}. "
        f"See data/README.md.")


def _patient_key(folder_name: str) -> str:
    # folder 'TCGA_CS_4941_19960909' -> data.csv Patient 'TCGA_CS_4941'
    parts = folder_name.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else folder_name


def _build_gene_table(csv_path: str, vocab_patients=None
                      ) -> Tuple[Dict[str, np.ndarray], int, List[str]]:
    """Build the per-patient one-hot genomic table. The one-hot *vocabulary*
    (which cluster categories exist) is derived ONLY from ``vocab_patients``
    (the training patients) to avoid any test information leaking into the
    feature definition; test patients with an unseen category map to all-zeros
    for that slot."""
    if not os.path.exists(csv_path):
        return {}, 0, []
    df = pd.read_csv(csv_path)
    pid_col = "Patient" if "Patient" in df.columns else df.columns[0]
    cols = [c for c in GENE_COLS if c in df.columns]
    vocab_df = df if vocab_patients is None else \
        df[df[pid_col].astype(str).isin(set(vocab_patients))]
    # one-hot each cluster column over its TRAIN-observed integer categories
    cats = {}
    for c in cols:
        vals = pd.to_numeric(vocab_df[c], errors="coerce")
        uniq = sorted(int(v) for v in vals.dropna().unique())
        cats[c] = uniq
    dim = sum(len(cats[c]) for c in cols)
    table = {}
    feat_names = []
    for c in cols:
        for u in cats[c]:
            feat_names.append(f"{c}_{u}")
    for _, row in df.iterrows():
        vec = []
        for c in cols:
            v = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
            oh = [0.0] * len(cats[c])
            if not pd.isna(v) and int(v) in cats[c]:
                oh[cats[c].index(int(v))] = 1.0
            vec += oh
        table[str(row[pid_col])] = np.asarray(vec, dtype=np.float32)
    return table, dim, feat_names


def _list_slices(data_root: str) -> List[Tuple[str, str, str]]:
    samples = []
    for folder in sorted(glob.glob(os.path.join(data_root, "TCGA_*"))):
        if not os.path.isdir(folder):
            continue
        pid = _patient_key(os.path.basename(folder))
        for mask_path in sorted(glob.glob(os.path.join(folder, "*_mask.tif"))):
            img_path = mask_path.replace("_mask.tif", ".tif")
            if os.path.exists(img_path):
                samples.append((img_path, mask_path, pid))
    return samples


def _patient_split(pids: List[str], ratios, seed) -> Dict[str, set]:
    uniq = sorted(set(pids))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(round(n * ratios[0]))
    n_va = int(round(n * ratios[1]))
    return {"train": set(uniq[:n_tr]), "val": set(uniq[n_tr:n_tr + n_va]),
            "test": set(uniq[n_tr + n_va:])}


class LGGRadiogenomicsDataset(_BASE):
    def __init__(self, samples, gene_table, gene_dim, image_size=256,
                 augment=False):
        self.samples = samples
        self.gene_table = gene_table
        self.gene_dim = gene_dim
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _load_img(self, path):
        from PIL import Image
        im = Image.open(path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0  # HWC

    def _load_mask(self, path):
        from PIL import Image
        m = Image.open(path).convert("L").resize(
            (self.image_size, self.image_size), Image.NEAREST)
        return (np.asarray(m, dtype=np.float32) > 127).astype(np.float32)

    def __getitem__(self, i):
        img_path, mask_path, pid = self.samples[i]
        img = self._load_img(img_path)          # HWC
        mask = self._load_mask(mask_path)        # HW
        if self.augment and np.random.rand() < 0.5:
            img = img[:, ::-1, :].copy()
            mask = mask[:, ::-1].copy()
        img = np.transpose(img, (2, 0, 1))       # CHW
        gene = self.gene_table.get(pid, np.zeros(self.gene_dim, dtype=np.float32))
        return {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "gene": torch.from_numpy(gene),
            "patient_id": pid,
            "has_tumor": int(mask.sum() > 0),
        }


def build_lgg_datasets(cfg) -> Tuple[Dict[str, "LGGRadiogenomicsDataset"], Dict]:
    d = cfg["data"]
    root = d["root"]
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"LGG dataset root '{root}' not found. See data/README.md.")
    data_root = _find_root(root)
    samples = _list_slices(data_root)
    if not samples:
        raise RuntimeError(f"No MRI slices found under {data_root}.")
    pids = [s[2] for s in samples]
    # split FIRST, then derive the genomic one-hot vocabulary from train patients
    masks = _patient_split(pids, d.get("split_ratios", [0.7, 0.15, 0.15]),
                           cfg.get("seed", 42))
    gene_table, gene_dim, gene_names = _build_gene_table(
        os.path.join(data_root, "data.csv"), vocab_patients=masks["train"])

    size = d.get("image_size", 256)
    datasets = {}
    for split in ("train", "val", "test"):
        sub = [s for s in samples if s[2] in masks[split]]
        datasets[split] = LGGRadiogenomicsDataset(
            sub, gene_table, gene_dim, size, augment=(split == "train"))

    meta = {
        "n_slices": len(samples),
        "n_patients": len(set(pids)),
        "gene_dim": gene_dim,
        "gene_names": gene_names,
        "image_size": size,
        "split_slice_counts": {k: sum(1 for s in samples if s[2] in v)
                               for k, v in masks.items()},
        "split_patient_counts": {k: len(v) for k, v in masks.items()},
    }
    return datasets, meta
