"""Feature-level image + gene (omics) dataset.

Reads pre-extracted CSVs, merges by `patient_id`, builds reproducible splits,
normalizes features (train-only statistics), and optionally constructs a
PCA-reduced gene latent target for JEPA-style prediction.

Each item is a dict:
    patient_id : str
    image      : FloatTensor (D,)
    gene       : FloatTensor (G,)         # encoder input (optionally PCA-reduced)
    label      : LongTensor  ()
    clinical   : FloatTensor (C,)         # empty tensor if disabled/missing
    gene_pca_target : FloatTensor (P,)    # present iff PCA target is built
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset

    _BASE = Dataset
except ImportError:  # allow dataset stats scripts without torch
    torch = None
    _BASE = object

from datasets.transforms import FeatureScaler
from utils.gene_utils import parse_gene_symbols
from utils.pca_utils import PCAReducer


def _resolve(root: str, name: Optional[str], fallbacks: List[str]) -> Optional[str]:
    """Return first existing path among [name] + fallbacks, else None."""
    candidates = ([name] if name else []) + fallbacks
    for c in candidates:
        if not c:
            continue
        p = c if os.path.isabs(c) else os.path.join(root, c)
        if os.path.exists(p):
            return p
    return None


def _read_features(path: str, id_col: str = "patient_id") -> pd.DataFrame:
    df = pd.read_csv(path)
    if id_col not in df.columns:
        df = df.rename(columns={df.columns[0]: id_col})
    return df


class ImageGeneFeatureDataset(_BASE):
    def __init__(self, ids, image, gene, label, clinical=None, gene_target=None,
                 gene_columns=None):
        self.ids = list(ids)
        self.image = np.asarray(image, dtype=np.float32)
        self.gene = np.asarray(gene, dtype=np.float32)
        self.label = np.asarray(label, dtype=np.int64)
        self.clinical = (None if clinical is None
                         else np.asarray(clinical, dtype=np.float32))
        self.gene_target = (None if gene_target is None
                            else np.asarray(gene_target, dtype=np.float32))
        self.gene_columns = gene_columns or []

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        item = {
            "patient_id": self.ids[i],
            "image": torch.from_numpy(self.image[i]),
            "gene": torch.from_numpy(self.gene[i]),
            "label": torch.tensor(self.label[i], dtype=torch.long),
        }
        if self.clinical is not None:
            item["clinical"] = torch.from_numpy(self.clinical[i])
        else:
            item["clinical"] = torch.zeros(0)
        if self.gene_target is not None:
            item["gene_pca_target"] = torch.from_numpy(self.gene_target[i])
        return item


def _make_splits(ids: List[str], split_df: Optional[pd.DataFrame],
                 ratios, labels: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    """Return boolean masks for train/val/test over `ids`."""
    n = len(ids)
    if split_df is not None:
        m = dict(zip(split_df["patient_id"], split_df["split"]))
        sp = np.array([str(m.get(pid, "train")).lower() for pid in ids])
        # normalize naming
        sp[np.isin(sp, ["valid", "validation"])] = "val"
        masks = {s: (sp == s) for s in ("train", "val", "test")}
        if masks["val"].sum() == 0 or masks["test"].sum() == 0:
            return _random_split(n, ratios, labels, seed)
        return masks
    return _random_split(n, ratios, labels, seed)


def _random_split(n, ratios, labels, seed) -> Dict[str, np.ndarray]:
    """Deterministic stratified split."""
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    train_m = np.zeros(n, bool)
    val_m = np.zeros(n, bool)
    test_m = np.zeros(n, bool)
    r_tr, r_va, _ = ratios
    for c in np.unique(labels):
        cidx = idx[labels == c]
        rng.shuffle(cidx)
        nc = len(cidx)
        n_tr = int(round(nc * r_tr))
        n_va = int(round(nc * r_va))
        train_m[cidx[:n_tr]] = True
        val_m[cidx[n_tr:n_tr + n_va]] = True
        test_m[cidx[n_tr + n_va:]] = True
    return {"train": train_m, "val": val_m, "test": test_m}


def build_datasets(cfg: Dict, override_splits: Optional[Dict[str, str]] = None
                   ) -> Tuple[Dict[str, ImageGeneFeatureDataset], Dict]:
    """Build train/val/test datasets + metadata dict from a config.

    ``override_splits`` maps patient_id -> {'train','val','test'} and, when given,
    replaces splits.csv (used for leakage-safe cross-validation). All scalers /
    PCA are still fit on that fold's train split only.
    """
    d = cfg["data"]
    root = d["root"]
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Dataset root '{root}' not found.\n"
            f"See data/README.md for preparation instructions. The pipeline is "
            f"ready to run as soon as the expected CSVs are placed there."
        )

    img_path = _resolve(root, d.get("image_features"), ["image_features.csv"])
    gene_path = _resolve(root, d.get("gene_features"),
                         ["genomic_features.csv", "gene_features.csv"])
    lab_path = _resolve(root, d.get("labels"), ["labels.csv"])
    for nm, p in [("image_features", img_path), ("gene_features", gene_path),
                  ("labels", lab_path)]:
        if p is None:
            raise FileNotFoundError(
                f"Required CSV '{nm}' not found under {root}. See data/README.md.")

    img_df = _read_features(img_path)
    gene_df = _read_features(gene_path)
    lab_df = _read_features(lab_path)

    clin_df = None
    if d.get("use_clinical", False):
        clin_path = _resolve(root, d.get("clinical"), ["clinical.csv"])
        if clin_path:
            clin_df = _read_features(clin_path)

    split_df = None
    if override_splits is not None:
        split_df = pd.DataFrame(
            {"patient_id": list(override_splits.keys()),
             "split": list(override_splits.values())})
    else:
        sp_path = _resolve(root, d.get("splits"), ["splits.csv"])
        if sp_path:
            split_df = pd.read_csv(sp_path)

    # ---- inner-join on patient_id (only fully-paired patients) ----
    common = set(img_df["patient_id"]) & set(gene_df["patient_id"]) & set(
        lab_df["patient_id"])
    ids = sorted(common)
    if len(ids) == 0:
        raise RuntimeError("No patients shared across image/gene/label CSVs.")

    img_df = img_df.set_index("patient_id").loc[ids]
    gene_df = gene_df.set_index("patient_id").loc[ids]
    lab_df = lab_df.set_index("patient_id").loc[ids]

    img_cols = [c for c in img_df.columns if c != "patient_id"]
    gene_cols = [c for c in gene_df.columns if c != "patient_id"]
    img_X = img_df[img_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    gene_X = gene_df[gene_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    labels = lab_df["label"].astype(int).values

    clin_X = None
    clin_cols: List[str] = []
    if clin_df is not None:
        clin_df = clin_df.set_index("patient_id").loc[ids]
        clin_cols = [c for c in clin_df.columns]
        clin_X = clin_df[clin_cols].apply(
            pd.to_numeric, errors="coerce").fillna(0.0).values

    masks = _make_splits(ids, split_df, d.get("split_ratios", [0.7, 0.15, 0.15]),
                         labels, cfg.get("seed", 42))

    # ---- fit scalers on TRAIN only ----
    tr = masks["train"]
    img_scaler = FeatureScaler(d.get("normalize", "standard")).fit(img_X[tr])
    gene_scaler = FeatureScaler(d.get("normalize", "standard")).fit(gene_X[tr])
    img_Xn = img_scaler.transform(img_X)
    gene_Xn = gene_scaler.transform(gene_X)

    if clin_X is not None:
        clin_scaler = FeatureScaler(d.get("normalize", "standard")).fit(clin_X[tr])
        clin_Xn = clin_scaler.transform(clin_X)
    else:
        clin_Xn = None

    # ---- gene target construction (fit on train normalized gene features) ----
    gene_target = None
    pca_dim = 0
    pathway_names = []
    pca_cfg = d.get("gene_pca", {})
    target_type = cfg.get("model", {}).get("target", {}).get("gene_target_type",
                                                             "learned")
    if target_type == "pathway":
        from utils.gene_utils import build_pathway_signatures
        symbols_full = parse_gene_symbols(gene_cols)
        sigs, pathway_names, member_counts = build_pathway_signatures(
            symbols_full, gene_Xn)
        if sigs is not None:
            sscaler = FeatureScaler("standard").fit(sigs[tr])  # train-only
            gene_target = sscaler.transform(sigs)
            pca_dim = gene_target.shape[1]
        else:
            target_type = "learned"  # no signatures available -> fall back

    want_pca_target = (target_type == "pca") or pca_cfg.get("enabled", False)
    if gene_target is None and want_pca_target:
        n_comp = pca_cfg.get("n_components",
                             cfg.get("model", {}).get("target", {}).get("target_dim", 32))
        reducer = PCAReducer(n_components=n_comp, whiten=True)
        reducer.fit(gene_Xn[tr])
        gene_target = reducer.transform(gene_Xn)
        pca_dim = gene_target.shape[1]

    # ---- optionally PCA-reduce the gene *encoder input* too ----
    if pca_cfg.get("enabled", False):
        # Reuse same reduction for encoder input dimensionality.
        in_reducer = PCAReducer(n_components=pca_cfg.get("n_components", 64),
                                whiten=False)
        in_reducer.fit(gene_Xn[tr])
        gene_Xn = in_reducer.transform(gene_Xn)
        gene_cols = [f"pc_{i}" for i in range(gene_Xn.shape[1])]

    datasets = {}
    for split in ("train", "val", "test"):
        m = masks[split]
        datasets[split] = ImageGeneFeatureDataset(
            ids=[ids[i] for i in np.where(m)[0]],
            image=img_Xn[m],
            gene=gene_Xn[m],
            label=labels[m],
            clinical=None if clin_Xn is None else clin_Xn[m],
            gene_target=None if gene_target is None else gene_target[m],
            gene_columns=gene_cols,
        )

    meta = {
        "n_total": len(ids),
        "image_dim": img_Xn.shape[1],
        "gene_dim": gene_Xn.shape[1],
        "clinical_dim": 0 if clin_Xn is None else clin_Xn.shape[1],
        "num_classes": int(len(np.unique(labels))),
        "class_counts": {int(c): int((labels == c).sum())
                         for c in np.unique(labels)},
        "split_counts": {s: int(masks[s].sum()) for s in masks},
        "gene_pca_target_dim": pca_dim,
        "pathway_names": pathway_names,
        "gene_symbols": parse_gene_symbols(gene_cols),
        "gene_columns": gene_cols,
        "clinical_columns": clin_cols,
    }
    return datasets, meta
