"""Gene/omics helpers: parse symbols from column names, retrieval & correlation
utilities used by the alignment / gene-prediction analyses."""
from typing import List

import numpy as np


def parse_gene_symbols(columns: List[str]) -> List[str]:
    """Genomic columns are named like 'gene_0_CLEC3A' or 'gene_0'.
    Return the trailing SYMBOL when present, else the raw column name."""
    symbols = []
    for c in columns:
        parts = c.split("_")
        if len(parts) >= 3 and parts[0] == "gene":
            symbols.append("_".join(parts[2:]))
        else:
            symbols.append(c)
    return symbols


def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity matrix between A (Na,d) and B (Nb,d)."""
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A @ B.T


def retrieval_recall_at_k(sim: np.ndarray, ks=(1, 5, 10)) -> dict:
    """Given a square similarity matrix where the diagonal is the matching pair,
    compute Recall@k (each row's true match is its own index)."""
    n = sim.shape[0]
    order = np.argsort(-sim, axis=1)  # descending
    ranks = np.array([np.where(order[i] == i)[0][0] for i in range(n)])
    out = {}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(ranks < k))
    out["median_rank"] = float(np.median(ranks) + 1)
    return out


def pearson_per_dim(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-dimension Pearson correlation between predicted and true latents."""
    pred = pred - pred.mean(axis=0, keepdims=True)
    true = true - true.mean(axis=0, keepdims=True)
    num = (pred * true).sum(axis=0)
    den = np.sqrt((pred ** 2).sum(axis=0) * (true ** 2).sum(axis=0)) + 1e-8
    return num / den


def rowwise_cosine(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    tn = true / (np.linalg.norm(true, axis=1, keepdims=True) + 1e-8)
    return (pn * tn).sum(axis=1)


# Curated, interpretable gene-set signatures (the dominant, morphologically
# distinct breast-cancer molecular axes). Used as pathway-level JEPA targets so
# the histology->gene prediction is biologically interpretable. Only members
# present in a cohort's gene panel are used.
PATHWAY_SIGNATURES = {
    "luminal_ER": ["ESR1", "PGR", "TFF1", "TFF3", "AGR2", "AGR3", "SERPINA6",
                   "ANKRD30A", "SCGB2A2", "FOXA1", "GATA3", "XBP1"],
    "basal": ["KRT5", "KRT14", "KRT17", "KRT6B", "SOX10", "MIA", "ELF5",
              "SFRP1", "CRABP1", "FABP7", "PRAME", "FOXC1"],
    "proliferation": ["MKI67", "CCNB1", "CCNB2", "AURKA", "BUB1", "TOP2A",
                      "CENPF", "CDK1", "UBE2C", "RRM2", "BIRC5"],
    "immune": ["CD8A", "PTPRC", "CD3D", "CD2", "CXCL9", "CXCL13", "GZMB",
               "IDO1", "LAG3"],
}


def build_pathway_signatures(symbols, X, min_members=3):
    """Return (scores (N, P), names): each column is the mean of the
    (normalized) member genes present in `symbols`. Signatures with fewer than
    `min_members` present are dropped."""
    idx = {s: i for i, s in enumerate(symbols)}
    cols, names, member_counts = [], [], []
    for name, members in PATHWAY_SIGNATURES.items():
        present = [idx[m] for m in members if m in idx]
        if len(present) >= min_members:
            cols.append(X[:, present].mean(axis=1))
            names.append(name)
            member_counts.append(len(present))
    if not cols:
        return None, [], []
    return np.stack(cols, axis=1).astype(np.float32), names, member_counts
