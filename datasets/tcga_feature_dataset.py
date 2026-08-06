"""TCGA feature-level dataset.

The generic :class:`ImageGeneFeatureDataset` already handles the TCGA CSV layout
(`img_*` columns, `gene_<i>_<SYMBOL>` columns, binary `label`). This module is a
thin, explicitly-named alias so that `configs/tcga_*.yaml` and the paper can
refer to a TCGA-specific entry point, and to document the join semantics:

  * image_features.csv  -- pre-extracted WSI patch-mean features (ResNet/CTransPath)
  * genomic_features.csv -- top-variance RNA-seq genes (named columns)
  * labels.csv           -- binary clinical endpoint (+ optional survival columns)
  * only patients with BOTH histology and genomics are retained.
"""
from datasets.image_gene_feature_dataset import build_datasets  # noqa: F401


def build_tcga_datasets(cfg):
    """Identical to build_datasets; kept for naming clarity in TCGA configs."""
    return build_datasets(cfg)
