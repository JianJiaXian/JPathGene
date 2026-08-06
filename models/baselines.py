"""Baseline models for the feature-level image-gene benchmark.

All baselines expose the same forward contract as PathGeneJEPA's downstream mode:
they return a dict containing at least 'fusion_logits'. This lets train.py /
evaluate.py treat every model uniformly.
"""
from typing import Dict

import torch
import torch.nn as nn

from models.contrastive import ProjectionHead
from models.fusion_blocks import ConcatFusion, GatedFusion
from models.gene_encoder import build_gene_encoder
from models.image_encoder import build_image_encoder
from models.pathgene_jepa import build_pathgene_jepa

BASELINE_NAMES = [
    "image_only", "gene_only", "early_fusion", "late_fusion",
    "concat_fusion", "gated_fusion", "contrastive", "pathgene_jepa",
]


class UnimodalBaseline(nn.Module):
    def __init__(self, cfg, meta, modality: str):
        super().__init__()
        self.modality = modality
        nc = cfg["task"].get("num_classes", 2)
        if modality == "image":
            self.encoder = build_image_encoder(cfg, meta["image_dim"])
            d = cfg["model"]["image_encoder"]["latent_dim"]
        else:
            self.encoder = build_gene_encoder(cfg, meta["gene_dim"])
            d = cfg["model"]["gene_encoder"]["latent_dim"]
        self.head = nn.Sequential(nn.LayerNorm(d), nn.GELU(),
                                  nn.Dropout(0.3), nn.Linear(d, nc))

    def forward(self, image, gene, **kw) -> Dict[str, torch.Tensor]:
        x = image if self.modality == "image" else gene
        z = self.encoder(x)
        if isinstance(z, tuple):
            z = z[0]
        return {"fusion_logits": self.head(z),
                f"{self.modality}_latent": z}


class FusionBaseline(nn.Module):
    """early / late / concat / gated / contrastive fusion of both modalities."""

    def __init__(self, cfg, meta, mode: str):
        super().__init__()
        self.mode = mode
        nc = cfg["task"].get("num_classes", 2)
        self.image_encoder = build_image_encoder(cfg, meta["image_dim"])
        self.gene_encoder = build_gene_encoder(cfg, meta["gene_dim"])
        di = cfg["model"]["image_encoder"]["latent_dim"]
        dg = cfg["model"]["gene_encoder"]["latent_dim"]
        hid = cfg["model"]["fusion"].get("hidden_dim", 256)
        drop = cfg["model"]["fusion"].get("dropout", 0.3)

        if mode == "early_fusion":
            # fuse raw feature vectors before encoding
            self.early = nn.Sequential(
                nn.Linear(meta["image_dim"] + meta["gene_dim"], hid),
                nn.LayerNorm(hid), nn.GELU(), nn.Dropout(drop),
                nn.Linear(hid, nc))
        elif mode == "late_fusion":
            self.image_head = nn.Linear(di, nc)
            self.gene_head = nn.Linear(dg, nc)
        elif mode == "gated_fusion":
            self.fusion = GatedFusion([di, dg], nc, hid, drop)
        elif mode == "cross_attention":
            from models.fusion_blocks import CrossAttentionFusion
            self.fusion = CrossAttentionFusion([di, dg], nc, hid, drop)
        else:  # concat_fusion and contrastive both use concat head
            self.fusion = ConcatFusion([di, dg], nc, hid, drop)
        if mode == "contrastive":
            self.img_proj = ProjectionHead(di)
            self.gene_proj = ProjectionHead(dg)

    def forward(self, image, gene, **kw) -> Dict[str, torch.Tensor]:
        if self.mode == "early_fusion":
            logits = self.early(torch.cat([image, gene], dim=-1))
            return {"fusion_logits": logits}
        zi = self.image_encoder(image)
        if isinstance(zi, tuple):
            zi = zi[0]
        zg = self.gene_encoder(gene)
        out = {"image_latent": zi, "gene_latent": zg}
        if self.mode == "late_fusion":
            out["fusion_logits"] = 0.5 * (self.image_head(zi) + self.gene_head(zg))
        else:
            out["fusion_logits"] = self.fusion([zi, zg])
        if self.mode == "contrastive":
            out["img_proj"] = self.img_proj(zi)
            out["gene_proj"] = self.gene_proj(zg)
        return out


def build_baseline(name: str, cfg, meta) -> nn.Module:
    name = name.lower()
    if name == "image_only":
        return UnimodalBaseline(cfg, meta, "image")
    if name == "gene_only":
        return UnimodalBaseline(cfg, meta, "gene")
    if name in ("early_fusion", "late_fusion", "concat_fusion",
                "gated_fusion", "cross_attention", "contrastive"):
        return FusionBaseline(cfg, meta, name)
    if name == "pathgene_jepa":
        return build_pathgene_jepa(cfg, meta)
    raise ValueError(f"unknown model '{name}'. choices: {BASELINE_NAMES}")
