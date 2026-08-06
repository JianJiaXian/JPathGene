"""Multimodal fusion heads.

All fusion blocks consume a list of latent vectors and produce class logits.
PathGeneFusion additionally consumes the *cross-predicted* latents so the
downstream head benefits from the JEPA correspondence, not just the raw latents.
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    def __init__(self, dims: List[int], num_classes: int, hidden: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(sum(dims), hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, feats: List[torch.Tensor]):
        return self.head(torch.cat(feats, dim=-1))


class GatedFusion(nn.Module):
    """Learn a per-modality gate, then classify the gated sum."""

    def __init__(self, dims: List[int], num_classes: int, hidden: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        d = max(dims)
        self.proj = nn.ModuleList([nn.Linear(x, d) for x in dims])
        self.gate = nn.Sequential(nn.Linear(d * len(dims), len(dims)), nn.Softmax(-1))
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d, num_classes))

    def forward(self, feats: List[torch.Tensor]):
        proj = [p(f) for p, f in zip(self.proj, feats)]          # each (B, d)
        stack = torch.stack(proj, dim=1)                          # (B, M, d)
        g = self.gate(torch.cat(proj, dim=-1)).unsqueeze(-1)      # (B, M, 1)
        fused = (g * stack).sum(dim=1)                            # (B, d)
        return self.head(fused)


class CrossAttentionFusion(nn.Module):
    """Single-layer cross-attention: image queries gene (and vice versa)."""

    def __init__(self, dims: List[int], num_classes: int, hidden: int = 256,
                 dropout: float = 0.3, heads: int = 4):
        super().__init__()
        d = max(dims)
        self.proj = nn.ModuleList([nn.Linear(x, d) for x in dims])
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                          batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d, num_classes))

    def forward(self, feats: List[torch.Tensor]):
        proj = [p(f) for p, f in zip(self.proj, feats)]
        seq = torch.stack(proj, dim=1)                            # (B, M, d)
        out, _ = self.attn(seq, seq, seq)
        return self.head(out.mean(dim=1))


class UncertaintyGatedFusion(nn.Module):
    """Gate fusion by the diffusion predictor's cross-modal uncertainty.

    Inputs are [z_img, z_gene, pred_gene, pred_img]; the cross-predicted latents
    (slots 2,3) are scaled by a reliability weight exp(-beta * uncertainty) before
    a learned attention gate combines all four. When the image->gene (or
    gene->image) prediction is uncertain, its contribution is automatically
    suppressed -- a trustworthy, uncertainty-aware fusion.
    """

    def __init__(self, dims: List[int], num_classes: int, hidden: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        d = max(dims)
        self.proj = nn.ModuleList([nn.Linear(x, d) for x in dims])
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gate = nn.Sequential(nn.Linear(d * len(dims), len(dims)),
                                  nn.Softmax(dim=-1))
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d, num_classes))

    def forward(self, feats: List[torch.Tensor], uncertainty=None):
        proj = [p(f) for p, f in zip(self.proj, feats)]          # each (B, d)
        if uncertainty is not None:
            # uncertainty: (B, 2) for cross slots [pred_gene, pred_img]
            rel = torch.exp(-F.softplus(self.beta) * uncertainty)  # (B,2) in (0,1]
            proj[2] = proj[2] * rel[:, 0:1]
            proj[3] = proj[3] * rel[:, 1:2]
        stack = torch.stack(proj, dim=1)                          # (B, M, d)
        g = self.gate(torch.cat(proj, dim=-1)).unsqueeze(-1)      # (B, M, 1)
        fused = (g * stack).sum(dim=1)
        return self.head(fused)


def build_fusion(mode: str, dims: List[int], num_classes: int, hidden: int = 256,
                 dropout: float = 0.3) -> nn.Module:
    if mode.lower() in ("uncertainty_gated", "uncertainty", "ugated"):
        return UncertaintyGatedFusion(dims, num_classes, hidden, dropout)
    mode = mode.lower()
    if mode in ("gated", "gated_fusion"):
        return GatedFusion(dims, num_classes, hidden, dropout)
    if mode in ("cross_attention", "crossattn", "cross_attention_fusion"):
        return CrossAttentionFusion(dims, num_classes, hidden, dropout)
    # concat and pathgene_jepa both use a concat MLP head; pathgene_jepa simply
    # feeds extra (cross-predicted) latents into the same concat head.
    return ConcatFusion(dims, num_classes, hidden, dropout)
