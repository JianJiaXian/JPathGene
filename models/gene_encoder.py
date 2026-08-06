"""Genetic / omics tabular encoders.

Default is an MLP over the (normalized, optionally PCA-reduced) gene feature
vector. A lightweight TabTransformer-style encoder is provided as an option:
each scalar gene feature is linearly embedded into a token, a shared Transformer
mixes the tokens, and a CLS token is read out as the gene latent z_gene.
"""
from typing import List

import torch
import torch.nn as nn

from models.image_encoder import mlp_block


class GeneMLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(256, 128), latent_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.net = mlp_block(in_dim, list(hidden_dims), latent_dim, dropout)
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        return self.out_norm(self.net(x))


class GeneTabTransformer(nn.Module):
    """Feature-tokenizer Transformer for tabular gene features.

    Each of the G scalar features becomes a d-dim token via a per-feature affine
    embedding; a learned CLS token is prepended and read out after L layers.
    """

    def __init__(self, in_dim: int, latent_dim: int = 128, d_token: int = 32,
                 depth: int = 2, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.d_token = d_token
        # per-feature weight & bias -> token embedding
        self.feat_w = nn.Parameter(torch.randn(in_dim, d_token) * 0.02)
        self.feat_b = nn.Parameter(torch.zeros(in_dim, d_token))
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=heads, dim_feedforward=d_token * 2,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Sequential(nn.LayerNorm(d_token),
                                  nn.Linear(d_token, latent_dim))
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(self, x):  # x: (B, G)
        B = x.shape[0]
        tokens = x.unsqueeze(-1) * self.feat_w.unsqueeze(0) + self.feat_b.unsqueeze(0)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, tokens], dim=1)        # (B, G+1, d)
        h = self.encoder(h)
        return self.out_norm(self.head(h[:, 0]))   # CLS readout


def build_gene_encoder(cfg: dict, in_dim: int) -> nn.Module:
    ge = cfg["model"]["gene_encoder"]
    latent = ge.get("latent_dim", 128)
    if ge.get("type", "mlp") == "tabtransformer":
        return GeneTabTransformer(
            in_dim=in_dim, latent_dim=latent,
            d_token=ge.get("d_token", 32), depth=ge.get("depth", 2),
            heads=ge.get("heads", 4), dropout=ge.get("dropout", 0.2))
    return GeneMLPEncoder(
        in_dim=in_dim, hidden_dims=ge.get("hidden_dims", [256, 128]),
        latent_dim=latent, dropout=ge.get("dropout", 0.2))
