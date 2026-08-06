"""Histopathology image-feature encoders.

Default is a FeatureMLP over pre-extracted WSI/patch features. An optional
attention-based MIL pooling head is provided for the case where a patient is
represented by multiple patch feature vectors (input shape (B, P, D)).
"""
from typing import List

import torch
import torch.nn as nn


def mlp_block(in_dim: int, hidden_dims: List[int], out_dim: int,
              dropout: float = 0.2) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(),
                   nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class MILAttentionPooling(nn.Module):
    """Gated attention pooling (Ilse et al., 2018) over patch features."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (B, P, D)
        a = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        a = torch.softmax(self.w(a), dim=1)        # (B, P, 1)
        return (a * x).sum(dim=1), a.squeeze(-1)   # (B, D), attn


class FeatureMLP(nn.Module):
    """Encode pre-extracted image features into an image latent z_img."""

    def __init__(self, in_dim: int, hidden_dims=(512, 256), latent_dim: int = 128,
                 dropout: float = 0.2, use_mil: bool = False):
        super().__init__()
        self.use_mil = use_mil
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        if use_mil:
            self.mil = MILAttentionPooling(in_dim)
        self.net = mlp_block(in_dim, list(hidden_dims), latent_dim, dropout)
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        attn = None
        if self.use_mil and x.dim() == 3:
            x, attn = self.mil(x)
        z = self.out_norm(self.net(x))
        return z if attn is None else (z, attn)


def build_image_encoder(cfg: dict, in_dim: int) -> FeatureMLP:
    ie = cfg["model"]["image_encoder"]
    return FeatureMLP(
        in_dim=in_dim,
        hidden_dims=ie.get("hidden_dims", [512, 256]),
        latent_dim=ie.get("latent_dim", 128),
        dropout=ie.get("dropout", 0.2),
        use_mil=ie.get("mil", False),
    )
