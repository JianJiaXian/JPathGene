"""Cross-modal JEPA predictors.

Predict a *latent* target of the other modality from a context representation:
  ImageToGenePredictor : z_img_context  -> predicted gene latent target
  GeneToImagePredictor : z_gene_context -> predicted image latent target

Both are small residual MLPs. The output is L2-normalisable; cosine distance to
the (stop-grad) target is the default JEPA objective.
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))
        return x + h


class JEPAPredictor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256,
                 residual: bool = True, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.block = (ResidualBlock(hidden_dim, dropout) if residual
                      else nn.Sequential(nn.LayerNorm(hidden_dim), nn.GELU(),
                                         nn.Linear(hidden_dim, hidden_dim)))
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.inp(x)
        h = self.block(h)
        return self.out(h)


def ImageToGenePredictor(in_dim, out_dim, hidden_dim=256, residual=True):
    return JEPAPredictor(in_dim, out_dim, hidden_dim, residual)


def GeneToImagePredictor(in_dim, out_dim, hidden_dim=256, residual=True):
    return JEPAPredictor(in_dim, out_dim, hidden_dim, residual)
