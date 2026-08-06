"""Image-gene contrastive alignment (optional auxiliary objective).

Symmetric InfoNCE over a batch: the gene embedding of patient i should be the
nearest match to that patient's image embedding among all genes in the batch,
and vice versa (CLIP-style). A projection head maps each modality latent into a
shared alignment space before computing similarities.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim), nn.GELU(),
            nn.Linear(proj_dim, proj_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def info_nce(img_emb, gene_emb, temperature: float = 0.1):
    """Symmetric image<->gene InfoNCE. Inputs are L2-normalised (B, d)."""
    logits = img_emb @ gene_emb.t() / temperature        # (B, B)
    targets = torch.arange(img_emb.size(0), device=img_emb.device)
    loss_i2g = F.cross_entropy(logits, targets)
    loss_g2i = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_i2g + loss_g2i)
