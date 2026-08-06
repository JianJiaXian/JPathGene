"""Target encoders for JEPA-style latent prediction.

In JEPA the *target* representation must not receive gradient from the predictor
(otherwise the model collapses). We support two regimes:

  * EMA target  : a frozen exponential-moving-average copy of the online
                  context encoder (I-JEPA / BYOL style).
  * stop-grad   : reuse the online encoder but detach its output (a cheaper
                  approximation; clearly documented as such).

For the gene side, a third option is a *fixed* PCA latent target precomputed by
the dataset (`gene_pca_target`); this needs no encoder at all and is the most
reproducible biological-program target.
"""
import copy

import torch
import torch.nn as nn


def build_ema_copy(module: nn.Module) -> nn.Module:
    """Frozen deep copy used as an EMA target encoder."""
    tgt = copy.deepcopy(module)
    for p in tgt.parameters():
        p.requires_grad_(False)
    tgt.eval()
    return tgt


@torch.no_grad()
def ema_update(target: nn.Module, source: nn.Module, decay: float = 0.996):
    """target = decay * target + (1 - decay) * source (params and buffers)."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.mul_(decay).add_(sp.detach(), alpha=1.0 - decay)
    for tb, sb in zip(target.buffers(), source.buffers()):
        tb.copy_(sb)


class TargetProjector(nn.Module):
    """Optional projection applied on top of a (possibly EMA) target encoder so
    the JEPA target dimensionality can differ from the latent dimensionality."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.proj(x))
