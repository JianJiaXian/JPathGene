"""Gene-conditioned U-Net for radiogenomic disease segmentation.

Task: given a brain-MRI slice (and, when available, the patient's genomic
profile), segment the tumor / FLAIR-abnormality region.

Genetics is patient-level, so it does not get segmented; instead a genetic
embedding *conditions* the U-Net decoder via FiLM modulation. Crucially, we reuse
the JPathGene idea: an image->gene predictor imputes the genetic latent from
the image bottleneck, so the segmentation is genomics-guided even when genomics
is unavailable at test time (``gene_mode='jepa'``).

Modes (set at forward time):
  'none'  : image-only U-Net (no conditioning)
  'true'  : FiLM-condition on the real genetic latent
  'jepa'  : FiLM-condition on the image-imputed genetic latent
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True))


class FiLM(nn.Module):
    """Feature-wise linear modulation of a feature map by a conditioning vector."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * channels)

    def forward(self, x, cond):
        if cond is None:
            return x
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return x * (1 + scale) + shift


class GeneEncoderMLP(nn.Module):
    def __init__(self, in_dim, latent_dim=128, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, latent_dim))
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        return self.norm(self.net(x))


class GeneConditionedUNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=1, gene_dim=16, gene_latent=128,
                 base=32, dropout=0.2):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        # encoder
        self.e1 = conv_block(in_ch, c[0])
        self.e2 = conv_block(c[0], c[1])
        self.e3 = conv_block(c[1], c[2])
        self.b = conv_block(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        # gene encoder + JEPA image->gene predictor (from pooled bottleneck)
        self.gene_encoder = GeneEncoderMLP(gene_dim, gene_latent, dropout=dropout)
        self.img2gene = nn.Sequential(
            nn.Linear(c[3], gene_latent), nn.GELU(),
            nn.Linear(gene_latent, gene_latent))
        # decoder with FiLM conditioning
        self.u3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.d3 = conv_block(c[3], c[2])
        self.f3 = FiLM(gene_latent, c[2])
        self.u2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.d2 = conv_block(c[2], c[1])
        self.f2 = FiLM(gene_latent, c[1])
        self.u1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.d1 = conv_block(c[1], c[0])
        self.f1 = FiLM(gene_latent, c[0])
        self.out = nn.Conv2d(c[0], n_classes, 1)
        self.gene_latent_dim = gene_latent

    def forward(self, image, gene: Optional[torch.Tensor] = None,
                gene_mode: str = "true"):
        e1 = self.e1(image)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))

        # imputed genetic latent from the image bottleneck (JEPA)
        pooled = F.adaptive_avg_pool2d(b, 1).flatten(1)
        gene_hat = self.img2gene(pooled)

        z_gene_true = self.gene_encoder(gene) if gene is not None else None
        if gene_mode == "none":
            cond = None
        elif gene_mode == "jepa":
            cond = gene_hat
        else:  # 'true'
            cond = z_gene_true if z_gene_true is not None else gene_hat

        d3 = self.f3(self.d3(torch.cat([self.u3(b), e3], 1)), cond)
        d2 = self.f2(self.d2(torch.cat([self.u2(d3), e2], 1)), cond)
        d1 = self.f1(self.d1(torch.cat([self.u1(d2), e1], 1)), cond)
        logits = self.out(d1)
        return {"logits": logits, "gene_hat": gene_hat,
                "gene_latent": z_gene_true}


# ----------------------------- losses / metrics ----------------------------
def dice_loss(logits, target, eps=1.0):
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1, 2, 3)) + eps
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def tversky_loss(logits, target, alpha=0.3, beta=0.7, eps=1.0):
    """Tversky loss; beta>alpha penalizes false negatives (missed tumor) more,
    which prevents collapse to an empty mask under heavy class imbalance."""
    p = torch.sigmoid(logits)
    tp = (p * target).sum(dim=(1, 2, 3))
    fp = (p * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - p) * target).sum(dim=(1, 2, 3))
    t = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1 - t).mean()


def seg_loss(logits, target):
    return 0.5 * F.binary_cross_entropy_with_logits(logits, target) \
        + 0.5 * dice_loss(logits, target)


@torch.no_grad()
def dice_iou_per(logits, target, thr=0.5, eps=1e-6):
    """Per-sample Dice/IoU and a tumor-presence mask (GT has foreground)."""
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    psum = pred.sum(dim=(1, 2, 3))
    tsum = target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (psum + tsum + eps)
    iou = (inter + eps) / (psum + tsum - inter + eps)
    return dice, iou, (tsum > 0)


@torch.no_grad()
def dice_iou(logits, target, thr=0.5, eps=1e-6):
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    psum = pred.sum(dim=(1, 2, 3))
    tsum = target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (psum + tsum + eps)
    iou = (inter + eps) / (psum + tsum - inter + eps)
    return dice.mean().item(), iou.mean().item()
