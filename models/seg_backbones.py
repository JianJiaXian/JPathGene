"""Segmentation backbones for the radiogenomics comparison.

All backbones share the gene-conditioning interface of
:class:`GeneConditionedUNet` -- they accept ``(image, gene, gene_mode)`` and
return ``{logits, gene_hat, gene_latent}`` -- so genetics conditions every model
via FiLM and the image->gene JEPA imputation is available for each. This lets us
compare architectures fairly under identical conditioning.

Architectures:
  unet      : plain U-Net (baseline, in models/segmentation.py)
  attunet   : Attention U-Net (attention gates on skips)
  transunet : CNN encoder + Transformer bottleneck + U-Net decoder (transformer)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.segmentation import FiLM, GeneEncoderMLP, conv_block


# ----------------------------- Attention U-Net -----------------------------
class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter):
        super().__init__()
        self.wg = nn.Conv2d(g_ch, inter, 1)
        self.wx = nn.Conv2d(x_ch, inter, 1)
        self.psi = nn.Conv2d(inter, 1, 1)

    def forward(self, g, x):
        a = torch.relu(self.wg(g) + self.wx(x))
        a = torch.sigmoid(self.psi(a))
        return x * a


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=1, gene_dim=16, gene_latent=128,
                 base=32, dropout=0.2):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1 = conv_block(in_ch, c[0]); self.e2 = conv_block(c[0], c[1])
        self.e3 = conv_block(c[1], c[2]); self.b = conv_block(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.gene_encoder = GeneEncoderMLP(gene_dim, gene_latent, dropout=dropout)
        self.img2gene = nn.Sequential(nn.Linear(c[3], gene_latent), nn.GELU(),
                                      nn.Linear(gene_latent, gene_latent))
        self.u3 = nn.ConvTranspose2d(c[3], c[2], 2, 2); self.a3 = AttentionGate(c[2], c[2], c[2] // 2)
        self.d3 = conv_block(c[3], c[2]); self.f3 = FiLM(gene_latent, c[2])
        self.u2 = nn.ConvTranspose2d(c[2], c[1], 2, 2); self.a2 = AttentionGate(c[1], c[1], c[1] // 2)
        self.d2 = conv_block(c[2], c[1]); self.f2 = FiLM(gene_latent, c[1])
        self.u1 = nn.ConvTranspose2d(c[1], c[0], 2, 2); self.a1 = AttentionGate(c[0], c[0], c[0] // 2)
        self.d1 = conv_block(c[1], c[0]); self.f1 = FiLM(gene_latent, c[0])
        self.out = nn.Conv2d(c[0], n_classes, 1)
        self.gene_latent_dim = gene_latent

    def _cond(self, gene, gene_hat, gene_mode):
        zt = self.gene_encoder(gene) if gene is not None else None
        if gene_mode == "none":
            return None, zt
        if gene_mode == "jepa":
            return gene_hat, zt
        return (zt if zt is not None else gene_hat), zt

    def forward(self, image, gene=None, gene_mode="true"):
        e1 = self.e1(image); e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2)); b = self.b(self.pool(e3))
        gene_hat = self.img2gene(F.adaptive_avg_pool2d(b, 1).flatten(1))
        cond, zt = self._cond(gene, gene_hat, gene_mode)
        u3 = self.u3(b); d3 = self.f3(self.d3(torch.cat([u3, self.a3(u3, e3)], 1)), cond)
        u2 = self.u2(d3); d2 = self.f2(self.d2(torch.cat([u2, self.a2(u2, e2)], 1)), cond)
        u1 = self.u1(d2); d1 = self.f1(self.d1(torch.cat([u1, self.a1(u1, e1)], 1)), cond)
        return {"logits": self.out(d1), "gene_hat": gene_hat, "gene_latent": zt}


# ------------------------------- TransUNet ---------------------------------
class TransUNet(nn.Module):
    """CNN encoder + Transformer bottleneck + U-Net decoder (transformer backbone)."""

    def __init__(self, in_ch=3, n_classes=1, gene_dim=16, gene_latent=128,
                 base=32, dropout=0.2, depth=4, heads=8, image_size=256):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1 = conv_block(in_ch, c[0]); self.e2 = conv_block(c[0], c[1])
        self.e3 = conv_block(c[1], c[2]); self.b = conv_block(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        # transformer over the bottleneck tokens (bottleneck is image/8)
        n_tokens = (image_size // 8) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, c[3]))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(c[3], heads, c[3] * 2, dropout,
                                           batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, depth)
        self.gene_encoder = GeneEncoderMLP(gene_dim, gene_latent, dropout=dropout)
        self.img2gene = nn.Sequential(nn.Linear(c[3], gene_latent), nn.GELU(),
                                      nn.Linear(gene_latent, gene_latent))
        self.u3 = nn.ConvTranspose2d(c[3], c[2], 2, 2)
        self.d3 = conv_block(c[3], c[2]); self.f3 = FiLM(gene_latent, c[2])
        self.u2 = nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.d2 = conv_block(c[2], c[1]); self.f2 = FiLM(gene_latent, c[1])
        self.u1 = nn.ConvTranspose2d(c[1], c[0], 2, 2)
        self.d1 = conv_block(c[1], c[0]); self.f1 = FiLM(gene_latent, c[0])
        self.out = nn.Conv2d(c[0], n_classes, 1)
        self.gene_latent_dim = gene_latent

    def _cond(self, gene, gene_hat, gene_mode):
        zt = self.gene_encoder(gene) if gene is not None else None
        if gene_mode == "none":
            return None, zt
        if gene_mode == "jepa":
            return gene_hat, zt
        return (zt if zt is not None else gene_hat), zt

    def forward(self, image, gene=None, gene_mode="true"):
        e1 = self.e1(image); e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2)); b = self.b(self.pool(e3))
        B, C, H, W = b.shape
        tok = b.flatten(2).transpose(1, 2)                 # (B, HW, C)
        pos = self.pos if self.pos.shape[1] == tok.shape[1] else \
            F.interpolate(self.pos.transpose(1, 2), size=tok.shape[1],
                          mode="linear").transpose(1, 2)
        tok = self.transformer(tok + pos)
        b = tok.transpose(1, 2).reshape(B, C, H, W)
        gene_hat = self.img2gene(F.adaptive_avg_pool2d(b, 1).flatten(1))
        cond, zt = self._cond(gene, gene_hat, gene_mode)
        d3 = self.f3(self.d3(torch.cat([self.u3(b), e3], 1)), cond)
        d2 = self.f2(self.d2(torch.cat([self.u2(d3), e2], 1)), cond)
        d1 = self.f1(self.d1(torch.cat([self.u1(d2), e1], 1)), cond)
        return {"logits": self.out(d1), "gene_hat": gene_hat, "gene_latent": zt}
