"""JPathGene: cross-modal latent predictive learning between histopathology
image features and genetic profiles.

Slogan: *predict genetic programs in latent space, not raw gene expression.*

Pipeline
--------
  z_img  = ImageEncoder(image)
  z_gene = GeneEncoder(gene)
  pred_gene = ImageToGenePredictor(z_img)      ~  target_gene  (stop-grad / EMA / PCA)
  pred_img  = GeneToImagePredictor(z_gene)      ~  target_img   (stop-grad / EMA)
  logits    = FusionHead([z_img, z_gene, pred_gene, pred_img])

The cross-predicted latents are fed to the fusion head so the downstream
predictor benefits from the learned image<->gene biological correspondence.
Ablation flags toggle each JEPA direction and the contrastive term while keeping
the head dimensionality fixed (disabled branches are zeroed), so comparisons are
architecturally fair.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn

from models.contrastive import ProjectionHead
from models.diffusion_predictor import ConditionalLatentDiffusion
from models.fusion_blocks import build_fusion
from models.gene_encoder import build_gene_encoder
from models.image_encoder import build_image_encoder
from models.jepa_predictors import GeneToImagePredictor, ImageToGenePredictor
from models.target_encoders import build_ema_copy, ema_update


class PathGeneJEPA(nn.Module):
    def __init__(self, cfg: Dict, meta: Dict):
        super().__init__()
        self.cfg = cfg
        m = cfg["model"]
        self.latent_dim = m["image_encoder"]["latent_dim"]
        gene_latent = m["gene_encoder"]["latent_dim"]
        assert gene_latent == self.latent_dim, \
            "image and gene latent dims must match for symmetric fusion"

        self.gene_target_type = m["target"].get("gene_target_type", "learned")
        self.use_ema = m["target"].get("ema", True)
        self.ema_decay = m["target"].get("ema_decay", 0.996)

        # ablation switches (overridable by run_ablation.py)
        self.use_img2gene = True
        self.use_gene2img = True
        self.use_contrastive = cfg.get("loss", {}).get("use_contrastive", True)

        num_classes = cfg["task"].get("num_classes", 2)
        self.task = cfg["task"].get("type", "binary")

        # ---- encoders ----
        self.image_encoder = build_image_encoder(cfg, meta["image_dim"])
        self.gene_encoder = build_gene_encoder(cfg, meta["gene_dim"])

        # ---- gene target dimensionality ----
        if self.gene_target_type in ("pca", "pathway"):
            # both supply a fixed, precomputed target via the dataset
            self.gene_target_dim = meta.get("gene_pca_target_dim") or \
                m["target"].get("target_dim", 64)
        else:  # learned
            self.gene_target_dim = gene_latent

        # ---- EMA target encoders (built only if requested) ----
        if self.use_ema:
            self.image_target_encoder = build_ema_copy(self.image_encoder)
            if self.gene_target_type == "learned":
                self.gene_target_encoder = build_ema_copy(self.gene_encoder)
            else:
                self.gene_target_encoder = None
        else:
            self.image_target_encoder = None
            self.gene_target_encoder = None

        # ---- predictors: deterministic MLP or conditional latent diffusion ----
        pcfg = m["predictor"]
        ph = pcfg.get("hidden_dim", 256)
        res = pcfg.get("residual", True)
        self.predictor_type = pcfg.get("type", "mlp")  # mlp | diffusion
        self.n_uncertainty = pcfg.get("n_uncertainty_samples", 8)
        if self.predictor_type == "diffusion":
            T = pcfg.get("timesteps", 1000)
            K = pcfg.get("sample_steps", 8)
            depth = pcfg.get("depth", 3)
            # D_{i->g}: distribution over the gene latent target given z_img
            self.img2gene = ConditionalLatentDiffusion(
                self.gene_target_dim, self.latent_dim, T, K, ph, depth)
            # D_{g->i}: distribution over the image latent target given z_gene
            self.gene2img = ConditionalLatentDiffusion(
                self.latent_dim, self.latent_dim, T, K, ph, depth)
        else:
            self.img2gene = ImageToGenePredictor(self.latent_dim,
                                                 self.gene_target_dim, ph, res)
            self.gene2img = GeneToImagePredictor(self.latent_dim,
                                                 self.latent_dim, ph, res)

        # Within-modality masked latent prediction (the canonical I-JEPA task):
        # the SAME gene/image-target diffusion predictor predicts the full target
        # latent from a masked context of the *same* modality. One predictor thus
        # models p(target | context) for both cross-modal and masked-self context,
        # unifying JEPA's two faces.
        self.use_masked_jepa = (pcfg.get("masked_jepa", True)
                                and self.predictor_type == "diffusion")
        self.mask_ratio = pcfg.get("mask_ratio", 0.5)

        # ---- contrastive projection heads ----
        self.img_proj = ProjectionHead(self.latent_dim)
        self.gene_proj = ProjectionHead(self.latent_dim)

        # ---- unimodal aux heads (for analysis / late fusion baseline) ----
        self.image_head = nn.Linear(self.latent_dim, num_classes)
        self.gene_head = nn.Linear(self.latent_dim, num_classes)

        # ---- fusion head over 4 latents ----
        self.fusion_mode = m["fusion"].get("mode", "pathgene_jepa")
        fdims = [self.latent_dim, self.latent_dim, self.gene_target_dim,
                 self.latent_dim]
        self.fusion = build_fusion(
            self.fusion_mode, fdims, num_classes,
            hidden=m["fusion"].get("hidden_dim", 256),
            dropout=m["fusion"].get("dropout", 0.3))

        # ---- gene-anchored residual head ----
        # A direct, full-strength genomics->logit pathway added to the fused
        # logits. By construction the multimodal model can fall back to the
        # genomic prediction (so it is never worse than a gene-only model) while
        # the image/cross-modal pathways add a learned correction on top. The
        # anchor reads a richer gene representation (encoder hidden) to preserve
        # the full molecular signal that a 128-d latent might bottleneck.
        drop = m["fusion"].get("dropout", 0.3)
        self.is_core = self.fusion_mode == "core"
        self.use_gene_anchor = m["fusion"].get("gene_anchor", False) or self.is_core
        if self.use_gene_anchor:
            self.gene_anchor = nn.Sequential(
                nn.Linear(meta["gene_dim"], 256), nn.LayerNorm(256), nn.GELU(),
                nn.Dropout(drop), nn.Linear(256, num_classes))
        if self.is_core:
            # CoRE-Fusion: image expert produces a residual correction to the
            # genomic anchor, gated per-patient by the cross-modal uncertainty.
            res_in = self.latent_dim + self.gene_target_dim + self.latent_dim
            self.image_residual = nn.Sequential(
                nn.Linear(res_in, 256), nn.LayerNorm(256), nn.GELU(),
                nn.Dropout(drop), nn.Linear(256, num_classes))
            # reliability gate alpha = sigmoid(w * (-uncertainty) + b) in (0,1)
            self.gate_w = nn.Parameter(torch.tensor(1.0))
            self.gate_b = nn.Parameter(torch.tensor(0.0))
            # optional hard fallback: alpha=0 (pure anchor) above this uncertainty
            self.core_fallback_tau = m["fusion"].get("core_fallback_tau", None)

    # ---- targets -----------------------------------------------------------
    def _image_target(self, image):
        if self.use_ema and self.image_target_encoder is not None:
            with torch.no_grad():
                return self.image_target_encoder(image)
        return self.image_encoder(image).detach()  # stop-grad approximation

    def _gene_target(self, gene, gene_pca_target):
        if self.gene_target_type in ("pca", "pathway"):
            assert gene_pca_target is not None, \
                f"gene_target_type={self.gene_target_type} requires the " \
                "precomputed target (gene_pca_target) in the batch"
            return gene_pca_target
        if self.use_ema and self.gene_target_encoder is not None:
            with torch.no_grad():
                return self.gene_target_encoder(gene)
        return self.gene_encoder(gene).detach()  # stop-grad approximation

    @torch.no_grad()
    def update_ema(self):
        if not self.use_ema:
            return
        if self.image_target_encoder is not None:
            ema_update(self.image_target_encoder, self.image_encoder, self.ema_decay)
        if self.gene_target_encoder is not None:
            ema_update(self.gene_target_encoder, self.gene_encoder, self.ema_decay)

    # ---- forward -----------------------------------------------------------
    def forward(self, image, gene, gene_pca_target: Optional[torch.Tensor] = None,
                mode: str = "downstream") -> Dict[str, torch.Tensor]:
        z_img = self.image_encoder(image)
        if isinstance(z_img, tuple):
            z_img = z_img[0]
        z_gene = self.gene_encoder(gene)

        out: Dict[str, torch.Tensor] = {
            "image_latent": z_img, "gene_latent": z_gene}

        tgt_gene = self._gene_target(gene, gene_pca_target)
        tgt_img = self._image_target(image)
        out["target_gene_latent"] = tgt_gene
        out["target_image_latent"] = tgt_img
        out["diff_loss_img2gene"] = None
        out["diff_loss_gene2img"] = None
        unc_gene = torch.zeros(z_img.size(0), device=z_img.device)
        unc_img = torch.zeros(z_img.size(0), device=z_img.device)

        if self.predictor_type == "diffusion":
            # Distributional cross-modal latent prediction. The denoising loss is
            # the JEPA objective; sampling yields a posterior mean (fusion/analysis
            # feature) and a per-patient uncertainty.
            if self.use_img2gene:
                out["diff_loss_img2gene"] = self.img2gene.loss(
                    tgt_gene.detach(), z_img)
                pred_gene, unc_gene = self.img2gene.sample_stats(
                    z_img.detach(), self.n_uncertainty)
            else:
                pred_gene = torch.zeros_like(tgt_gene)
            if self.use_gene2img:
                out["diff_loss_gene2img"] = self.gene2img.loss(
                    tgt_img.detach(), z_gene)
                pred_img, unc_img = self.gene2img.sample_stats(
                    z_gene.detach(), self.n_uncertainty)
            else:
                pred_img = torch.zeros_like(tgt_img)
            # ---- within-modality masked latent prediction (I-JEPA) ----
            out["mask_loss_gene"] = None
            out["mask_loss_img"] = None
            if self.use_masked_jepa and self.training:
                # mask a fraction of the raw features to form a context view, then
                # predict the FULL target latent with the same diffusion predictor.
                m_g = (torch.rand_like(gene) > self.mask_ratio).float()
                m_i = (torch.rand_like(image) > self.mask_ratio).float()
                z_gene_ctx = self.gene_encoder(gene * m_g)
                z_img_ctx = self.image_encoder(image * m_i)
                if isinstance(z_img_ctx, tuple):
                    z_img_ctx = z_img_ctx[0]
                if self.use_img2gene:
                    out["mask_loss_gene"] = self.img2gene.loss(
                        tgt_gene.detach(), z_gene_ctx)
                if self.use_gene2img:
                    out["mask_loss_img"] = self.gene2img.loss(
                        tgt_img.detach(), z_img_ctx)
            # deterministic-MLP-style keys kept None (loss path uses diff losses)
            out["predicted_gene_latent"] = None
            out["predicted_image_latent"] = None
        else:
            out["mask_loss_gene"] = None
            out["mask_loss_img"] = None
            pred_gene = self.img2gene(z_img)
            pred_img = self.gene2img(z_gene)
            out["predicted_gene_latent"] = pred_gene if self.use_img2gene else None
            out["predicted_image_latent"] = pred_img if self.use_gene2img else None

        # always-available copies for analysis regardless of ablation flags
        out["pred_gene_full"] = pred_gene
        out["pred_img_full"] = pred_img
        out["gene_uncertainty"] = unc_gene
        out["img_uncertainty"] = unc_img

        # contrastive projections
        if self.use_contrastive:
            out["img_proj"] = self.img_proj(z_img)
            out["gene_proj"] = self.gene_proj(z_gene)
        else:
            out["img_proj"] = None
            out["gene_proj"] = None

        if mode == "embedding":
            return out

        # unimodal logits
        out["image_logits"] = self.image_head(z_img)
        out["gene_logits"] = self.gene_head(z_gene)

        # fusion: zero disabled cross branches so head dims stay fixed
        cross_gene = pred_gene if self.use_img2gene else torch.zeros_like(pred_gene)
        cross_img = pred_img if self.use_gene2img else torch.zeros_like(pred_img)
        feats = [z_img, z_gene, cross_gene, cross_img]
        if self.is_core:
            # genomic anchor + uncertainty-gated complementary image residual
            gene_logit = self.gene_anchor(gene)
            img_res = self.image_residual(
                torch.cat([z_img, cross_gene, cross_img], dim=-1))
            alpha = torch.sigmoid(
                self.gate_w * (-unc_gene).unsqueeze(-1) + self.gate_b)  # (B,1)
            # explicit hard fallback to the genomic anchor (alpha=0) at inference,
            # e.g. when the cross-modal uncertainty exceeds a threshold.
            if getattr(self, "core_fallback_tau", None) is not None:
                alpha = alpha * (unc_gene.unsqueeze(-1) <
                                 self.core_fallback_tau).float()
            out["modality_gate"] = alpha.squeeze(-1)
            out["anchor_logits"] = gene_logit
            out["fusion_logits"] = gene_logit + alpha * img_res
            return out
        if self.fusion_mode in ("uncertainty_gated", "uncertainty", "ugated"):
            unc = torch.stack([unc_gene, unc_img], dim=-1)  # (B, 2)
            logits = self.fusion(feats, uncertainty=unc)
        else:
            logits = self.fusion(feats)
        if self.use_gene_anchor:
            logits = logits + self.gene_anchor(gene)  # direct genomic pathway
        out["fusion_logits"] = logits
        return out


def build_pathgene_jepa(cfg: Dict, meta: Dict) -> PathGeneJEPA:
    return PathGeneJEPA(cfg, meta)
