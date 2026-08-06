"""Loss functions for JPathGene.

Total objective:
  L = lambda_img2gene * L_img->gene      (JEPA latent prediction)
    + lambda_gene2img * L_gene->img      (JEPA latent prediction)
    + lambda_align    * L_contrastive    (optional image-gene InfoNCE)
    + lambda_cls      * L_classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def jepa_distance(pred, target, kind: str = "cosine"):
    """Latent prediction distance. `target` must already be detached/stop-grad."""
    if kind == "cosine":
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        return (1.0 - (pred * target).sum(dim=-1)).mean()
    elif kind == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    elif kind == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"unknown jepa distance: {kind}")


def classification_loss(logits, target, task: str = "binary"):
    if task == "multiclass":
        return F.cross_entropy(logits, target.long())
    # binary / multilabel-friendly: use 2-way CE on (B, 2) logits if provided,
    # otherwise BCE on single-logit.
    if logits.dim() == 2 and logits.size(1) > 1:
        return F.cross_entropy(logits, target.long())
    return F.binary_cross_entropy_with_logits(
        logits.squeeze(-1), target.float())


class PathGeneLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        lc = cfg.get("loss", {})
        self.l_i2g = lc.get("lambda_img2gene", 1.0)
        self.l_g2i = lc.get("lambda_gene2img", 1.0)
        self.l_align = lc.get("lambda_align", 0.1)
        self.l_cls = lc.get("lambda_cls", 1.0)
        self.l_mask = lc.get("lambda_masked_jepa", 0.5)
        self.use_contrastive = lc.get("use_contrastive", True)
        self.temp = lc.get("contrastive_temp", 0.1)
        self.distance = cfg.get("jepa", {}).get("distance", "cosine")
        self.task = cfg.get("task", {}).get("type", "binary")

    def forward(self, out: dict, labels=None,
                include_cls: bool = True, include_jepa: bool = True):
        from models.contrastive import info_nce

        device = out["image_latent"].device
        zero = torch.zeros((), device=device)
        terms = {"img_to_gene": zero, "gene_to_img": zero,
                 "contrastive": zero, "classification": zero,
                 "masked_jepa": zero}

        if include_jepa:
            # diffusion predictor: denoising score-matching loss IS the JEPA term
            if out.get("diff_loss_img2gene") is not None:
                terms["img_to_gene"] = out["diff_loss_img2gene"]
            elif out.get("predicted_gene_latent") is not None:
                terms["img_to_gene"] = jepa_distance(
                    out["predicted_gene_latent"],
                    out["target_gene_latent"].detach(), self.distance)
            if out.get("diff_loss_gene2img") is not None:
                terms["gene_to_img"] = out["diff_loss_gene2img"]
            elif out.get("predicted_image_latent") is not None:
                terms["gene_to_img"] = jepa_distance(
                    out["predicted_image_latent"],
                    out["target_image_latent"].detach(), self.distance)
            if (self.use_contrastive and out.get("img_proj") is not None
                    and out["img_proj"].size(0) > 1):
                terms["contrastive"] = info_nce(
                    out["img_proj"], out["gene_proj"], self.temp)
            # within-modality masked latent prediction (I-JEPA)
            mg, mi = out.get("mask_loss_gene"), out.get("mask_loss_img")
            if mg is not None or mi is not None:
                terms["masked_jepa"] = ((mg if mg is not None else zero)
                                        + (mi if mi is not None else zero))

        if include_cls and labels is not None and out.get("fusion_logits") is not None:
            terms["classification"] = classification_loss(
                out["fusion_logits"], labels, self.task)

        total = (self.l_i2g * terms["img_to_gene"]
                 + self.l_g2i * terms["gene_to_img"]
                 + self.l_align * terms["contrastive"]
                 + self.l_mask * terms["masked_jepa"]
                 + self.l_cls * terms["classification"])
        terms["total"] = total
        return total, {k: float(v.detach()) for k, v in terms.items()}
