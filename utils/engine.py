"""Shared training / evaluation engine used by all entry scripts."""
import math
import os
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.metrics import compute_metrics


def make_loaders(datasets: Dict, cfg: Dict, batch_size: Optional[int] = None):
    bs = batch_size or cfg["train"]["batch_size"]
    nw = cfg["train"].get("num_workers", 4)
    loaders = {}
    for split, ds in datasets.items():
        loaders[split] = DataLoader(
            ds, batch_size=bs, shuffle=(split == "train"),
            num_workers=nw, drop_last=(split == "train" and len(ds) > bs),
            pin_memory=torch.cuda.is_available())
    return loaders


def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def build_optimizer(model, lr, weight_decay):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, epochs, warmup_epochs=0, kind="cosine"):
    if kind != "cosine":
        return None

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        prog = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def logits_to_prob(logits: torch.Tensor, num_classes: int) -> np.ndarray:
    if logits.dim() == 2 and logits.size(1) >= 2:
        p = torch.softmax(logits, dim=-1)
    else:
        pos = torch.sigmoid(logits.squeeze(-1))
        p = torch.stack([1 - pos, pos], dim=-1)
    return p.detach().cpu().numpy()


@torch.no_grad()
def predict(model, loader, device, num_classes: int):
    model.eval()
    ys, ps, ids = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["image"], batch["gene"],
                    gene_pca_target=batch.get("gene_pca_target"),
                    mode="downstream") if _is_pathgene(model) else \
            model(batch["image"], batch["gene"])
        ps.append(logits_to_prob(out["fusion_logits"], num_classes))
        ys.append(batch["label"].cpu().numpy())
        ids.extend(batch["patient_id"])
    return np.concatenate(ys), np.concatenate(ps), ids


def _is_pathgene(model):
    return model.__class__.__name__ == "PathGeneJEPA"


def evaluate_split(model, loader, device, num_classes: int):
    y, p, ids = predict(model, loader, device, num_classes)
    metrics = compute_metrics(y, p, num_classes)
    return metrics, (y, p, ids)


def save_checkpoint(path, model, cfg, meta, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg, "meta": meta,
                "extra": extra or {}}, path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)
