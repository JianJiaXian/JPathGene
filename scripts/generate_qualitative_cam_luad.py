#!/usr/bin/env python
"""Stage 2: LUAD qualitative figure with real input-image thumbnails + a
post-hoc per-tile Grad-CAM overlay.

Pipeline (all real, nothing fabricated):
  1. Train the SAME model whose OOF predictions appear in the figure
     -- JPathGene "ours" = MLP predictor + concat fusion + gene anchor
     (run_qualitative.py) -- on every LUAD-stage patient EXCEPT the four
     curated cases, which are held out.
  2. For each held-out case, mean-pool its per-tile UNI2-h features (the exact
     model input), backprop the predicted-class logit to the pooled input to get
     a class-direction g, and score each tile by ReLU(g . z_tile)
     (Grad-CAM / class-activation over the tile grid).
  3. Rasterise the tile scores onto the real WSI thumbnail (via tile coords) and
     overlay -- an honest *post-hoc attribution*, not a native attention map.
  4. Figure: [input image] [Grad-CAM overlay] [real per-method predictions].

Run inside the container via sbatch (needs torch, h5py, tiffslide, matplotlib).
Assets are staged by scripts/prep_luad_cam_assets.py.
"""
import csv
import glob
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

from datasets.image_gene_feature_dataset import build_datasets  # noqa: E402
from run_crossval_compare import (_apply_override, _apply_flags,  # noqa: E402
                                  _train_pathgene, _make_fold_splits)
from models.pathgene_jepa import build_pathgene_jepa  # noqa: E402
from utils.engine import make_loaders  # noqa: E402
from utils.io import load_config, get_device  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from utils.visualization import PALETTE, apply_style  # noqa: E402

CONFIG = "configs/tcga_luad_stage.yaml"
DATA = "data/TCGA_LUAD_UNI"
TILE_DIR = "data/_luad_tiles"
THUMB_DIR = "outputs/tcga_luad_stage/figures/_thumbs"
SLIDE_DIR = "data/_luad_slides"
PREDS = "outputs/tcga_luad_stage/analysis/qualitative_preds.csv"
CAM_CACHE = "outputs/tcga_luad_stage/analysis/cam_cache.npz"
OUT = "outputs/tcga_luad_stage/figures/qualitative_luad_cases.pdf"
PAPER = ("JPathGene__Cross_Modal_Latent_Diffusion__10_/figures/qualitative_luad.pdf")
EXPORT_DIR = "outputs/tcga_luad_stage/figures/export"

CASES = ["TCGA-53-A4EZ", "TCGA-55-7284", "TCGA-NJ-A4YP", "TCGA-86-A456"]
OURS = dict(predictor="mlp", target="learned", fusion="concat", anchor=True)
METHODS = ["image_only", "gene_only", "concat_fusion", "JPathGene (ours)"]
DISP = {"image_only": "Image-only", "gene_only": "Gene-only",
        "concat_fusion": "Concat fusion", "JPathGene (ours)": "JPathGene"}


# -----------------------------------------------------------------------------
def find_h5(pid):
    hits = glob.glob(os.path.join(TILE_DIR, "**", f"{pid}-*DX*.h5"), recursive=True)
    return hits[0] if hits else None


def load_tiles(pid):
    import h5py
    p = find_h5(pid)
    if p is None:
        return None, None
    with h5py.File(p, "r") as h:
        feats = np.asarray(h["features"][:], dtype=np.float32)
        feats = feats.reshape(-1, feats.shape[-1])          # (N,1536), drop batch dim
        coords = None
        if "coords" in h:
            coords = np.asarray(h["coords"][:], dtype=np.float64).reshape(-1, 2)
    return feats, coords


def image_scaler_from_train(train_ids):
    """Reproduce the dataset's standard scaler (train-only mean/std) for images."""
    ids = set(train_ids)
    rows = []
    with open(os.path.join(DATA, "image_features.csv")) as f:
        next(f)
        for line in f:
            pid, rest = line.split(",", 1)
            if pid in ids:
                rows.append(np.fromstring(rest, sep=","))
    X = np.asarray(rows, dtype=np.float64)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


# -----------------------------------------------------------------------------
SEEDS = [42, 1, 7, 11, 23]   # same seeds as the OOF predictions in the CSV
EPOCHS = 80


def cam_one(model, device, feats, mu, sd, gene_vec, true_cls):
    """Per-tile Grad-CAM toward the TRUE class: ReLU(g . z_tile),
    g = d logit_true / d pooled_input. Returns (per-tile importance, p_adv)."""
    z = (feats - mu) / sd                         # (N,1536) scaled tiles
    pooled = torch.tensor(z.mean(0, keepdims=True), dtype=torch.float32,
                          device=device, requires_grad=True)
    gene = torch.tensor(gene_vec[None], dtype=torch.float32, device=device)
    model.eval()
    out = model(pooled, gene, mode="downstream")
    logit = out["fusion_logits"]
    model.zero_grad(set_to_none=True)
    logit[0, true_cls].backward()
    g = pooled.grad.detach().cpu().numpy()[0]     # (1536,)
    imp = np.maximum(z @ g, 0.0)                  # (N,) ReLU class-activation
    if imp.max() > imp.min():
        imp = (imp - imp.min()) / (imp.max() - imp.min())
    prob = torch.softmax(logit, -1)[0, 1].item()
    return imp, prob


def oof_cam(device):
    """Reproduce the OOF protocol for the 'ours' model and, for each target
    patient, average the Grad-CAM (toward its TRUE class) over the fold-models
    that held it out -- so the CAM matches the ensemble predictions in the panel.
    """
    if os.path.exists(CAM_CACHE) and os.environ.get("CAM_RECOMPUTE") != "1":
        out = np.load(CAM_CACHE, allow_pickle=True)["out"].item()
        print(f"[cam] loaded cached attributions {CAM_CACHE}")
        return out
    cfg = load_config(CONFIG)
    cfg = _apply_override(cfg, "pathgene", OURS)
    ds_all, _ = build_datasets(cfg)
    ids, labels = [], []
    for s in ("train", "val", "test"):
        ids += list(ds_all[s].ids); labels += list(ds_all[s].label)
    ids, labels = np.array(ids), np.array(labels)

    raw = {pid: load_tiles(pid) for pid in CASES}          # (feats, coords)
    acc = {pid: {"cam": [], "prob": [], "label": None} for pid in CASES}

    for seed in SEEDS:
        cfg["seed"] = seed
        for override in _make_fold_splits(ids, labels, 5, seed):
            here = [p for p in CASES if override.get(p) == "test"]
            if not here:
                continue
            set_seed(seed)
            datasets, meta = build_datasets(cfg, override_splits=override)
            loaders = make_loaders(datasets, cfg)
            model = build_pathgene_jepa(cfg, meta).to(device)
            model = _apply_flags(model, OURS)
            model = _train_pathgene(model, loaders, cfg, device, EPOCHS)
            mu, sd = image_scaler_from_train(datasets["train"].ids)
            gene_by = {pid: datasets["test"].gene[i]
                       for i, pid in enumerate(datasets["test"].ids)}
            lab_by = {pid: int(datasets["test"].label[i])
                      for i, pid in enumerate(datasets["test"].ids)}
            for pid in here:
                feats, _ = raw[pid]
                imp, prob = cam_one(model, device, feats, mu, sd,
                                    gene_by[pid], lab_by[pid])
                acc[pid]["cam"].append(imp)
                acc[pid]["prob"].append(prob)
                acc[pid]["label"] = lab_by[pid]
            print(f"[cam] seed={seed} fold trained; targets={here}", flush=True)

    out = {}
    for pid in CASES:
        if not acc[pid]["cam"]:
            print(f"[cam] {pid}: no held-out fold?!"); continue
        imp = np.mean(np.stack(acc[pid]["cam"], 0), 0)
        if imp.max() > imp.min():
            imp = (imp - imp.min()) / (imp.max() - imp.min())
        out[pid] = {"imp": imp, "coords": raw[pid][1],
                    "prob": float(np.mean(acc[pid]["prob"])),
                    "label": acc[pid]["label"], "n": len(acc[pid]["cam"])}
        print(f"[cam] {pid}: n_models={out[pid]['n']} "
              f"reproduced_mean_p_adv={out[pid]['prob']:.3f} "
              f"label={out[pid]['label']}", flush=True)
    os.makedirs(os.path.dirname(CAM_CACHE), exist_ok=True)
    np.savez(CAM_CACHE, out=np.array(out, dtype=object))
    return out


def rasterize(coords, imp, thumb_size, slide_w):
    """Place tile scores on a thumbnail-sized heatmap via level-0 coords."""
    tw, th = thumb_size
    if coords is None:
        return None
    xs, ys = coords[:, 0], coords[:, 1]
    stride = np.median(np.diff(np.unique(np.round(xs)))) if len(np.unique(xs)) > 1 \
        else (slide_w / 50.0)
    stride = max(stride, 1.0)
    gx = np.round((xs - xs.min()) / stride).astype(int)
    gy = np.round((ys - ys.min()) / stride).astype(int)
    grid = np.full((gy.max() + 1, gx.max() + 1), np.nan)
    for i in range(len(imp)):
        cur = grid[gy[i], gx[i]]
        grid[gy[i], gx[i]] = imp[i] if np.isnan(cur) else max(cur, imp[i])
    mask = ~np.isnan(grid)
    grid_filled = np.where(mask, grid, 0.0)
    # upscale to thumbnail size (bilinear) for a smooth overlay
    heat = np.asarray(Image.fromarray((grid_filled * 255).astype(np.uint8))
                      .resize((tw, th), Image.BILINEAR), dtype=np.float32) / 255.0
    tissue = np.asarray(Image.fromarray((mask * 255).astype(np.uint8))
                        .resize((tw, th), Image.BILINEAR), dtype=np.float32) / 255.0
    # opacity scales with importance so only hot regions are coloured and the
    # underlying histology stays visible elsewhere (gamma sharpens the contrast)
    alpha = 0.85 * (heat ** 0.6) * tissue
    return heat, alpha


def load_thumb(pid):
    p = os.path.join(THUMB_DIR, f"{pid}.png")
    if os.path.exists(p):
        return Image.open(p).convert("RGB")
    import tiffslide
    sl = tiffslide.TiffSlide(os.path.join(SLIDE_DIR, f"{pid}.svs"))
    W, H = sl.dimensions
    tw = 1024
    return sl.get_thumbnail((tw, max(1, int(tw * H / W)))).convert("RGB")


def slide_width(pid):
    import tiffslide
    return tiffslide.TiffSlide(os.path.join(SLIDE_DIR, f"{pid}.svs")).dimensions[0]


# -----------------------------------------------------------------------------
def main():
    device = get_device("cuda")
    rows = {r["patient_id"]: r for r in csv.DictReader(open(PREDS))}
    cases = [rows[p] for p in CASES if p in rows]
    cam = oof_cam(device)   # ensemble Grad-CAM toward true class, per patient

    panels = []
    for r in cases:
        pid = r["patient_id"]
        thumb = load_thumb(pid)
        ras = None
        if pid in cam:
            ras = rasterize(cam[pid]["coords"], cam[pid]["imp"], thumb.size,
                            slide_width(pid))
        panels.append((r, thumb, ras))

    apply_style()
    n = len(panels)
    fig = plt.figure(figsize=(9.8, 1.85 * n + 1.2))
    gs = fig.add_gridspec(n, 3, width_ratios=[1.2, 1.2, 3.0],
                          hspace=0.55, wspace=0.38,
                          left=0.175, right=0.985, top=0.84, bottom=0.10)

    for i, (r, thumb, ras) in enumerate(panels):
        pid = r["patient_id"]; y = int(r["label"])
        axi, axc, axp = (fig.add_subplot(gs[i, j]) for j in range(3))
        axi.imshow(thumb); axi.set_xticks([]); axi.set_yticks([])
        axc.imshow(thumb)
        if ras is not None:
            heat, alpha = ras
            axc.imshow(heat, cmap="jet", alpha=alpha, vmin=0, vmax=1)
        axc.set_xticks([]); axc.set_yticks([])
        axi.set_ylabel(f"{pid}\n(y={y}, {'advanced' if y else 'early'})",
                       fontsize=8.0, rotation=90, va="center", labelpad=6)
        if i == 0:
            axi.set_title("Input histology (WSI)", fontsize=9.3,
                          color=PALETTE["image"], fontweight="bold", pad=6)
            axc.set_title("Grad-CAM (image pathway)", fontsize=9.3,
                          color=PALETTE["gene"], fontweight="bold", pad=6)
            axp.set_title("Predicted $P(\\mathrm{advanced\\ stage})$",
                          fontsize=9.3, fontweight="bold", pad=6)

        axp.axvspan(0.5 if y == 1 else 0.0, 1.0 if y == 1 else 0.5,
                    color=PALETTE["light"], zorder=0)
        axp.axvline(0.5, color="#333", lw=1.3, ls="--", zorder=1)
        for mi, m in enumerate(METHODS):
            p = float(r[m]); ok = (p > 0.5) == bool(y); ours = m == "JPathGene (ours)"
            yy = len(METHODS) - 1 - mi
            axp.scatter([p], [yy], marker="o" if ok else "X",
                        s=190 if ours else 95,
                        color=PALETTE["predict"] if ok else PALETTE["gene"],
                        edgecolor="black" if ours else "white",
                        linewidth=1.6 if ours else 0.7, zorder=3)
            axp.text(p, yy + 0.30, f"{p:.2f}", ha="center", va="bottom",
                     fontsize=7.6, fontweight="bold" if ours else "normal",
                     color="#222")
        axp.set_xlim(-0.02, 1.02); axp.set_ylim(-0.6, len(METHODS) - 0.4)
        axp.set_yticks(range(len(METHODS)))
        axp.set_yticklabels([DISP[m] for m in reversed(METHODS)], fontsize=8.6)
        axp.tick_params(axis="y", length=0); axp.grid(axis="y", visible=False)
        if i < n - 1:
            axp.set_xticklabels([])

    fig.text(0.71, 0.035,
             "predicted probability (threshold 0.5; shaded = correct half)",
             ha="center", fontsize=8.3, color=PALETTE["neutral"])
    leg = [Line2D([0], [0], marker="o", color="w",
                  markerfacecolor=PALETTE["predict"], markersize=9, label="correct"),
           Line2D([0], [0], marker="X", color="w",
                  markerfacecolor=PALETTE["gene"], markersize=9, label="wrong"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
                  markeredgecolor="black", markersize=11,
                  label="JPathGene (emphasised)")]
    fig.legend(handles=leg, loc="upper center", ncol=3, fontsize=8.6,
               bbox_to_anchor=(0.5, 0.998))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT); fig.savefig(OUT[:-4] + ".png")
    if os.path.isdir(os.path.dirname(PAPER)):
        shutil.copyfile(OUT, PAPER)
    print(f"[cam] saved {OUT} (+paper copy) cases={CASES}")

    # ---- export individual images into one folder ----
    os.makedirs(EXPORT_DIR, exist_ok=True)
    cmap = plt.get_cmap("jet")
    for r, thumb, ras in panels:
        pid = r["patient_id"]; y = int(r["label"])
        tag = f"{pid}_y{y}_{'advanced' if y else 'early'}"
        thumb.save(os.path.join(EXPORT_DIR, f"{tag}_1_input.png"))
        base = np.asarray(thumb, dtype=np.float32) / 255.0
        if ras is not None:
            heat, alpha = ras
            rgb = cmap(heat)[..., :3]
            a = alpha[..., None]
            over = (base * (1 - a) + rgb * a).clip(0, 1)
            Image.fromarray((over * 255).astype(np.uint8)).save(
                os.path.join(EXPORT_DIR, f"{tag}_2_gradcam.png"))
    shutil.copyfile(OUT, os.path.join(EXPORT_DIR, "figure3_combined.pdf"))
    shutil.copyfile(OUT[:-4] + ".png",
                    os.path.join(EXPORT_DIR, "figure3_combined.png"))
    print(f"[cam] exported per-image PNGs -> {EXPORT_DIR}")


if __name__ == "__main__":
    main()
