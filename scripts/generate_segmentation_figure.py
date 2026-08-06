#!/usr/bin/env python
"""Segmentation figures + comparison table for TCGA-LGG radiogenomics.

Always produces a conceptual conditioning schematic. When trained checkpoints
exist it also renders a qualitative grid (image / GT / predictions per gene mode)
from real test slices, and a Dice/IoU comparison table from the analysis JSONs.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utils.visualization import PALETTE, apply_style, arrow, rounded_box, save  # noqa: E402
from utils.io import OUT_ROOT as _OUT  # noqa: E402

SCHEM = os.path.join(_OUT, "figures/lgg_gene_conditioned_unet.pdf")
QUAL = os.path.join(_OUT, "figures/lgg_qualitative.pdf")
PAPER_TAB = "Springer Lecture Notes in Computer Science/tables/lgg_segmentation.tex"


def schematic():
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 16); ax.set_ylim(0, 8); ax.axis("off")
    ax.text(8, 7.5, "Gene-conditioned U-Net with JEPA-imputed genomics",
            ha="center", fontsize=13.5, fontweight="bold")
    rounded_box(ax, (0.3, 4.6), 2.4, 1.2, "MRI slice\n(3 channels)", PALETTE["image"])
    rounded_box(ax, (3.4, 4.6), 2.3, 1.2, "U-Net\nencoder", PALETTE["image"])
    rounded_box(ax, (6.3, 4.6), 2.0, 1.2, "bottleneck", PALETTE["neutral"])
    rounded_box(ax, (9.0, 4.6), 2.3, 1.2, "U-Net\ndecoder", PALETTE["fusion"])
    rounded_box(ax, (12.1, 4.6), 2.4, 1.2, "tumor\nmask", PALETTE["accent"])
    for a, b in [((2.7, 5.2), (3.4, 5.2)), ((5.7, 5.2), (6.3, 5.2)),
                 ((8.3, 5.2), (9.0, 5.2)), ((11.3, 5.2), (12.1, 5.2))]:
        arrow(ax, a, b)
    # genetics path
    rounded_box(ax, (0.3, 1.4), 2.4, 1.1, "genomic\nprofile", PALETTE["gene"])
    rounded_box(ax, (3.4, 1.4), 2.3, 1.1, "gene\nencoder", PALETTE["gene"])
    rounded_box(ax, (6.3, 1.4), 2.0, 1.1, "gene latent", PALETTE["target"])
    rounded_box(ax, (9.0, 2.7), 2.3, 1.0, "FiLM\nconditioning", PALETTE["predict"])
    arrow(ax, (2.7, 1.95), (3.4, 1.95), color=PALETTE["gene"])
    arrow(ax, (5.7, 1.95), (6.3, 1.95), color=PALETTE["gene"])
    arrow(ax, (8.3, 1.95), (9.7, 2.7), color=PALETTE["target"], rad=0.2)
    arrow(ax, (10.1, 3.7), (10.1, 4.6), color=PALETTE["predict"])
    # JEPA imputation: bottleneck -> gene latent (image->gene)
    arrow(ax, (7.3, 4.6), (7.3, 2.5), color=PALETTE["image"], ls="--", rad=0.0)
    ax.text(7.45, 3.5, "JEPA: image$\\rightarrow$gene\n(genomics optional)",
            ha="left", fontsize=8.5, color=PALETTE["neutral"], style="italic")
    fig.text(0.5, 0.02, "Genetics conditions segmentation via FiLM; at test the "
             "genetic latent can be imputed from the image (JEPA).",
             ha="center", fontsize=10, style="italic", color=PALETTE["gene"])
    save(fig, SCHEM)


ARCH_NAME = {"unet": "U-Net", "attunet": "Attention U-Net",
             "transunet": "TransUNet (transformer)", "diffseg": "DiffSeg (diffusion, ours)"}
GENE_NAME = {"none": "none", "true": "real genomics", "jepa": "JEPA-imputed"}
PAPER_TAB2 = "Springer Lecture Notes in Computer Science/tables/lgg_genetics.tex"


def _load_seg_results():
    """Return {(arch,mode): {'dice','iou'}} from lggseg_{arch}_{mode}_test.json."""
    res = {}
    for p in glob.glob(os.path.join(_OUT, "analysis", "lggseg_*_test.json")):
        try:
            d = json.load(open(p))
            stem = os.path.basename(p)[len("lggseg_"):-len("_test.json")]
            arch, mode = stem.rsplit("_", 1)
            res[(arch, mode)] = d["test"]
        except Exception:
            continue
    return res


def _emit(path, caption, label, header, rows, note=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    col = "l" + "c" * (len(header) - 1)
    lines = [r"\begin{table}[t]", r"\centering", "\\caption{%s}" % caption,
             "\\label{%s}" % label, "\\begin{tabular}{%s}" % col, r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append("\\\\[2pt]\\footnotesize %s" % note)
    lines.append(r"\end{table}")
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def comparison_table():
    res = _load_seg_results()
    # Table 1: architecture comparison (image-only, gene_mode=none)
    header1 = ["Architecture", "Dice", "IoU"]
    rows1 = []
    for arch in ["unet", "attunet", "transunet", "diffseg"]:
        r = res.get((arch, "none"))
        if r:
            rows1.append([ARCH_NAME[arch], f"{r['dice']:.3f}", f"{r['iou']:.3f}"])
    if not rows1:
        rows1 = [["DiffSeg (diffusion, ours)", r"\TODO{--}", r"\TODO{--}"]]
        note1 = "Run train\\_segmentation.py to populate."
    else:
        note1 = ("TCGA-LGG tumor segmentation, image-only. All backbones share "
                 "the same conditioning interface for a fair comparison.")
    _emit(PAPER_TAB, note1, "tab:segarch", header1, rows1)

    # Table 2: genetic conditioning (none / real / JEPA-imputed) per backbone
    header2 = ["Backbone", "Genetics", "Dice", "IoU"]
    rows2 = []
    for arch in ["unet", "transunet", "diffseg"]:
        for mode in ["none", "true", "jepa"]:
            r = res.get((arch, mode))
            if r:
                rows2.append([ARCH_NAME[arch], GENE_NAME[mode],
                              f"{r['dice']:.3f}", f"{r['iou']:.3f}"])
    if rows2:
        _emit(PAPER_TAB2,
              "Effect of genetic conditioning. `JEPA-imputed' needs no genomics "
              "at test time (imputed from the image).", "tab:seggene",
              header2, rows2)
    else:
        _emit(PAPER_TAB2, "Effect of genetic conditioning.", "tab:seggene",
              header2, [["DiffSeg (ours)", "JEPA-imputed", r"\TODO{--}", r"\TODO{--}"]])

    if os.path.exists(SCHEM):
        import shutil
        shutil.copyfile(SCHEM, "Springer Lecture Notes in Computer Science/"
                              "figures/lgg_gene_conditioned_unet.pdf")


def qualitative(cfg_path):
    """Render image/GT/pred grid from trained checkpoints (real test slices)."""
    try:
        import torch
        from datasets.lgg_radiogenomics_dataset import build_lgg_datasets
        from train_segmentation import build_seg_model
        from utils.io import load_config, get_device
    except Exception as e:  # noqa
        print(f"[seg-fig] qualitative skipped: {e}")
        return
    # show image-only U-Net, transformer+JEPA, and diffusion+JEPA when available
    want = [("unet", "none", "U-Net"), ("transunet", "jepa", "TransUNet+gene"),
            ("diffseg", "jepa", "DiffSeg+gene (ours)")]
    cols_models = []
    for arch, mode, label in want:
        p = os.path.join(_OUT, "checkpoints", f"best_lggseg_{arch}_{mode}.pth")
        if os.path.exists(p):
            cols_models.append((arch, mode, label, p))
    if not cols_models:
        print("[seg-fig] no seg checkpoints; schematic only")
        return
    cfg = load_config(cfg_path)
    device = get_device(cfg.get("device", "cuda"))
    datasets, meta = build_lgg_datasets(cfg)
    ds = datasets["test"]
    rng = np.random.RandomState(0)
    pick = []
    for i in rng.permutation(len(ds)):
        item = ds[int(i)]
        if item["mask"].sum() > 30:
            pick.append(item)
        if len(pick) == 4:
            break
    if not pick:
        return
    models = []
    for arch, mode, label, p in cols_models:
        st = torch.load(p, map_location=device, weights_only=False)
        m = build_seg_model(arch, meta, cfg).to(device)
        m.load_state_dict(st["model"])
        m.eval()
        models.append((m, mode, label))

    apply_style()
    cols = ["MRI", "GT"] + [lbl for _, _, lbl in models]
    fig, axes = plt.subplots(len(pick), len(cols),
                             figsize=(2.1 * len(cols), 2.1 * len(pick)))
    for r, item in enumerate(pick):
        img = item["image"].numpy().transpose(1, 2, 0)
        gt = item["mask"][0].numpy()
        axes[r, 0].imshow(img)
        axes[r, 1].imshow(img); axes[r, 1].imshow(gt, alpha=0.4, cmap="autumn")
        with torch.no_grad():
            x = item["image"].unsqueeze(0).to(device)
            g = item["gene"].unsqueeze(0).to(device)
            for c, (m, gm, _) in enumerate(models):
                pr = torch.sigmoid(m(x, g, gene_mode=gm)["logits"])[0, 0].cpu().numpy()
                axes[r, 2 + c].imshow(img)
                axes[r, 2 + c].imshow(pr > 0.5, alpha=0.4, cmap="winter")
        for c in range(len(cols)):
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=10)
    save(fig, QUAL)
    if os.path.exists(QUAL):
        import shutil
        shutil.copyfile(QUAL, "Springer Lecture Notes in Computer Science/"
                              "figures/lgg_qualitative.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tcga_lgg_seg.yaml")
    args = ap.parse_args()
    ensure = os.makedirs
    ensure(os.path.join(_OUT, "figures"), exist_ok=True)
    schematic()
    qualitative(args.config)
    comparison_table()
    print("segmentation figures/table done.")


if __name__ == "__main__":
    main()
