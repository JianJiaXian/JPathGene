#!/usr/bin/env python3
"""Extract one representative H&E patch from a WSI for use as a figure icon.

Requires numpy, Pillow, and openslide-python (plus the OpenSlide C library):
    pip install numpy Pillow openslide-python
    python extract_patch.py /path/to/slide.svs figures/histology_patch.png

It builds a low-res thumbnail, finds the most tissue-dense window (high
saturation, not background-white), reads that region at full resolution, and
saves a square PNG. Tune --size / --out_px as needed.
"""
import argparse
import numpy as np
from PIL import Image
import openslide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slide")
    ap.add_argument("out")
    ap.add_argument("--size", type=int, default=2048,
                    help="patch side at level 0 (pixels)")
    ap.add_argument("--out_px", type=int, default=512,
                    help="final saved side (pixels)")
    args = ap.parse_args()

    slide = openslide.OpenSlide(args.slide)
    W, H = slide.level_dimensions[0]

    # 1) thumbnail + tissue mask (tissue = saturated and not white)
    tw = 1024
    th = max(1, int(tw * H / W))
    thumb = np.asarray(slide.get_thumbnail((tw, th)).convert("RGB")).astype(float)
    mx, mn = thumb.max(2), thumb.min(2)
    tissue = ((mx - mn) > 18) & (mx < 230)            # bool mask on thumbnail

    # 2) slide the patch window over the thumbnail, pick densest tissue block
    scale = W / thumb.shape[1]                          # level-0 px per thumb px
    win = max(1, int(args.size / scale))               # window size in thumb px
    step = max(1, win // 4)
    best, best_xy = -1.0, (0, 0)
    for ty in range(0, max(1, tissue.shape[0] - win + 1), step):
        for tx in range(0, max(1, tissue.shape[1] - win + 1), step):
            frac = tissue[ty:ty + win, tx:tx + win].mean()
            if frac > best:
                best, best_xy = frac, (tx, ty)
    x0 = int(best_xy[0] * scale)
    y0 = int(best_xy[1] * scale)
    x0 = min(x0, max(0, W - args.size))
    y0 = min(y0, max(0, H - args.size))

    # 3) read at full res, downsample, save
    patch = slide.read_region((x0, y0), 0, (args.size, args.size)).convert("RGB")
    patch = patch.resize((args.out_px, args.out_px), Image.LANCZOS)
    patch.save(args.out)
    print(f"saved {args.out}  (tissue fraction {best:.2f} at level-0 {x0},{y0})")


if __name__ == "__main__":
    main()
