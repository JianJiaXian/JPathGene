#!/usr/bin/env python
"""Stage 1 for the LUAD qualitative CAM figure: fetch + probe assets.

For the four curated LUAD patients, download (a) the open-access diagnostic WSI
from the GDC and (b) the per-tile UNI2-h feature h5 from the HuggingFace dataset
the cohort was built from, then verify that the h5 carries spatial ``coords`` and
that the slide thumbnail renders. Nothing here trains or fabricates -- it only
stages real data and reports what is present, so Stage 2 can build the figure.

Run inside the container via sbatch (compute nodes have internet here).
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SLIDE_DIR = "data/_luad_slides"
TILE_DIR = "data/_luad_tiles"
THUMB_DIR = "outputs/tcga_luad_stage/figures/_thumbs"
HF_REPO = "W8Yi/tcga-wsi-uni2h-features"

# (patient_id, gdc_file_id, file_name) from the GDC open-access query.
WSIS = [
    ("TCGA-53-A4EZ", "235b48c5-3e04-48f5-bc7f-5f04505abe48",
     "TCGA-53-A4EZ-01Z-00-DX1.5D155F0B-A677-4589-AF00-A4C451F5B6B6.svs"),
    ("TCGA-86-A456", "5d3ccaae-0d6d-49b5-bd48-f09065d6fd47",
     "TCGA-86-A456-01Z-00-DX1.5C7CBF9B-0AE3-4776-9434-296AA0C605CC.svs"),
    ("TCGA-NJ-A4YP", "37885959-e972-4fdc-8410-d06a4d48185b",
     "TCGA-NJ-A4YP-01Z-00-DX1.148BFF66-4DC7-468C-8783-97F27C2E1245.svs"),
    ("TCGA-55-7284", "03309e6b-eb69-4009-9f7b-74b0a0f00de4",
     "TCGA-55-7284-01Z-00-DX1.68b95b9b-1aab-4f03-aad3-1132467b7499.svs"),
]


def ensure(pkg):
    try:
        __import__(pkg)
    except Exception:
        os.system(f"pip install --user -q {pkg}")
        __import__(pkg)


def download_wsis():
    os.makedirs(SLIDE_DIR, exist_ok=True)
    for pid, fid, fname in WSIS:
        dst = os.path.join(SLIDE_DIR, f"{pid}.svs")
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
            print(f"[wsi] {pid} present ({os.path.getsize(dst)/1e9:.2f} GB)")
            continue
        url = f"https://api.gdc.cancer.gov/data/{fid}"
        print(f"[wsi] downloading {pid} <- {url}", flush=True)
        urllib.request.urlretrieve(url, dst)
        print(f"[wsi] saved {dst} ({os.path.getsize(dst)/1e9:.2f} GB)", flush=True)


def download_tiles():
    ensure("huggingface_hub")
    from huggingface_hub import HfApi, hf_hub_download
    os.makedirs(TILE_DIR, exist_ok=True)
    files = HfApi().list_repo_files(HF_REPO, repo_type="dataset")
    luad = [f for f in files if f.startswith("TCGA-LUAD/") and f.endswith(".h5")]
    dx = [f for f in luad if "-DX" in os.path.basename(f).upper()] or luad
    by_pid = {}
    for f in sorted(dx):
        pid = "-".join(os.path.basename(f).split("-")[:3])
        by_pid.setdefault(pid, f)
    out = {}
    for pid, _, _ in WSIS:
        rel = by_pid.get(pid)
        if rel is None:
            print(f"[tile] {pid} NOT in HF repo"); continue
        p = hf_hub_download(HF_REPO, rel, repo_type="dataset", local_dir=TILE_DIR)
        out[pid] = p
        print(f"[tile] {pid} <- {rel}", flush=True)
    return out


def probe_h5(tile_paths):
    ensure("h5py")
    import h5py
    import numpy as np
    for pid, p in tile_paths.items():
        with h5py.File(p, "r") as h:
            keys = list(h.keys())
            info = {}
            for k in keys:
                info[k] = tuple(h[k].shape)
            attrs = {}
            for k in keys:
                attrs[k] = {ak: h[k].attrs[ak] for ak in h[k].attrs}
            print(f"[h5] {pid}: keys={info} attrs={json.dumps(attrs, default=str)[:200]}")
            if "coords" in h:
                c = np.asarray(h["coords"][:8])
                print(f"      coords sample={c.tolist()}")


def render_thumbs():
    ensure("tiffslide")
    import tiffslide
    import numpy as np
    from PIL import Image
    os.makedirs(THUMB_DIR, exist_ok=True)
    for pid, _, _ in WSIS:
        src = os.path.join(SLIDE_DIR, f"{pid}.svs")
        if not os.path.exists(src):
            print(f"[thumb] {pid} slide missing"); continue
        sl = tiffslide.TiffSlide(src)
        W, H = sl.dimensions
        tw = 1024
        th = max(1, int(tw * H / W))
        thumb = sl.get_thumbnail((tw, th)).convert("RGB")
        thumb.save(os.path.join(THUMB_DIR, f"{pid}.png"))
        print(f"[thumb] {pid} level0={W}x{H} thumb={thumb.size} "
              f"mpp={sl.properties.get('tiffslide.mpp-x')}", flush=True)


def main():
    download_wsis()
    tile_paths = download_tiles()
    probe_h5(tile_paths)
    render_thumbs()
    print("[prep] done")


if __name__ == "__main__":
    main()
