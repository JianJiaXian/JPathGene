#!/usr/bin/env python3
"""Download one open-access TCGA-BRCA diagnostic H&E slide (.svs) from the GDC.

Queries the GDC files API for the smallest open-access TCGA-BRCA diagnostic
slide and downloads it. Needs internet access; slides are ~0.3-1.5 GB.

    python get_tcga_brca_slide.py                 # -> ./tcga_brca_slides/
    python get_tcga_brca_slide.py --dest /data    # custom destination

Then pass the printed path to extract_patch.py.
"""
import argparse
import json
import os
import urllib.request

FILES_API = "https://api.gdc.cancer.gov/files"
DATA_API = "https://api.gdc.cancer.gov/data"


def query_smallest_slide():
    """Return (file_id, file_name) of the smallest open TCGA-BRCA slide."""
    payload = {
        "filters": {"op": "and", "content": [
            {"op": "in", "content": {"field": "cases.project.project_id",
                                     "value": ["TCGA-BRCA"]}},
            {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
            {"op": "in", "content": {"field": "experimental_strategy",
                                     "value": ["Diagnostic Slide"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ]},
        "fields": "file_id,file_name,file_size",
        "format": "JSON",
        "size": "1",
        "sort": "file_size:asc",
    }
    req = urllib.request.Request(
        FILES_API, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        hits = json.load(resp)["data"]["hits"]
    if not hits:
        raise SystemExit("no open-access TCGA-BRCA slide found")
    return hits[0]["file_id"], hits[0]["file_name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="./tcga_brca_slides",
                    help="download directory")
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)

    print("[1/2] querying GDC for a small TCGA-BRCA diagnostic slide...")
    file_id, file_name = query_smallest_slide()
    out = os.path.join(args.dest, file_name)
    print(f"    -> {file_name} ({file_id})")

    print(f"[2/2] downloading to {out} ...")
    urllib.request.urlretrieve(f"{DATA_API}/{file_id}", out)
    print(f"done. slide: {out}")


if __name__ == "__main__":
    main()
