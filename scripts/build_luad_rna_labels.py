#!/usr/bin/env python
"""Build TCGA-LUAD RNA panel + labels + clinical (Job A; the cheap critical path).
Runs in the container on a compute node (internet).

Sources (all confirmed reachable):
  - GDC API  : LUAD case list, AJCC stage, demographics, survival, TP53/KRAS muts
  - UCSC Toil: pan-cancer log2(FPKM+0.001) matrix -> subset LUAD primary tumors,
               top-200 most-variable genes -> genomic_features.csv
Writes to OUT_DIR: genomic_features.csv, labels.csv, clinical.csv, gene_map.json
(image_features.csv is built separately in Job B.)
"""
import csv, gzip, heapq, json, os, re, statistics, urllib.request

# exclude non-coding / clone-based / pseudogene symbols so the panel is
# protein-coding and interpretable (matches the BRCA panel's character)
NONCODING = re.compile(
    r"^(RNU|SNOR|SCARNA|MIR|RN7|RNA5|RNVU|VTRNA|LINC\d|RPL\d+P\d|RPS\d+P\d)"
    r"|^(AC|AL|AP|AF|AD|Z)\d{5,}"
    r"|^(RP\d+-|CT[CD]-|LL\d|XX)"
    r"|-AS\d*$|\.\d+$")


def is_coding_symbol(s):
    return bool(s) and not NONCODING.search(s)

OUT_DIR = "data/TCGA_LUAD_UNI"
WORK = "data"
TOIL = ("https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/"
        "tcga_RSEM_gene_fpkm.gz")
PROBEMAP = ("https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/"
            "probeMap%2Fgencode.v23.annotation.gene.probemap")
N_GENES = 200


def gdc_post(endpoint, payload, timeout=120):
    req = urllib.request.Request(
        f"https://api.gdc.cancer.gov/{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_clinical():
    d = gdc_post("cases", {
        "filters": {"op": "in", "content": {
            "field": "project.project_id", "value": ["TCGA-LUAD"]}},
        "fields": ("submitter_id,diagnoses.ajcc_pathologic_stage,"
                   "diagnoses.days_to_death,diagnoses.days_to_last_follow_up,"
                   "demographic.vital_status,demographic.gender,"
                   "demographic.age_at_index"),
        "size": 1000, "expand": "diagnoses,demographic"})
    out = {}
    for h in d["data"]["hits"]:
        pid = h["submitter_id"]                       # TCGA-05-4244
        dg = (h.get("diagnoses") or [{}])[0]
        dem = h.get("demographic") or {}
        stage = dg.get("ajcc_pathologic_stage")
        dtd = dg.get("days_to_death")
        dlf = dg.get("days_to_last_follow_up")
        vital = dem.get("vital_status")
        out[pid] = {
            "stage": stage,
            "age": dem.get("age_at_index"),
            "gender": 1 if dem.get("gender") == "male" else 0,
            "event": 1 if vital == "Dead" else 0,
            "time": dtd if dtd not in (None, "") else dlf,
        }
    return out


def fetch_mutated(gene):
    d = gdc_post("ssm_occurrences", {
        "filters": {"op": "and", "content": [
            {"op": "in", "content": {"field": "case.project.project_id",
                                     "value": ["TCGA-LUAD"]}},
            {"op": "in", "content": {
                "field": "ssm.consequence.transcript.gene.symbol",
                "value": [gene]}}]},
        "fields": "case.submitter_id", "size": 5000})
    return {h["case"]["submitter_id"] for h in d["data"]["hits"]
            if h.get("case")}


def stage_binary(stage):
    if not stage or "IS" in str(stage):
        return None
    s = stage.replace("Stage ", "").strip()
    if s in ("I", "IA", "IB"):
        return 0          # early
    if s.startswith(("II", "III", "IV")):
        return 1          # advanced
    return None


def download(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 10_000:
        print("cached", dst, flush=True)
        return
    print("downloading", url, "->", dst, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    print("done", os.path.getsize(dst), "bytes", flush=True)


def load_probemap(path):
    m = {}
    try:
        with open(path) as f:
            r = csv.reader(f, delimiter="\t")
            next(r, None)
            for row in r:
                if len(row) >= 2:
                    m[row[0]] = row[1]            # ensembl(.ver) -> symbol
    except Exception as e:  # noqa
        print("probemap parse failed:", e, flush=True)
    return m


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    clin = fetch_clinical()
    tp53 = fetch_mutated("TP53")
    kras = fetch_mutated("KRAS")
    print(f"clinical={len(clin)} TP53mut={len(tp53)} KRASmut={len(kras)}",
          flush=True)

    # --- RNA: download toil matrix + probemap, subset LUAD primary tumors ---
    toil_gz = os.path.join(WORK, "_toil_fpkm.gz")
    pmap_f = os.path.join(WORK, "_gencode_probemap.tsv")
    download(TOIL, toil_gz)
    download(PROBEMAP, pmap_f)
    sym = load_probemap(pmap_f)

    luad_pat = set(clin)                       # TCGA-05-4244
    with gzip.open(toil_gz, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        samples = header[1:]
        # keep primary-tumor (-01) columns whose patient is a LUAD case
        keep_idx, keep_pat = [], []
        for j, s in enumerate(samples):
            parts = s.split("-")
            if len(parts) >= 4:
                pid = "-".join(parts[:3])
                samp = parts[3][:2]
                if pid in luad_pat and samp == "01":
                    keep_idx.append(j)
                    keep_pat.append(pid)
        # dedup patients (first occurrence)
        seen, cols, pats = set(), [], []
        for j, pid in zip(keep_idx, keep_pat):
            if pid not in seen:
                seen.add(pid)
                cols.append(j)
                pats.append(pid)
        print(f"LUAD expression samples: {len(pats)}", flush=True)

        MEAN_MIN = 1.0                          # log2(fpkm+.001) ~ fpkm>=1
        heap = []                               # (variance, gene_id, symbol, vals)
        n = kept = 0
        for line in f:
            row = line.rstrip("\n").split("\t")
            gid = row[0]
            n += 1
            if n % 10000 == 0:
                print(f"  scanned {n} genes (kept-eligible {kept})", flush=True)
            s = sym.get(gid) or sym.get(gid.split(".")[0])
            if not is_coding_symbol(s):
                continue
            try:
                vals = [float(row[1 + j]) for j in cols]
            except (ValueError, IndexError):
                continue
            if len(vals) < 3 or statistics.fmean(vals) < MEAN_MIN:
                continue
            kept += 1
            v = statistics.pvariance(vals)
            if len(heap) < N_GENES:
                heapq.heappush(heap, (v, gid, s, vals))
            elif v > heap[0][0]:
                heapq.heapreplace(heap, (v, gid, s, vals))
    print(f"scanned {n} genes; {kept} coding+expressed; selected top {len(heap)}",
          flush=True)

    selected = sorted(heap, key=lambda t: -t[0])
    gene_cols, gene_map = [], {}
    for k, (_, gid, s, _) in enumerate(selected):
        col = f"gene_{k}_{s}"
        gene_cols.append(col)
        gene_map[col] = {"ensembl": gid, "symbol": s}

    # write genomic_features.csv
    with open(os.path.join(OUT_DIR, "genomic_features.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id"] + gene_cols)
        for i, pid in enumerate(pats):
            row = [pid] + [f"{selected[k][3][i]:.5f}" for k in range(len(selected))]
            w.writerow(row)

    # labels.csv: primary label = TP53 mutation; plus stage + kras + survival
    nL = {"tp53": 0, "stage_na": 0}
    with open(os.path.join(OUT_DIR, "labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "label", "label_tp53", "label_kras",
                    "label_stage", "survival_time", "survival_event"])
        for pid in pats:
            tp = 1 if pid in tp53 else 0
            kr = 1 if pid in kras else 0
            st = stage_binary(clin.get(pid, {}).get("stage"))
            nL["tp53"] += tp
            if st is None:
                nL["stage_na"] += 1
            t = clin.get(pid, {}).get("time")
            e = clin.get(pid, {}).get("event", "")
            w.writerow([pid, tp, tp, kr, "" if st is None else st,
                        "" if t in (None, "") else t, e])

    # clinical.csv
    with open(os.path.join(OUT_DIR, "clinical.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "age", "gender", "stage_raw"])
        for pid in pats:
            c = clin.get(pid, {})
            w.writerow([pid, c.get("age", ""), c.get("gender", ""),
                        c.get("stage", "")])

    json.dump(gene_map, open(os.path.join(OUT_DIR, "gene_map.json"), "w"),
              indent=2)
    n_stage_pos = sum(1 for pid in pats
                      if stage_binary(clin.get(pid, {}).get("stage")) == 1)
    n_stage_known = sum(1 for pid in pats
                        if stage_binary(clin.get(pid, {}).get("stage")) is not None)
    print(f"\nWROTE {OUT_DIR}")
    print(f"  patients (RNA): {len(pats)}  genes: {len(gene_cols)}")
    print(f"  TP53+ : {nL['tp53']}/{len(pats)} "
          f"({100*nL['tp53']/len(pats):.1f}%)")
    print(f"  stage known: {n_stage_known}  advanced(II+): {n_stage_pos}")
    print(f"  example genes: {gene_cols[:6]}")
    print("JOB_A_DONE")


if __name__ == "__main__":
    main()
