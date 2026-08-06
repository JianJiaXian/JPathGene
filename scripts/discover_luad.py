#!/usr/bin/env python
"""Feasibility probe v2 for a TCGA-LUAD second cohort. Runs in the container on a
compute node (internet confirmed: GDC API reachable). Checks:
  1. Does W8Yi/tcga-wsi-uni2h-features cover LUAD? layout + count.
  2. Which public RNA matrix host is reachable (Xena candidates, GET).
  3. GDC clinical: AJCC stage distribution -> a balanced binary endpoint?
  4. GDC mutations: EGFR / KRAS / TP53 mutated-case counts -> balanced label?
Downloads nothing heavy (ranged GETs only).
"""
import json, os, re, collections, urllib.request

REPO = "W8Yi/tcga-wsi-uni2h-features"
OUT = "data/_luad_probe.json"
report = {}


def probe_hf():
    try:
        from huggingface_hub import HfApi
    except Exception:
        os.system("pip install --user -q huggingface_hub >/dev/null 2>&1")
        from huggingface_hub import HfApi
    try:
        files = HfApi().list_repo_files(REPO, repo_type="dataset")
        report["hf_total_files"] = len(files)
        coh = collections.Counter()
        for f in files:
            for c in re.findall(r"TCGA[_-]([A-Z]{2,5})", f):
                coh[c] += 1
        report["hf_cohort_tokens"] = dict(coh.most_common(40))
        report["hf_top_dirs"] = dict(collections.Counter(
            f.split("/")[0] for f in files if "/" in f).most_common(40))
        luad = [f for f in files if "LUAD" in f.upper()]
        report["hf_luad_count"] = len(luad)
        report["hf_luad_examples"] = luad[:8]
        report["hf_h5_examples"] = [f for f in files if f.endswith(".h5")][:5]
    except Exception as e:  # noqa
        report["hf_error"] = repr(e)[:300]


def get_head(name, url, nbytes=300, timeout=30):
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0",
                          "Range": f"bytes=0-{nbytes}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            report[name] = {"ok": True, "status": r.status,
                            "first": data[:80].decode("latin1", "replace")}
            return True
    except Exception as e:  # noqa
        report[name] = {"ok": False, "err": repr(e)[:160]}
        return False


def gdc_post(endpoint, payload, timeout=60):
    req = urllib.request.Request(
        f"https://api.gdc.cancer.gov/{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def probe_gdc_stage():
    try:
        d = gdc_post("cases", {
            "filters": {"op": "in", "content": {
                "field": "project.project_id", "value": ["TCGA-LUAD"]}},
            "fields": "diagnoses.ajcc_pathologic_stage,submitter_id",
            "size": 1000, "expand": "diagnoses"})
        hits = d["data"]["hits"]
        stages = collections.Counter()
        for h in hits:
            dg = (h.get("diagnoses") or [{}])[0]
            stages[dg.get("ajcc_pathologic_stage", "NA")] += 1
        report["gdc_stage_dist"] = dict(stages)
        report["gdc_n_cases_with_clinical"] = len(hits)
    except Exception as e:  # noqa
        report["gdc_stage_err"] = repr(e)[:200]


def probe_gdc_gene(gene):
    # count LUAD cases with a simple somatic mutation in `gene`
    try:
        d = gdc_post("ssm_occurrences", {
            "filters": {"op": "and", "content": [
                {"op": "in", "content": {"field": "case.project.project_id",
                                         "value": ["TCGA-LUAD"]}},
                {"op": "in", "content": {
                    "field": "ssm.consequence.transcript.gene.symbol",
                    "value": [gene]}}]},
            "fields": "case.submitter_id", "size": 2000})
        cases = {h["case"]["submitter_id"] for h in d["data"]["hits"]
                 if h.get("case")}
        report.setdefault("gdc_mut_cases", {})[gene] = len(cases)
    except Exception as e:  # noqa
        report.setdefault("gdc_mut_err", {})[gene] = repr(e)[:160]


def main():
    probe_hf()
    hosts = {
        "xena_gdchub": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/"
                       "TCGA-LUAD.htseq_fpkm.tsv.gz",
        "xena_gdc_net": "https://gdc.xenahubs.net/download/"
                        "TCGA-LUAD.htseq_fpkm.tsv.gz",
        "xena_pancan": "https://toil-xena-hub.s3.us-east-1.amazonaws.com/"
                       "download/tcga_RSEM_gene_fpkm.gz",
    }
    for n, u in hosts.items():
        get_head(n, u)
    get_head("xena_pheno_gdc_net",
             "https://gdc.xenahubs.net/download/TCGA-LUAD.GDC_phenotype.tsv.gz")
    probe_gdc_stage()
    for g in ["TP53", "KRAS", "EGFR", "STK11", "KEAP1"]:
        probe_gdc_gene(g)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
