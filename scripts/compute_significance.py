#!/usr/bin/env python
"""Welch's t-test (per-seed) for JPathGene (full) vs Gene-only on AUC and ECE,
from outputs/<cohort>/tables/per_seed.csv. Pure stdlib. Writes a small JSON and
prints a paper-ready sentence. Addresses the reviewer's significance request.
"""
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.io import OUT_ROOT as _OUT  # noqa: E402

CSV = os.path.join(_OUT, "tables", "per_seed.csv")
OUT = os.path.join(_OUT, "analysis", "significance.json")
OURS = "JPathGene (full, ours)"
BASE = "Gene-only"


def _stats(xs):
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, v, n


def welch(a, b):
    ma, va, na = _stats(a)
    mb, vb, nb = _stats(b)
    se = math.sqrt(va / na + vb / nb) if (va or vb) else 1e-9
    t = (ma - mb) / se if se else 0.0
    # Welch-Satterthwaite df
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1) + 1e-12
    df = num / den if den else na + nb - 2
    # two-sided p via survival of t-dist (normal approx for small df is rough;
    # use a simple t-CDF via regularized incomplete beta)
    p = _t_sf(abs(t), df) * 2
    # 95% CI of the difference (normal approx)
    ci = (ma - mb - 1.96 * se, ma - mb + 1.96 * se)
    return dict(mean_ours=ma, mean_base=mb, diff=ma - mb, t=t, df=df, p=p,
                ci95=ci)


def _t_sf(t, df):
    # survival function of Student-t via incomplete beta
    x = df / (df + t * t)
    return 0.5 * _betai(df / 2.0, 0.5, x)


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    # continued fraction
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1e-30 if abs(d) < 1e-30 else d
    d = 1.0 / d
    h = d
    for i in range(1, 200):
        m = i // 2
        if i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        h *= d * c
        if abs(d * c - 1) < 1e-8:
            break
    return front * h


def main():
    if not os.path.exists(CSV):
        print(f"[signif] {CSV} missing"); return
    rows = list(csv.DictReader(open(CSV)))
    by = {}
    for r in rows:
        by.setdefault(r["method"], {"auc": [], "ece": []})
        by[r["method"]]["auc"].append(float(r["auc"]))
        by[r["method"]]["ece"].append(float(r["ece"]))
    if OURS not in by or BASE not in by:
        print(f"[signif] need both '{OURS}' and '{BASE}' in {CSV}; have "
              f"{list(by)}"); return
    res = {"auc": welch(by[OURS]["auc"], by[BASE]["auc"]),
           "ece": welch(by[OURS]["ece"], by[BASE]["ece"]),
           "n_seeds": len(by[OURS]["auc"])}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    a, e = res["auc"], res["ece"]
    print(f"AUC: ours {a['mean_ours']:.3f} vs gene-only {a['mean_base']:.3f}, "
          f"diff {a['diff']:+.3f} (95% CI [{a['ci95'][0]:+.3f},{a['ci95'][1]:+.3f}]), "
          f"p={a['p']:.3f}")
    print(f"ECE: ours {e['mean_ours']:.3f} vs gene-only {e['mean_base']:.3f}, "
          f"diff {e['diff']:+.3f}, p={e['p']:.4f}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
