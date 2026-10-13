#!/usr/bin/env python
"""Generate a synthetic long audit table so the pipeline runs without data.

    python scripts/make_synthetic_long.py --out results/synthetic_long.csv

The HEST prediction cache is not redistributable, so a reader who clones this
repository cannot run scripts/run_audit.py. This script fabricates a table with
the same schema and the same qualitative structure, which makes the analysis
half of the pipeline executable end to end and lets the test suite assert on
behaviour rather than on constants.

What it is for: checking that the code runs, that the estimators recover a
known ground truth, and that a change to the library moves the numbers it
should move.

What it is not for: the numbers it produces are not the paper's numbers and
will not match them. The published values are in results/, and only
results/audit_long.csv -- which a reader must obtain separately, see
REPRODUCING.md -- regenerates them.

The generative model is deliberately simple and is the null hypothesis the
paper argues against for one operator family and for which it finds support in
the other. Each operator score is

    score = g(matched_pcc) * coupling + noise * (1 - coupling)

so `coupling` is the ground-truth R2 the conditioner should recover, and the
true distinctness of each operator is 1 - coupling by construction.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# name, family, true coupling to accuracy, recoverability, null sd
SPEC = [
    ("lr_magnitude",    "lr",      0.88, 0.87, 0.030),
    ("lr_hotspot",      "lr",      0.55, 0.80, 0.020),
    ("spatialdm",       "lr",      0.14, 0.60, 0.040),
    ("mcp",             "lineage", 0.91, 0.92, 0.025),
    ("mcp_hotspot",     "lineage", 0.59, 0.82, 0.030),
    ("neighborhood",    "lineage", 0.07, 0.42, 0.120),
    ("progeny",         "pathway", 0.90, 0.84, 0.028),
    ("coexpr",          "coexpr",  0.18, 0.94, 0.035),
    ("moran",           "autocorr", 0.05, 0.34, 0.150),
    ("moran_pea",       "autocorr", 0.07, 0.34, 0.140),
    ("lisa",            "autocorr", 0.34, 0.77, 0.045),
    ("svg_top200",      "autocorr", 0.00, 0.26, 0.200),
    ("domain_ari",      "domain",   0.04, 0.31, 0.160),
    ("domain_nmi",      "domain",   0.05, 0.37, 0.155),
    ("domain_boundary", "domain",   0.27, 0.60, 0.050),
]

PREDICTORS = ["STPath", "DeepSpot-M", "OmiCLIP", "Ridge", "PLS", "kNN"]

# slide counts matched to the real cohort so the grouped CV sees the same
# cluster-size imbalance
TISSUES = {"ccRCC": 24, "PRAD": 23, "COAD": 4, "READ": 4, "LYMPH_IDC": 4, "HCC": 2}

# per-predictor accuracy offsets, so the conditioner sees a spread of accuracy
PRED_OFFSET = {"STPath": 0.02, "DeepSpot-M": 0.015, "OmiCLIP": -0.01,
               "Ridge": 0.0, "PLS": -0.005, "kNN": -0.02}

# per-tissue accuracy level, lowest where the real predictors do worst
TISSUE_LEVEL = {"ccRCC": 0.136, "PRAD": 0.110, "COAD": 0.050,
                "READ": 0.060, "LYMPH_IDC": 0.070, "HCC": 0.010}


def build(seed=0, B=99):
    rng = np.random.default_rng(seed)

    slides, tissue_of = [], {}
    for t, n in TISSUES.items():
        for i in range(n):
            s = f"{t}_{i:02d}"
            slides.append(s)
            tissue_of[s] = t

    # one accuracy draw per slide-predictor cell, shared by every operator
    pcc = {}
    for s in slides:
        base = TISSUE_LEVEL[tissue_of[s]] + rng.normal(0, 0.025)
        for p in PREDICTORS:
            pcc[(s, p)] = float(np.clip(base + PRED_OFFSET[p] + rng.normal(0, 0.012),
                                        0.001, 0.95))

    acc = np.array(list(pcc.values()))
    lo, hi = acc.min(), acc.max()

    rows = []
    for name, family, coupling, rec, nsd in SPEC:
        for s in slides:
            for p in PREDICTORS:
                a = pcc[(s, p)]
                # monotone saturating signal in accuracy, scaled to [0, 1]
                signal = np.sqrt((a - lo) / (hi - lo + 1e-12))
                score = (coupling * signal
                         + (1 - coupling) * rng.uniform()
                         + rng.normal(0, 0.02))
                # permutation p is drawn so the expected hit rate is `rec`
                pv = (rng.uniform(0, 0.05) if rng.uniform() < rec
                      else rng.uniform(0.05, 1.0))
                pv = max(pv, 1.0 / (1.0 + B))
                rows.append({
                    "slide": s,
                    "predictor": p,
                    "operator": name,
                    "family": family,
                    "score": float(np.clip(score, 0, None)),
                    "p": float(pv),
                    "z": float(max(0.0, rng.normal(3.0 if pv <= 0.05 else 0.5, 1.0))),
                    "null_sd": float(nsd * rng.uniform(0.8, 1.2)),
                    "B": B,
                    "matched_pcc": a,
                    "err_rmse": float(0.9 - 1.2 * a + rng.normal(0, 0.05)),
                    "err_mae": float(0.65 - 0.9 * a + rng.normal(0, 0.04)),
                    "err_moran": float(0.21 - 0.15 * a + rng.normal(0, 0.03)),
                    "synthetic": True,
                })
    return pd.DataFrame(rows), tissue_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/synthetic_long.csv")
    ap.add_argument("--tissue-out", default="results/synthetic_slide_tissue.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    long, tissue_of = build(seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    long.to_csv(args.out, index=False)
    pd.DataFrame({"slide": list(tissue_of), "tissue": list(tissue_of.values())}).to_csv(
        args.tissue_out, index=False
    )

    truth = pd.DataFrame(
        [{"operator": n, "true_coupling": c, "true_D": round(1 - c, 3),
          "true_recoverability": r} for n, _, c, r, _ in SPEC]
    )
    print(f"wrote {args.out}  ({len(long)} rows, {long['slide'].nunique()} slides, "
          f"{long['operator'].nunique()} operators)")
    print(f"wrote {args.tissue_out}")
    print("\nground truth the estimators should recover:")
    print(truth.to_string(index=False))
    print("\nSYNTHETIC. These are not the paper's numbers.")


if __name__ == "__main__":
    main()
