#!/usr/bin/env python
"""Turn the long audit table into the reported results.

    python scripts/analyze.py --long results/audit_long.csv --out results/

Writes operator_table.csv, paired_deltas.csv, sensitivity.csv.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audit.baselines import RICH_FEATURES
from audit.distinctness import bootstrap_D, fit_D, paired_delta_D, recoverability

MATCHED_FAMILIES = [
    ("lr_magnitude", "lr_hotspot"),
    ("mcp", "mcp_hotspot"),
    ("moran", "lisa"),
]

RECOVERABLE_CUTOFFS = (0.3, 0.5, 0.7)


def operator_table(long, n_boot, seed):
    rows = []
    for name, d in long.groupby("operator"):
        rec = recoverability(d, n_boot=n_boot, seed=seed)
        dis = bootstrap_D(d, n_boot=n_boot, seed=seed)
        if rec is None or dis is None:
            continue
        rows.append(
            {
                "operator": name,
                "family": d["family"].iloc[0],
                "n_slides": rec["n_slides"],
                "n_pairs": rec["n_pairs"],
                "n_untestable": rec["n_untestable"],
                "B": int(d["B"].median()),
                "recoverability": rec["recoverability"],
                "rec_lo": rec["ci_lo"],
                "rec_hi": rec["ci_hi"],
                "rec_z3": float((d["z"] > 3).mean()),
                "D": dis["D"],
                "D_lo": dis["ci_lo"],
                "D_hi": dis["ci_hi"],
                "r2_cv": dis["r2_cv"],
                "null_sd_median": float(d["null_sd"].median()),
            }
        )
    t = pd.DataFrame(rows)
    t["group"] = np.where(
        t["recoverability"] < 0.5,
        "C: fails recoverability",
        np.where(t["D"] >= 0.30, "A: recoverable and distinct", "B: redundant anchor"),
    )
    return t.sort_values(["group", "D"], ascending=[True, False])


def tissue_spread(long, tissue_map):
    """|D_ccRCC - D_PRAD| per operator."""
    long = long.copy()
    long["tissue"] = long["slide"].map(tissue_map)
    out = []
    for name, d in long.groupby("operator"):
        vals = {}
        for t in ("ccRCC", "PRAD"):
            sub = d[d["tissue"] == t]
            r = fit_D(sub) if len(sub) else None
            vals[t] = r["D"] if r else np.nan
        out.append(
            {
                "operator": name,
                "D_ccRCC": vals["ccRCC"],
                "D_PRAD": vals["PRAD"],
                "spread": abs(vals["ccRCC"] - vals["PRAD"]),
            }
        )
    return pd.DataFrame(out)


def paired_table(long, table, n_boot, seed):
    rec = dict(zip(table["operator"], table["recoverability"]))
    rows = []
    for mag, loc in MATCHED_FAMILIES:
        dm = long[long["operator"] == mag]
        dl = long[long["operator"] == loc]
        if dm.empty or dl.empty:
            continue
        r = paired_delta_D(dm, dl, n_boot=n_boot, seed=seed)
        if r is None:
            continue
        interpretable = min(rec.get(mag, 0), rec.get(loc, 0)) >= 0.5
        rows.append(
            {
                "family": f"{mag} -> {loc}",
                "magnitude": mag,
                "localization": loc,
                **r,
                "interpretable": interpretable,
            }
        )
    return pd.DataFrame(rows)


def _delta(long, mag="lr_magnitude", loc="lr_hotspot", **kw):
    """Point-estimate paired contrast. No bootstrap: these rows report the
    estimate only, and a 500-draw interval per variant is 30x the cost."""
    a = fit_D(long[long["operator"] == mag], **kw)
    b = fit_D(long[long["operator"] == loc], **kw)
    return None if (a is None or b is None) else b["D"] - a["D"]


def sensitivity(long, table, n_boot, seed):
    """Robustness of the headline contrast to analysis choices."""
    rows = []

    for c in RECOVERABLE_CUTOFFS:
        n = int(((table["recoverability"] >= c) & (table["D"] >= 0.30)).sum())
        rows.append(
            {"check": f"recoverability cutoff {c}", "value": n,
             "note": "distinct-and-recoverable operators"}
        )

    for cond in ("spline", "linear", "rf", "gbm", "knn"):
        v = _delta(long, conditioner=cond)
        if v is not None:
            rows.append({"check": f"conditioner={cond}", "value": v,
                         "note": "LR family paired delta D"})

    if set(RICH_FEATURES).issubset(long.columns):
        v = _delta(long, features=RICH_FEATURES)
        if v is not None:
            rows.append({"check": "rich 8-feature baseline", "value": v,
                         "note": "wider conditioner widens the gap"})

    for model in sorted(long["predictor"].unique()):
        sub = long[long["predictor"] != model]
        v = _delta(sub)
        if v is not None:
            rows.append({"check": f"leave-out {model}", "value": v,
                         "note": "LR family paired delta D"})

    parts = [v for k in (3, 4, 5, 6, 8, 10) if (v := _delta(long, n_splits=k)) is not None]
    if parts:
        rows.append(
            {"check": "cv folds 3-10", "value": float(np.median(parts)),
             "note": f"min {min(parts):.3f}, max {max(parts):.3f}"}
        )

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tissue-map", default=None, help="CSV with slide,tissue")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    long = pd.read_csv(args.long)

    table = operator_table(long, args.n_boot, args.seed)
    if args.tissue_map:
        tm = pd.read_csv(args.tissue_map)
        table = table.merge(
            tissue_spread(long, dict(zip(tm["slide"], tm["tissue"])))[["operator", "spread"]],
            on="operator",
            how="left",
        )
    table.to_csv(os.path.join(args.out, "operator_table.csv"), index=False)

    paired_table(long, table, args.n_boot, args.seed).to_csv(
        os.path.join(args.out, "paired_deltas.csv"), index=False
    )
    sensitivity(long, table, args.n_boot, args.seed).to_csv(
        os.path.join(args.out, "sensitivity.csv"), index=False
    )

    with pd.option_context("display.width", 160, "display.max_columns", 40):
        print(table[["operator", "group", "recoverability", "D", "D_lo", "D_hi"]])


if __name__ == "__main__":
    main()
