#!/usr/bin/env python
"""The analyses added during the ML4H 2026 response period.

    python scripts/rebuttal_analyses.py \
        --long results/audit_long.csv \
        --tissue-map resources/slide_tissue.csv \
        --out results/

Writes, one file each:

    discrimination.csv        model-pair reordering rate per operator
    per_tissue_delta.csv      family contrast refit inside ccRCC and PRAD
    out_of_tissue.csv         primary / held-out / pooled cohorts
    leave_one_tissue_out.csv  one refit per omitted tissue
    regime_split.csv          in-domain supervised vs pretrained transfer
    error_conditioned_D.csv   accuracy plus error magnitude and smoothness

The block permutation null is not here. It needs the prediction cache rather
than the long table, so it runs inside scripts/run_audit.py with
--null block --n-blocks 25.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audit.analyses import (
    discrimination_table,
    leave_one_tissue_out,
    out_of_tissue,
    per_tissue_delta,
    regime_split,
)
from audit.errors import error_conditioned_D
from audit.io import describe, load_long

FAMILIES = {"LR": ("lr_magnitude", "lr_hotspot"), "MCP": ("mcp", "mcp_hotspot")}
PRIMARY = ("ccRCC", "PRAD")
REGIMES = {
    "in-domain supervised": ["Ridge", "PLS", "kNN"],
    "pretrained transfer": ["STPath", "DeepSpot-M"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", required=True)
    ap.add_argument("--tissue-map", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", default="", help="comma-separated analysis names")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    long = load_long(args.long, tissue_map=args.tissue_map)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    print("long table:", describe(long), flush=True)

    def write(name, obj):
        if name in skip or obj is None:
            return
        df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
        if df.empty:
            print(f"  {name}: empty, not written")
            return
        path = os.path.join(args.out, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  wrote {path}  ({len(df)} rows)", flush=True)

    if "discrimination" not in skip:
        write("discrimination", discrimination_table(long, n_boot=2000, seed=args.seed))

    write("per_tissue_delta", per_tissue_delta(long, FAMILIES, min_slides=10))

    if "out_of_tissue" not in skip:
        oot, comp = out_of_tissue(long, FAMILIES, PRIMARY,
                                  n_boot=args.n_boot, seed=args.seed)
        write("out_of_tissue", oot)
        print("  cohort composition:\n", comp.to_string(), flush=True)

    write("leave_one_tissue_out", leave_one_tissue_out(long, FAMILIES))
    write("regime_split", regime_split(long, FAMILIES, REGIMES))

    if {"err_rmse", "err_moran"}.issubset(long.columns):
        rows = []
        for fam, (mag, loc) in FAMILIES.items():
            for r in error_conditioned_D(long, mag, loc):
                rows.append({"family": fam, **r})
        write("error_conditioned_D", rows)
    else:
        print("  error_conditioned_D: long table has no err_* columns, skipped")
        print("  (rerun scripts/run_audit.py with --error-features)")


if __name__ == "__main__":
    main()
