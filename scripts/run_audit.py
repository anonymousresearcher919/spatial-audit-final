#!/usr/bin/env python
"""Score every operator on every slide-model pair and write the long table.

Expects predictions cached once per (slide, predictor) over the frozen gene
universe, as .npz with keys: Xhat, Xmeas, coords, genes.

    python scripts/run_audit.py --cache /path/to/cache --out results/audit_long.csv

--rich-baseline adds eight summary statistics of the per-gene accuracy
distribution, --error-features adds the magnitude and spatial smoothness of the
reconstruction error. Both widen the conditioning baseline that distinctness is
measured against, and both are needed to regenerate the response-period tables.

--null block replaces the independent spot shuffle with a permutation of
contiguous tiles, which keeps spatial autocorrelation and removes only the
alignment with the measured field. Check the tiling with
audit.spatial_nulls.block_report before choosing --n-blocks: a tiling with k
occupied tiles admits k! arrangements, and B beyond that spends draws on
repeats while producing an ordinary-looking p-value.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audit.baselines import matched_genes, per_gene_pcc, rich_baseline
from audit.core import REGISTRY, knn_graph, run_null
from audit.errors import error_features
from audit.resources import select_hvg
from audit.spatial_nulls import block_perturb, block_report

PREDICTORS = ["STPath", "DeepSpot-M", "OmiCLIP", "Ridge", "PLS", "kNN"]


def load_cell(cache, slide, predictor):
    path = os.path.join(cache, f"{slide}__{predictor}.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    return (
        z["Xhat"].astype(np.float32),
        z["Xmeas"].astype(np.float32),
        z["coords"].astype(np.float32),
        [str(g) for g in z["genes"]],
    )


def restrict(X, Xm, genes, gene_set):
    """Narrow to the panel an operator reads, so Phi and its baseline agree."""
    if gene_set == "hvg":
        idx, sub = select_hvg(Xm, genes, n_top=1611, min_detect=0.10)
    elif gene_set == "coexpr":
        idx, sub = select_hvg(Xm, genes, n_top=1200, min_detect=0.10)
    else:
        idx, sub = np.arange(len(genes)), genes
    return X[:, idx], Xm[:, idx], sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="results/audit_long.csv")
    ap.add_argument("--slides", default=None, help="comma-separated; default all in cache")
    ap.add_argument("--operators", default=None, help="comma-separated; default all")
    ap.add_argument("--rich-baseline", action="store_true")
    ap.add_argument("--error-features", action="store_true")
    ap.add_argument("--null", choices=("spot", "block"), default="spot")
    ap.add_argument("--n-blocks", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.null == "block":
        print(f"block permutation null, {args.n_blocks} requested tiles")

    if args.slides:
        slides = args.slides.split(",")
    else:
        slides = sorted({f.split("__")[0] for f in os.listdir(args.cache) if f.endswith(".npz")})

    ops = args.operators.split(",") if args.operators else list(REGISTRY)
    rows = []
    kept_own_null = set()

    for si, slide in enumerate(slides):
        for predictor in PREDICTORS:
            cell = load_cell(args.cache, slide, predictor)
            if cell is None:
                continue
            Xhat, Xmeas, C, genes = cell

            if args.null == "block" and si == 0 and predictor == PREDICTORS[0]:
                print("  tiling:", block_report(C, args.n_blocks), flush=True)

            W = knn_graph(C, k=8) if args.error_features else None

            for name in ops:
                op = REGISTRY[name]
                Xh, Xm, sub = restrict(Xhat, Xmeas, genes, op.gene_set)

                midx = matched_genes(name, sub, Xm)
                if midx.size == 0:
                    continue
                pcc, _ = per_gene_pcc(Xh, Xm, midx)

                if args.null == "block":
                    # an operator with its own perturb already constrains the
                    # null for its own reason; overriding it would replace a
                    # per-gene or re-clustering null with a row permutation
                    if op.perturb is None and op.null_on == "input":
                        op = replace(op, perturb=block_perturb(args.n_blocks))
                    elif name not in kept_own_null:
                        kept_own_null.add(name)
                        print(f"  {name}: keeps its own null, --null block ignored",
                              file=sys.stderr)

                try:
                    r = run_null(
                        op, Xh, Xm, C, sub,
                        seed=args.seed + 1000 * si + hash(name) % 997,
                    )
                except Exception as e:
                    print(f"  skip {slide}/{predictor}/{name}: {e}", file=sys.stderr)
                    continue

                row = {
                    "slide": slide,
                    "predictor": predictor,
                    "operator": name,
                    "family": op.family,
                    "score": r["score"],
                    "p": r["p"],
                    "z": r["z"],
                    "null_sd": r["null_sd"],
                    "B": r["B"],
                    "matched_pcc": pcc,
                    "n_matched_genes": int(midx.size),
                    "n_spots": int(Xm.shape[0]),
                }
                if args.rich_baseline:
                    row.update(rich_baseline(Xh, Xm, midx))
                if args.error_features:
                    row.update(error_features(Xh, Xm, C, idx=midx, W=W))
                if args.null == "block":
                    row["null_kind"] = "block"
                    row["n_blocks"] = int(args.n_blocks)
                rows.append(row)

        print(f"[{si + 1}/{len(slides)}] {slide}  rows={len(rows)}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
