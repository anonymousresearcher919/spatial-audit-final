"""Loading and validating the long audit table.

Every analysis downstream of the audit run reads this one table, so the schema
check lives here rather than being repeated. Two columns are easy to get
wrong: the predictor column was called `model` in earlier drafts, and `tissue`
is not produced by the audit run at all and has to be joined on.
"""

from __future__ import annotations

import os

import pandas as pd

REQUIRED = ("slide", "predictor", "operator", "family", "score", "p", "matched_pcc")
OPTIONAL = ("z", "null_sd", "B", "n_matched_genes", "n_spots")


def load_long(path, tissue_map=None, require_tissue=False):
    """Read the long audit table, normalise names, and attach tissue.

    `tissue_map` is a path to a CSV with columns slide,tissue, or a dict. The
    tissue and regime analyses need it; the operator table does not.
    """
    d = pd.read_csv(path)
    if "predictor" not in d.columns and "model" in d.columns:
        d = d.rename(columns={"model": "predictor"})

    missing = [c for c in REQUIRED if c not in d.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required columns {missing}. "
            f"Present: {sorted(d.columns)}"
        )

    if tissue_map is not None:
        if isinstance(tissue_map, (str, os.PathLike)):
            tm = pd.read_csv(tissue_map)
            tissue_map = dict(zip(tm["slide"], tm["tissue"]))
        d["tissue"] = d["slide"].map(tissue_map)
        unmapped = int(d["tissue"].isna().sum())
        if unmapped:
            raise ValueError(
                f"{unmapped} rows have no tissue. Unmapped slides: "
                f"{sorted(d.loc[d['tissue'].isna(), 'slide'].unique())[:5]}"
            )
    elif require_tissue:
        raise ValueError("this analysis needs tissue; pass tissue_map")

    return d


def describe(d):
    """Shape of the table, for a run log."""
    return {
        "rows": int(len(d)),
        "slides": int(d["slide"].nunique()),
        "predictors": int(d["predictor"].nunique()),
        "operators": int(d["operator"].nunique()),
        "tissues": int(d["tissue"].nunique()) if "tissue" in d else None,
    }
