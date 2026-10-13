"""Gene resources and the frozen gene universe.

Predictions are cached once over a frozen universe so that every operator is
scored on identical predictions. The universe is the union of the ligand-
receptor, pathway, marker and highly-variable gene sets, intersected with the
vocabularies all predictors emit.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

# AUDIT_RESOURCES is the name to use. The older project-specific variable is
# still read so an existing environment keeps working, but it names the
# project and should not appear in an anonymised release.
RESOURCE_DIR = (
    os.environ.get("AUDIT_RESOURCES")
    or os.environ.get("NICHECAST_RESOURCES")
    or os.path.join(os.path.dirname(__file__), "..", "resources")
)

LINEAGE_ORDER = [
    "T cells",
    "B lineage",
    "Monocytic lineage",
    "Neutrophils",
    "Endothelial cells",
    "Fibroblasts",
    "NK cells",
    "Myeloid dendritic cells",
]


@lru_cache(maxsize=1)
def lr_pairs():
    """LIANA consensus ligand-receptor pairs, restricted to one-to-one."""
    df = pd.read_csv(os.path.join(RESOURCE_DIR, "liana_consensus.csv"))
    df = df[~df["ligand"].str.contains("_") & ~df["receptor"].str.contains("_")]
    pairs = list(dict.fromkeys(zip(df["ligand"], df["receptor"])))
    return tuple(pairs)


@lru_cache(maxsize=1)
def mcp_markers():
    """MCP-counter lineage marker panels."""
    with open(os.path.join(RESOURCE_DIR, "mcp_counter.json")) as fh:
        d = json.load(fh)
    return {k: tuple(v) for k, v in d.items()}


@lru_cache(maxsize=1)
def progeny_weights():
    """PROGENy signed pathway weights, top 500 genes per pathway.

    Cached as a flat table so the audit has no runtime dependency on the
    decoupler package, whose install pulls a pandas major version that
    conflicts with the rest of the stack.
    """
    df = pd.read_csv(os.path.join(RESOURCE_DIR, "progeny_top500.csv"))
    out = {}
    for pw, sub in df.groupby("pathway"):
        out[pw] = dict(zip(sub["gene"], sub["weight"].astype(np.float32)))
    return out


def select_hvg(X, genes, n_top=1611, min_detect=0.10):
    """Detection-filtered, variance-ranked gene selection."""
    detect = (X > 0).mean(axis=0)
    keep = np.where(detect >= min_detect)[0]
    v = X[:, keep].var(axis=0)
    order = keep[np.argsort(-v)][:n_top]
    order = np.sort(order)
    return order, [genes[i] for i in order]


def freeze_universe(gene_lists, predictor_vocabs, path):
    """Union of operator gene sets intersected with every predictor vocabulary.

    Written once. Every cached prediction is indexed against this list, so
    rebuilding it invalidates the cache.
    """
    union = set()
    for gl in gene_lists:
        union |= set(gl)
    for vocab in predictor_vocabs:
        union &= set(vocab)
    universe = sorted(union)
    with open(path, "w") as fh:
        json.dump(universe, fh)
    return universe


def load_universe(path):
    with open(path) as fh:
        return json.load(fh)


def gene_index(genes, wanted):
    """Positions of `wanted` within `genes`.

    Keyed on the full tuple rather than a length/endpoint digest: panels that
    share a length and endpoints are common among marker sets, and a digest
    key silently returns another panel's indices.
    """
    pos = {g: i for i, g in enumerate(genes)}
    return np.array([pos[g] for g in wanted if g in pos], dtype=int)
