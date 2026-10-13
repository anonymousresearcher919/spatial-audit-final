"""Operator-matched gene accuracy baselines.

Operators run on different panels, so a single global per-gene accuracy would
confound panel composition with operator behavior: an operator reading 56
sparse immune markers is being compared against a different reconstruction
problem than one reading 1,611 highly variable genes. Each operator is scored
against the per-gene accuracy of exactly the genes that feed it, after the
same expression filter the operator applies.
"""

from __future__ import annotations

import numpy as np

from .resources import gene_index, lr_pairs, mcp_markers, progeny_weights
from .operators import LINEAGE_ORDER


def per_gene_pcc(Xhat, Xmeas, idx=None):
    """Mean Pearson correlation over the given gene columns."""
    if idx is None:
        idx = np.arange(Xmeas.shape[1])
    A, B = Xhat[:, idx], Xmeas[:, idx]
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    num = np.einsum("ij,ij->j", Ac, Bc)
    den = np.sqrt(np.einsum("ij,ij->j", Ac, Ac) * np.einsum("ij,ij->j", Bc, Bc))
    r = np.where(den > 1e-12, num / den, np.nan)
    return float(np.nanmean(r)), r


def matched_genes(op_name, genes, Xmeas):
    """The gene columns feeding `op_name` on this slide, after its own filter."""
    detect = (Xmeas > 0).mean(axis=0)
    gs = set(genes)

    if op_name in ("lr_magnitude", "lr_hotspot", "spatialdm"):
        keep = {g for g, d in zip(genes, detect) if d >= 0.10}
        want = sorted({g for p in lr_pairs() for g in p if g in gs and g in keep})
        return gene_index(genes, want)

    if op_name in ("mcp", "mcp_hotspot", "neighborhood"):
        keep = {g for g, d in zip(genes, detect) if d >= 0.02}
        markers = mcp_markers()
        absent = [pop for pop in LINEAGE_ORDER if pop not in markers]
        if absent:
            raise KeyError(
                f"mcp_counter.json is missing {absent}. The lineage operators "
                f"read all of {LINEAGE_ORDER}, keyed exactly as MCP-counter "
                f"names them. Present: {sorted(markers)}"
            )
        want = sorted(
            {g for pop in LINEAGE_ORDER for g in markers[pop] if g in keep}
        )
        return gene_index(genes, want)

    if op_name == "progeny":
        keep = {g for g, d in zip(genes, detect) if d >= 0.10}
        want = set()
        for pw, w in progeny_weights().items():
            present = [g for g in w if g in keep]
            if len(present) >= 20 and len(present) / len(w) >= 0.30:
                want |= set(present)
        return gene_index(genes, sorted(want))

    keep = np.where(detect >= 0.10)[0]
    return keep


def rich_baseline(Xhat, Xmeas, idx):
    """Eight summary statistics of the per-gene accuracy distribution.

    A single mean is the weakest defensible conditioner. If distinctness were
    an artifact of under-conditioning, widening the baseline would shrink it.
    """
    _, r = per_gene_pcc(Xhat, Xmeas, idx)
    r = r[np.isfinite(r)]
    if r.size < 5:
        return {k: np.nan for k in (
            "pcc_mean", "pcc_median", "pcc_sd", "pcc_q10",
            "pcc_q90", "pcc_iqr", "pcc_frac_pos", "pcc_skew",
        )}
    q10, q25, q75, q90 = np.percentile(r, [10, 25, 75, 90])
    m, s = r.mean(), r.std()
    return {
        "pcc_mean": float(m),
        "pcc_median": float(np.median(r)),
        "pcc_sd": float(s),
        "pcc_q10": float(q10),
        "pcc_q90": float(q90),
        "pcc_iqr": float(q75 - q25),
        "pcc_frac_pos": float((r > 0).mean()),
        "pcc_skew": float(((r - m) ** 3).mean() / (s ** 3)) if s > 1e-12 else 0.0,
    }


RICH_FEATURES = (
    "pcc_mean",
    "pcc_median",
    "pcc_sd",
    "pcc_q10",
    "pcc_q90",
    "pcc_iqr",
    "pcc_frac_pos",
    "pcc_skew",
)
