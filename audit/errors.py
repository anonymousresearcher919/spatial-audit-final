"""Reconstruction-error conditioners.

The reported baseline conditions an operator score on the mean per-gene
correlation over the genes feeding it. Two other descriptions of
reconstruction quality are not functions of that distribution at all: how
large the error is, and how spatially structured it is. If localization's
distinctness were an artifact of an incomplete account of reconstruction
quality, adding either should absorb it.

Error smoothness is the sharper of the two. An operator that reads where a
signal's high values sit could in principle be scoring the geometry of the
model's mistakes rather than the geometry of the tissue, and no summary of the
per-gene accuracy distribution would reveal that.
"""

from __future__ import annotations

import numpy as np

from .core import knn_graph
from .distinctness import fit_D

ERROR_FEATURES = ("err_rmse", "err_mae", "err_moran")


def error_morans_i(E, W):
    """Moran's I of each column of the error matrix, on the given weights.

    Row-standardised weights make the denominator the column's own variance,
    so this is the lag-1 autocorrelation of the error field per gene. Positive
    means the model is wrong in patches rather than at random spots.
    """
    Ec = E - E.mean(axis=0, keepdims=True)
    num = np.einsum("ij,ij->j", Ec, np.asarray(W @ Ec))
    den = np.einsum("ij,ij->j", Ec, Ec)
    ok = den > 1e-12
    out = np.zeros_like(den)
    np.divide(num, den, out=out, where=ok)
    return out


def error_features(Xhat, Xmeas, C, idx=None, k=8, W=None):
    """Magnitude and spatial structure of the reconstruction error.

    `idx` restricts to the genes feeding one operator, so the error is
    described on the same panel the operator reads rather than transcriptome
    wide.
    """
    if idx is None:
        idx = np.arange(Xmeas.shape[1])
    E = (np.asarray(Xhat)[:, idx] - np.asarray(Xmeas)[:, idx]).astype(np.float32)
    if E.size == 0:
        return {f: np.nan for f in ERROR_FEATURES}
    if W is None:
        W = knn_graph(C, k=k)
    return {
        "err_rmse": float(np.sqrt((E ** 2).mean())),
        "err_mae": float(np.abs(E).mean()),
        "err_moran": float(np.nanmean(error_morans_i(E, W))),
    }


def error_conditioned_D(long, mag, loc, features=("err_rmse", "err_moran"),
                        base="matched_pcc", conditioner="linear", n_splits=5):
    """ΔD for one family under accuracy alone and under accuracy plus error terms.

    The reported D uses a spline conditioner on a single feature. Splines on
    several features multiply the basis, so the multi-feature rows use a linear
    conditioner on standardised inputs. That makes the added-feature rows
    comparable with each other and not directly with the spline row, which is
    why both are reported and labelled.
    """
    sets = [("accuracy only (reported)", (base,), "spline")]
    for f in features:
        sets.append((f"+ {f}", (base, f), conditioner))
    if len(features) > 1:
        sets.append(("+ all", (base, *features), conditioner))

    rows = []
    for label, feats, cond in sets:
        a = fit_D(long[long["operator"] == mag], features=feats,
                  conditioner=cond, n_splits=n_splits)
        b = fit_D(long[long["operator"] == loc], features=feats,
                  conditioner=cond, n_splits=n_splits)
        if a is None or b is None:
            continue
        rows.append({
            "baseline": label,
            "conditioner": cond,
            "features": "+".join(feats),
            "D_mag": a["D"],
            "D_loc": b["D"],
            "delta_D": b["D"] - a["D"],
        })
    return rows
