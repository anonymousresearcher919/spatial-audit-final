"""Distinctness D, slide-level cluster bootstrap, and paired family contrasts.

D = 1 - clip(R2_CV, 0, 1) where R2_CV is out-of-fold from a spline+ridge fit
of the operator score on its matched per-gene accuracy, grouped by slide.

One implementation is used everywhere: the table, the bootstrap, the paired
contrasts, and the sensitivity analyses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

ALPHAS = np.logspace(-3, 3, 25)
N_SPLITS = 5
N_BOOT = 500


def _conditioner(kind):
    if kind == "spline":
        return make_pipeline(
            SplineTransformer(n_knots=6, degree=3, include_bias=False),
            RidgeCV(alphas=ALPHAS),
        )
    if kind == "linear":
        return RidgeCV(alphas=ALPHAS)
    if kind == "rf":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0)
    if kind == "gbm":
        return GradientBoostingRegressor(random_state=0)
    if kind == "knn":
        return KNeighborsRegressor(n_neighbors=15)
    raise ValueError(f"unknown conditioner {kind}")


def fit_D(d, n_splits=N_SPLITS, conditioner="spline", features=("matched_pcc",)):
    """Out-of-fold distinctness for one operator.

    `d` needs columns: score, slide, and every name in `features`.
    Both the spline basis and the ridge penalty are fit inside each training
    fold; fitting the basis on the full sample leaks held-out slides into the
    representation of accuracy.
    """
    cols = ["score", "slide", *features]
    d = d.dropna(subset=cols).copy()
    if len(d) < 30 or d["slide"].nunique() < n_splits:
        return None

    X = d[list(features)].to_numpy(float)
    y = d["score"].to_numpy(float)
    g = d["slide"].astype(str).to_numpy()

    pred = np.full(len(d), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, g):
        mdl = _conditioner(conditioner)
        mdl.fit(X[tr], y[tr])
        pred[te] = mdl.predict(X[te])

    r2 = float(r2_score(y, pred))
    frame = d.copy()
    frame["pcc_pred"] = pred
    frame["residual"] = y - pred
    return {
        "D": 1.0 - float(np.clip(r2, 0.0, 1.0)),
        "D_untruncated": 1.0 - r2,
        "r2_cv": r2,
        "n": len(d),
        "n_slides": int(d["slide"].nunique()),
        "frame": frame,
    }


class SlideIndex:
    """Positional row indices per slide, computed once.

    Resampling by boolean mask rescans the whole frame once per slide per
    draw; at 500 draws over 61 slides and 15 operators that dominates the
    runtime of the entire analysis.
    """

    __slots__ = ("frame", "slides", "pos")

    def __init__(self, d):
        self.frame = d.reset_index(drop=True)
        key = self.frame["slide"].astype(str).to_numpy()
        order = np.argsort(key, kind="stable")
        sk = key[order]
        bounds = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1], True])
        self.slides = sk[bounds[:-1]]
        self.pos = {
            s: order[bounds[i] : bounds[i + 1]] for i, s in enumerate(self.slides)
        }

    def take(self, picks, relabel=True):
        idx = np.concatenate([self.pos[s] for s in picks])
        out = self.frame.take(idx).reset_index(drop=True)
        if relabel:
            labels = np.concatenate(
                [np.full(self.pos[s].size, f"{s}__b{i}") for i, s in enumerate(picks)]
            )
            out["slide"] = labels
        return out


def boot_slides(d, rng, index=None):
    """Cluster bootstrap over slides, with relabelling.

    A slide drawn twice must contribute two distinct CV groups. Without the
    relabel both copies land in the same fold, the effective number of
    independent groups shrinks, and the intervals come out too narrow.
    """
    ix = index or SlideIndex(d)
    picks = rng.choice(ix.slides, size=len(ix.slides), replace=True)
    return ix.take(picks)


def bootstrap_D(d, n_boot=N_BOOT, seed=0, **kw):
    """Percentile interval for D over slide resamples."""
    rng = np.random.default_rng(seed)
    base = fit_D(d, **kw)
    if base is None:
        return None
    ix = SlideIndex(d)
    draws = []
    for _ in range(n_boot):
        r = fit_D(boot_slides(d, rng, index=ix), **kw)
        if r is not None:
            draws.append(r["D"])
    draws = np.asarray(draws)
    lo, hi = (np.percentile(draws, [2.5, 97.5]) if draws.size else (np.nan, np.nan))
    return {
        "D": base["D"],
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_boot_ok": int(draws.size),
        "r2_cv": base["r2_cv"],
        "n": base["n"],
        "n_slides": base["n_slides"],
    }


def paired_delta_D(d_mag, d_loc, n_boot=N_BOOT, seed=0, **kw):
    """Paired slide bootstrap of D(localization) - D(magnitude).

    Both members are recomputed on the identical resample, which removes the
    shared slide-difficulty variance. Only interpretable when both members
    pass the recoverability gate.
    """
    rng = np.random.default_rng(seed)
    a, b = fit_D(d_mag, **kw), fit_D(d_loc, **kw)
    if a is None or b is None:
        return None
    obs = b["D"] - a["D"]

    im, il = SlideIndex(d_mag), SlideIndex(d_loc)
    shared = np.asarray(sorted(set(im.slides) & set(il.slides)))

    draws = []
    for _ in range(n_boot):
        picks = rng.choice(shared, size=len(shared), replace=True)
        ra = fit_D(im.take(picks), **kw)
        rb = fit_D(il.take(picks), **kw)
        if ra is not None and rb is not None:
            draws.append(rb["D"] - ra["D"])

    draws = np.asarray(draws)
    if draws.size == 0:
        return None
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "delta_D": float(obs),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_gt0": float((draws > 0).mean()),
        "n_slides": len(shared),
        "n_boot_ok": int(draws.size),
    }


def recoverability(df, alpha=0.05, n_boot=N_BOOT, seed=0):
    """Fraction of slide-model pairs with permutation p <= alpha, with CI.

    A cell whose permutation test could not be run leaves p undefined. Those
    cells are counted as non-recovery rather than dropped: the operator was
    asked to recover structure on that slide and did not, and excluding them
    would raise the reported rate by removing exactly the cells where the
    operator is least applicable. In this cohort it affects one operator,
    `progeny`, whose pathway coverage gate leaves 36 of 366 cells untestable;
    counting them as misses gives .836 against .927 for dropping them.

    `n_slides` counts slides with a defined p, so it can be smaller than the
    number of slides contributing to the denominator.
    """
    d = df.dropna(subset=["slide"]).copy()
    if d.empty:
        return None
    pv = pd.to_numeric(d["p"], errors="coerce").to_numpy()
    hit = ((pv <= alpha) & np.isfinite(pv)).astype(float)
    obs = float(hit.mean())

    rng = np.random.default_rng(seed)
    ix = SlideIndex(d)
    per_slide = {s: hit[ix.pos[s]] for s in ix.slides}
    draws = np.empty(n_boot)
    for i in range(n_boot):
        picks = rng.choice(ix.slides, size=len(ix.slides), replace=True)
        draws[i] = np.concatenate([per_slide[s] for s in picks]).mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "recoverability": obs,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_pairs": int(len(d)),
        "n_untestable": int((~np.isfinite(pv)).sum()),
        "n_slides": int(d.loc[np.isfinite(pv), "slide"].nunique()),
    }
