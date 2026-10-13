"""Analyses that run on the long audit table rather than on slide data.

None of these touch predictions or permutation nulls. Each consumes the long
per-cell table written by scripts/run_audit.py, one row per
slide x predictor x operator:

    slide predictor operator family score p z null_sd B matched_pcc

`tissue` is not written by the audit run and is attached by `io.load_long`
from resources/slide_tissue.csv. The tissue and regime functions here require
it. Use `io.load_long` rather than pandas.read_csv so the column names and the
tissue join are checked in one place.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .distinctness import SlideIndex, fit_D


# --------------------------------------------------------------------------
# Does the operator reorder the models?
# --------------------------------------------------------------------------

def rank_reversals(df, operator, ref="matched_pcc", n_boot=2000, seed=0):
    """Fraction of within-slide model pairs the operator orders against accuracy.

    This is the operational question a benchmark designer actually faces: if the
    leaderboard is rebuilt under this operator instead of per-gene accuracy, does
    the ordering change? A redundant operator reverses almost nothing. A noisy
    one reverses a great deal, which is why the rate is only interpretable for
    operators that pass the recoverability gate.
    """
    d = df[df["operator"] == operator]
    n_rev = n_tot = 0
    per_slide = []
    for _, g in d.groupby("slide"):
        g = g.set_index("predictor")
        rv = tot = 0
        for a, b in combinations(list(g.index), 2):
            dr = np.sign(g.loc[a, ref] - g.loc[b, ref])
            do = np.sign(g.loc[a, "score"] - g.loc[b, "score"])
            if dr != 0 and do != 0:
                tot += 1
                rv += int(dr != do)
        if tot:
            per_slide.append(rv / tot)
        n_rev += rv
        n_tot += tot
    if not per_slide:
        return None

    per_slide = np.asarray(per_slide)
    rng = np.random.default_rng(seed)
    draws = np.array([
        rng.choice(per_slide, per_slide.size, replace=True).mean()
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "operator": operator,
        "reversal": n_rev / max(n_tot, 1),
        "rev_lo": float(lo),
        "rev_hi": float(hi),
        "n_slides": int(per_slide.size),
        "n_comparisons": int(n_tot),
    }


def discrimination_table(df, n_boot=2000, seed=0):
    rows = []
    for op in sorted(df["operator"].unique()):
        r = rank_reversals(df, op, n_boot=n_boot, seed=seed)
        if r is None:
            continue
        d = fit_D(df[df["operator"] == op])
        r["D"] = d["D"] if d else np.nan
        r["recoverability"] = float((df.loc[df["operator"] == op, "p"] <= 0.05).mean())
        rows.append(r)
    t = pd.DataFrame(rows)
    # the floor is the rate expected when the operator carries no ordering
    # information beyond accuracy, taken from the most redundant anchor
    return t.sort_values("D", ascending=False)


# --------------------------------------------------------------------------
# Pair-level coupling
# --------------------------------------------------------------------------

def pair_level_coupling(PL, n_boot=500, seed=0):
    """Coupling of magnitude and localization to constituent-gene accuracy.

    PL carries one row per ligand-receptor pair per slide per model, with
    columns pair, slide, predictor, magnitude, hotspot, pcc_mean. Slide-level D
    aggregates several hundred pairs into one number, so a skeptic can call the
    family contrast an aggregation artifact. This repeats it one pair at a time.
    """
    def _rho(d):
        a = spearmanr(d["magnitude"], d["pcc_mean"]).statistic
        b = spearmanr(d["hotspot"], d["pcc_mean"]).statistic
        return float(a), float(b)

    rho_mag, rho_hot = _rho(PL)

    per_model = []
    for m, g in PL.groupby("predictor"):
        a, b = _rho(g)
        per_model.append(
            {"predictor": m, "rho_magnitude": a, "rho_hotspot": b, "gap": a - b}
        )

    ix = SlideIndex(PL)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        picks = rng.choice(ix.slides, size=len(ix.slides), replace=True)
        a, b = _rho(ix.take(picks, relabel=False))
        draws[i] = a - b
    lo, hi = np.percentile(draws, [2.5, 97.5])

    return {
        "pooled": {
            "rho_magnitude": rho_mag,
            "rho_hotspot": rho_hot,
            "gap": rho_mag - rho_hot,
            "gap_lo": float(lo),
            "gap_hi": float(hi),
            "p_gt0": float((draws > 0).mean()),
            "n_obs": int(len(PL)),
            "n_pairs": int(PL["pair"].nunique()),
        },
        "per_model": pd.DataFrame(per_model).sort_values("gap"),
        "boot": draws,
    }


# --------------------------------------------------------------------------
# How many independent evaluation axes are there?
# --------------------------------------------------------------------------

def effective_dimensionality(df, recoverable):
    """PCA on the accuracy-residualised scores of the recoverable operators.

    Residualising first is the point: the raw scores share whatever variance
    gene accuracy induces, so a PCA on them would report one dominant axis that
    is just reconstruction quality.
    """
    resid = {}
    for op in sorted(recoverable):
        r = fit_D(df[df["operator"] == op])
        if r is None:
            continue
        f = r["frame"]
        resid[op] = f.set_index(["slide", "predictor"])["residual"]
    if len(resid) < 2:
        return None

    R = pd.DataFrame(resid).dropna()
    Z = (R - R.mean()) / R.std(ddof=0).replace(0, 1.0)
    from sklearn.decomposition import PCA

    p = PCA().fit(Z.values)
    ev = p.explained_variance_
    cum = np.cumsum(ev) / ev.sum()
    return {
        "n_operators": R.shape[1],
        "n_obs": R.shape[0],
        "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
        "k_80": int(np.searchsorted(cum, 0.80) + 1),
        "k_90": int(np.searchsorted(cum, 0.90) + 1),
        "k_95": int(np.searchsorted(cum, 0.95) + 1),
        "explained_variance_ratio": (ev / ev.sum()).tolist(),
        "residuals": R,
    }


# --------------------------------------------------------------------------
# Out-of-tissue replication
# --------------------------------------------------------------------------

def _paired_on(dm, dl, n_splits, n_boot, seed):
    a, b = fit_D(dm, n_splits=n_splits), fit_D(dl, n_splits=n_splits)
    if a is None or b is None:
        return None
    shared = np.array(
        sorted(set(dm["slide"].astype(str)) & set(dl["slide"].astype(str)))
    )
    im = SlideIndex(dm[dm["slide"].astype(str).isin(shared)])
    il = SlideIndex(dl[dl["slide"].astype(str).isin(shared)])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        picks = rng.choice(shared, size=len(shared), replace=True)
        ra = fit_D(im.take(picks), n_splits=n_splits)
        rb = fit_D(il.take(picks), n_splits=n_splits)
        if ra and rb:
            draws.append(rb["D"] - ra["D"])
    draws = np.asarray(draws)
    lo, hi = (np.percentile(draws, [2.5, 97.5]) if draws.size else (np.nan, np.nan))
    return {
        "D_mag": a["D"],
        "D_loc": b["D"],
        "delta_D": b["D"] - a["D"],
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_gt0": float((draws > 0).mean()) if draws.size else np.nan,
        "n_slides": int(len(shared)),
        "n_boot_ok": int(draws.size),
    }


def out_of_tissue(df, families, primary, n_boot=500, seed=0, n_splits=5):
    """Paired family contrast on the primary cohort, the held-out tissues, and all.

    A per-tissue D is not attempted for the minority tissues: four slides is 24
    slide-model observations, below the estimator's floor, and 24 points over
    four cross-validation groups would not support a distinctness estimate.
    Pooling them is what makes the comparison possible.
    """
    comp = df.groupby("tissue")["slide"].nunique().sort_values(ascending=False)
    held = [t for t in comp.index if t not in primary]

    rows = []
    cohorts = [
        ("primary", list(primary)),
        ("held-out", held),
        ("all", list(comp.index)),
    ]
    for label, keep in cohorts:
        sub = df[df["tissue"].isin(keep)]
        for fam, (mag, loc) in families.items():
            r = _paired_on(
                sub[sub["operator"] == mag], sub[sub["operator"] == loc],
                n_splits, n_boot, seed,
            )
            if r:
                rows.append({"cohort": label, "family": fam,
                             "tissues": "+".join(keep), **r})
    return pd.DataFrame(rows), comp


def leave_one_tissue_out(df, families):
    comp = df.groupby("tissue")["slide"].nunique()
    rows = []
    for t in comp.index:
        sub = df[df["tissue"] != t]
        for fam, (mag, loc) in families.items():
            a = fit_D(sub[sub["operator"] == mag])
            b = fit_D(sub[sub["operator"] == loc])
            if a is None or b is None:
                continue
            rows.append({
                "omitted": t,
                "n_slides_dropped": int(comp[t]),
                "family": fam,
                "D_mag": a["D"],
                "D_loc": b["D"],
                "delta_D": b["D"] - a["D"],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Per-tissue and per-regime refits
# --------------------------------------------------------------------------

def per_tissue_delta(df, families, min_slides=10, n_boot=0, seed=0):
    """Paired family contrast refit inside each sufficiently large tissue.

    The floor matters. A tissue with four slides gives 24 slide-model
    observations across four cross-validation groups, below what the estimator
    will fit, so the minority tissues are reported pooled by `out_of_tissue`
    rather than one at a time here.
    """
    comp = df.groupby("tissue")["slide"].nunique()
    rows = []
    for t in comp[comp >= min_slides].index:
        sub = df[df["tissue"] == t]
        for fam, (mag, loc) in families.items():
            if n_boot:
                r = _paired_on(sub[sub["operator"] == mag],
                               sub[sub["operator"] == loc], 5, n_boot, seed)
                if r is None:
                    continue
                rows.append({"tissue": t, "n_slides": int(comp[t]),
                             "family": fam, **r})
                continue
            a = fit_D(sub[sub["operator"] == mag])
            b = fit_D(sub[sub["operator"] == loc])
            if a is None or b is None:
                continue
            rows.append({"tissue": t, "n_slides": int(comp[t]), "family": fam,
                         "D_mag": a["D"], "D_loc": b["D"],
                         "delta_D": b["D"] - a["D"]})
    return pd.DataFrame(rows)


def regime_split(df, families, regimes):
    """Paired family contrast refit within each information regime.

    `regimes` maps a label to the predictors in it, for instance
    {"in-domain supervised": ["Ridge", "PLS", "kNN"],
     "pretrained transfer": ["STPath", "DeepSpot-M"]}.

    The audit never compares predictors to one another: every operator score is
    conditioned on that predictor's own realised accuracy on the genes feeding
    the operator. Splitting by regime therefore narrows the range of accuracy
    the conditioner observes rather than removing a confound, and a regime with
    two models bounds what can be read from its cell.
    """
    rows = []
    for label, models in regimes.items():
        sub = df[df["predictor"].isin(models)]
        for fam, (mag, loc) in families.items():
            a = fit_D(sub[sub["operator"] == mag])
            b = fit_D(sub[sub["operator"] == loc])
            if a is None or b is None:
                continue
            rows.append({"regime": label, "n_models": int(sub["predictor"].nunique()),
                         "family": fam, "D_mag": a["D"], "D_loc": b["D"],
                         "delta_D": b["D"] - a["D"]})
    return pd.DataFrame(rows)
