"""Split-half reliability, and what it does and does not explain.

A distinctness gap between two operators in the same family has a competing
explanation: the localization member may simply be the noisier measurement.
Distinctness is one minus explained variance, so a score with more measurement
error has a higher D for a reason that has nothing to do with what it measures.

The test is to split the operator's constituent units -- ligand-receptor pairs,
marker panels -- into halves, recompute the slide-model score on each half
independently, and correlate the two. The Spearman-Brown step then projects
that half-length reliability back to the full-length instrument.
"""

from __future__ import annotations

import numpy as np

from .distinctness import fit_D


def spearman_brown(r_half):
    """Full-length reliability implied by a correlation between two halves."""
    r = float(r_half)
    return 2.0 * r / (1.0 + r) if r > -1.0 else np.nan


def split_half(scores_a, scores_b):
    """Reliability from one pair of half-panel score vectors."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return {"r_half": np.nan, "r_sb": np.nan, "n": int(m.sum())}
    r = float(np.corrcoef(a[m], b[m])[0, 1])
    return {"r_half": r, "r_sb": spearman_brown(r), "n": int(m.sum())}


def repeated_split_half(score_fn, units, n_rep=100, seed=0):
    """Reliability over `n_rep` random partitions of the constituent units.

    `score_fn(unit_subset) -> array` recomputes the slide-model score vector
    from a subset of units, with the slide-model rows in a fixed order across
    calls. A single arbitrary split is one draw from a distribution wide enough
    to matter, so the reported figure is the median over repartitions and the
    interval is the 2.5-97.5 range.
    """
    units = np.asarray(units)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_rep):
        order = rng.permutation(units.size)
        half = units.size // 2
        r = split_half(score_fn(units[order[:half]]), score_fn(units[order[half:]]))
        if np.isfinite(r["r_sb"]):
            draws.append(r["r_sb"])
    draws = np.asarray(draws)
    if draws.size == 0:
        return {"r_sb": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_rep": 0}
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "r_sb": float(np.median(draws)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_rep": int(draws.size),
        "draws": draws,
    }


def residualized_split_half(long_a, long_b, **kw):
    """Reliability of the residual D is computed from, not of the raw score.

    This is the version that answers the objection. Showing that a raw
    aggregate score reproduces across halves says nothing about the residual
    left after conditioning on accuracy, and the residual is the quantity
    distinctness measures. The conditioning fit is run separately inside each
    half so no information crosses between them.
    """
    ra = fit_D(long_a, **kw)
    rb = fit_D(long_b, **kw)
    if ra is None or rb is None:
        return None
    key = ["slide", "model"]
    fa = ra["frame"].set_index(key)["residual"]
    fb = rb["frame"].set_index(key)["residual"]
    shared = fa.index.intersection(fb.index)
    if len(shared) < 10:
        return None
    out = split_half(fa.loc[shared].to_numpy(), fb.loc[shared].to_numpy())
    out["n_cells"] = int(len(shared))
    return out


def corrected_D(r2_cv, r_sb):
    """D after dividing explained variance by the instrument's reliability.

    An unreliable score caps the variance any conditioner can explain at its
    reliability, so R2/r_sb is the explained share of the reliable variance.
    This is the classical correction for attenuation and it is an upper bound
    on how much of a D gap measurement error can account for.
    """
    if not np.isfinite(r_sb) or r_sb <= 0:
        return np.nan
    return 1.0 - float(np.clip(r2_cv / r_sb, 0.0, 1.0))


def corrected_delta(r2_mag, r_sb_mag, r2_loc, r_sb_loc):
    """Paired contrast after correcting both members for attenuation.

    `worst_case` takes the reliabilities at the ends of their intervals that
    maximise the correction, which bounds the share of the raw gap that
    differential reliability can explain.
    """
    dm = corrected_D(r2_mag, r_sb_mag)
    dl = corrected_D(r2_loc, r_sb_loc)
    return {
        "D_mag_corrected": dm,
        "D_loc_corrected": dl,
        "delta_D_corrected": dl - dm,
    }


def worst_case_delta(r2_mag, r2_loc, sb_mag_hi, sb_loc_lo):
    """The least favourable reading the reliability intervals permit.

    Magnitude reliability at its upper bound and localization at its lower
    bound is the combination that attributes as much of the gap as possible to
    measurement error rather than to the operators.
    """
    return corrected_delta(r2_mag, sb_mag_hi, r2_loc, sb_loc_lo)["delta_D_corrected"]
