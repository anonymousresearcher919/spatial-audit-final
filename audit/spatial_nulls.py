"""Spatially constrained permutation nulls.

The default input null in `core.run_null` shuffles spot rows independently.
That destroys spatial autocorrelation at the same time as it destroys
alignment with the measured field, so an operator that scored nothing but the
smoothness shared by any two tissue-shaped fields would still beat it. A block
permutation separates the two: contiguous tiles move as units, so
autocorrelation inside a tile survives, while the correspondence between
predicted and measured positions is gone.

Used as the `perturb` hook on an Operator:

    op = replace(REGISTRY["lr_magnitude"], perturb=block_perturb(25))
    run_null(op, Xhat, Xmeas, C, genes, B=99)
"""

from __future__ import annotations

from math import factorial

import numpy as np


def square_blocks(C, n_blocks=25):
    """Assign each spot to a contiguous square tile of the bounding box.

    The grid is square, so the number of occupied tiles is at most n_blocks and
    usually fewer: tissue does not fill its own bounding box. `block_report`
    gives the count that actually results.
    """
    side = int(np.ceil(np.sqrt(n_blocks)))
    C = np.asarray(C, dtype=float)
    lo = C.min(axis=0)
    span = C.max(axis=0) - lo
    span = np.where(span > 0, span, 1.0)
    ij = np.clip(np.floor((C - lo) / span * side).astype(int), 0, side - 1)
    return ij[:, 0] * side + ij[:, 1]


def block_report(C, n_blocks=25, min_block=2):
    """Occupancy of the tiling, and the p-value floor it can support.

    Two separate limits bite at small block counts and they are easy to
    confuse. The permutation floor 1/(1+B) is a property of B alone. The
    distinct-arrangement limit is a property of the tiling: k occupied blocks
    admit k! arrangements, so once B exceeds k! - 1 the extra draws are
    repeats and the null stops gaining resolution.
    """
    lab = square_blocks(C, n_blocks)
    uniq, counts = np.unique(lab, return_counts=True)
    usable = counts >= min_block
    k = int(usable.sum())
    arrangements = factorial(k) - 1 if k < 20 else float("inf")
    return {
        "n_blocks_requested": int(n_blocks),
        "n_blocks_occupied": int(uniq.size),
        "n_blocks_usable": k,
        "spots_per_block_median": float(np.median(counts[usable])) if k else 0.0,
        "spots_per_block_min": int(counts[usable].min()) if k else 0,
        "max_informative_B": arrangements,
        "B_for_alpha_05": 19,
    }


def block_perturb(n_blocks=25, min_block=2):
    """Return a perturb(X, C, rng) that permutes whole contiguous tiles.

    Tiles are unequal in size because tissue is not square. The permutation is
    therefore applied to the concatenated spot-index stream in permuted tile
    order, which means a tile receives values from its mapped source tile and,
    where the sizes differ, from the next source tile in the stream. The
    alternative of pairing only equal-sized tiles would discard most of the
    slide.

    Tiles below `min_block` spots are left in place rather than moved. Moving a
    one-spot tile is a plain row swap, which is the null this function exists
    to avoid.
    """

    def perturb(X, C, rng):
        lab = square_blocks(C, n_blocks)
        uniq, counts = np.unique(lab, return_counts=True)
        movable = [b for b, c in zip(uniq, counts) if c >= min_block]
        if len(movable) < 2:
            return X[rng.permutation(X.shape[0])]

        groups = [np.flatnonzero(lab == b) for b in movable]
        dst = np.concatenate(groups)
        src = np.concatenate([groups[o] for o in rng.permutation(len(groups))])

        perm = np.arange(X.shape[0])
        perm[dst] = src
        return X[perm]

    return perturb


def block_size_sweep(op, Xhat, Xmeas, C, genes, counts=(4, 9, 16, 25, 49, 100, 400),
                     B=19, seed=0, run=None):
    """Null mean and observed score across tilings, for one slide-model cell.

    A block null that is sensitive to block size is reporting the size rather
    than the autocorrelation. This is the preflight that establishes it is not,
    and it writes nothing: run it on a held-out slide before committing to a
    block count.
    """
    from dataclasses import replace

    if run is None:
        from .core import run_null as run

    rows = []
    for k in counts:
        rep = block_report(C, k)
        if rep["n_blocks_usable"] < 2:
            continue
        r = run(replace(op, perturb=block_perturb(k)), Xhat, Xmeas, C, genes,
                B=B, seed=seed)
        rows.append({
            "n_blocks": k,
            "n_usable": rep["n_blocks_usable"],
            "spots_per_block": rep["spots_per_block_median"],
            "score": r["score"],
            "null_mean": float(np.nanmean(r["null"])),
            "null_sd": r["null_sd"],
            "p": r["p"],
        })
    return rows


def random_set_jaccard_floor(q=0.90):
    """Expected Jaccard of two independent top-(1-q) sets: (1-q)/(1+q).

    The hotspot null should land here. It is the check that distinguishes a
    null that has removed the signal from one that has merely attenuated it.
    """
    return (1.0 - q) / (1.0 + q)
