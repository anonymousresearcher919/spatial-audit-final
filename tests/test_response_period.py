"""Invariants for the analyses added during the response period.

Same contract as tests/test_audit.py: no data download, no licensed
resources, and fast enough to run on every change. The properties here are
the ones whose violation would change a number in the author responses.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit.core import knn_graph
from audit.errors import error_features, error_morans_i
from audit.reliability import (
    corrected_D,
    repeated_split_half,
    spearman_brown,
    split_half,
    worst_case_delta,
)
from audit.spatial_nulls import (
    block_perturb,
    block_report,
    random_set_jaccard_floor,
    square_blocks,
)


# --------------------------------------------------------------------------
# Block permutation null
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grid():
    side = 40
    C = np.stack(np.meshgrid(np.arange(side), np.arange(side)), -1)
    C = C.reshape(-1, 2).astype(float)
    # a smooth field: value depends on position, so shuffling rows destroys
    # autocorrelation while moving whole tiles largely preserves it
    smooth = np.sin(C[:, 0] / 6.0) + np.cos(C[:, 1] / 6.0)
    return C, smooth.reshape(-1, 1).astype(np.float32)


def _morans(x, W):
    return float(error_morans_i(np.asarray(x).reshape(-1, 1), W)[0])


def test_block_permutation_preserves_autocorrelation(grid):
    """The whole reason the block null exists.

    A spot shuffle must drive Moran's I to roughly zero. A block permutation
    must leave most of it standing, or it is not a constrained null.
    """
    C, X = grid
    W = knn_graph(C, k=8)
    rng = np.random.default_rng(0)

    base = _morans(X[:, 0], W)
    spot = _morans(X[rng.permutation(len(C)), 0], W)
    block = _morans(block_perturb(25)(X, C, rng)[:, 0], W)

    assert base > 0.8
    assert abs(spot) < 0.1
    assert block > 0.5 * base


def test_block_permutation_is_a_permutation(grid):
    C, X = grid
    out = block_perturb(25)(X, C, np.random.default_rng(1))
    assert np.allclose(np.sort(out[:, 0]), np.sort(X[:, 0]))


def test_block_permutation_actually_moves_spots(grid):
    C, X = grid
    out = block_perturb(25)(X, C, np.random.default_rng(2))
    assert (out[:, 0] != X[:, 0]).mean() > 0.5


def test_square_blocks_are_contiguous(grid):
    """Every tile must be a connected region of the plane.

    A tiling that interleaves distant spots would preserve nothing, and the
    bug would be invisible in the null mean.
    """
    C, _ = grid
    lab = square_blocks(C, 25)
    for b in np.unique(lab):
        P = C[lab == b]
        assert np.ptp(P[:, 0]) <= np.ptp(C[:, 0]) / 5 + 1e-9
        assert np.ptp(P[:, 1]) <= np.ptp(C[:, 1]) / 5 + 1e-9


def test_block_report_flags_too_few_arrangements(grid):
    """Four tiles admit 23 distinct rearrangements.

    B=99 against a 4-tile tiling spends 76 draws on repeats. The report has to
    say so, because the resulting p-value looks ordinary.
    """
    C, _ = grid
    assert block_report(C, 4)["max_informative_B"] == 23
    assert block_report(C, 25)["max_informative_B"] == float("inf")
    assert block_report(C, 4)["B_for_alpha_05"] == 19


def test_jaccard_floor_matches_the_analytic_value():
    assert random_set_jaccard_floor(0.90) == pytest.approx(0.0526, abs=1e-4)


# --------------------------------------------------------------------------
# Error features
# --------------------------------------------------------------------------

def test_error_features_zero_for_perfect_reconstruction(grid):
    C, X = grid
    f = error_features(X, X, C)
    assert f["err_rmse"] == pytest.approx(0.0)
    assert f["err_mae"] == pytest.approx(0.0)


def test_error_moran_separates_patchy_from_speckled_error(grid):
    """The feature has to distinguish the two things it is named for.

    Error that is large but spatially random and error that is small but
    organised into patches are different failure modes. If err_moran did not
    separate them, conditioning on it would be conditioning on magnitude
    twice.
    """
    C, X = grid
    rng = np.random.default_rng(3)
    patchy = X + (np.sin(C[:, 0] / 5.0)).reshape(-1, 1) * 0.5
    speckled = X + rng.normal(0, 0.5, X.shape)
    assert error_features(patchy, X, C)["err_moran"] > 0.5
    assert error_features(speckled, X, C)["err_moran"] < 0.2


def test_error_rmse_is_blind_to_the_spatial_pattern(grid):
    """The converse: magnitude must not be a proxy for smoothness."""
    C, X = grid
    rng = np.random.default_rng(4)
    e = rng.normal(0, 0.3, X.shape)
    scrambled = e[rng.permutation(len(C))]
    a = error_features(X + e, X, C)["err_rmse"]
    b = error_features(X + scrambled, X, C)["err_rmse"]
    assert a == pytest.approx(b, rel=1e-6)


# --------------------------------------------------------------------------
# Reliability and the correction for attenuation
# --------------------------------------------------------------------------

def test_spearman_brown_raises_half_length_reliability():
    assert spearman_brown(0.5) == pytest.approx(2 / 3)
    assert spearman_brown(0.864) == pytest.approx(0.927, abs=5e-4)
    assert spearman_brown(1.0) == pytest.approx(1.0)


def test_correction_only_ever_lowers_D():
    """Dividing explained variance by a reliability below one raises R2.

    If the correction ever raised D it would be arguing the wrong direction:
    the point of correcting is to ask how much of a gap survives after the
    noisier instrument is given the benefit of the doubt.
    """
    for r2 in (0.1, 0.5, 0.9):
        for sb in (0.7, 0.85, 0.99):
            assert corrected_D(r2, sb) <= 1.0 - r2 + 1e-12


def test_correction_reproduces_the_reported_mcp_values():
    assert corrected_D(1 - 0.093, spearman_brown(0.864)) == pytest.approx(0.022, abs=2e-3)
    assert corrected_D(1 - 0.410, spearman_brown(0.779)) == pytest.approx(0.326, abs=2e-3)


def test_worst_case_delta_stays_positive_for_the_mcp_family():
    """The adverse bound reported to the reviewer.

    Magnitude reliability at its upper bound and localization at its lower
    bound is the reading that attributes as much of the gap as possible to
    measurement error. It has to leave the contrast positive or the claim in
    the response is wrong.
    """
    d = worst_case_delta(1 - 0.093, 1 - 0.410,
                         spearman_brown(0.904), spearman_brown(0.691))
    assert d == pytest.approx(0.233, abs=3e-3)
    assert d > 0


def test_corrected_D_guards_nonpositive_reliability():
    assert np.isnan(corrected_D(0.5, 0.0))
    assert np.isnan(corrected_D(0.5, -0.2))


def test_split_half_refuses_too_few_cells():
    assert np.isnan(split_half(np.arange(5.0), np.arange(5.0))["r_sb"])


def test_repeated_split_half_varies_over_partitions():
    """A single arbitrary split is one draw, not an estimate.

    The earlier implementation looped over n_rep while reading the same two
    fixed columns, so every repetition returned the same number and the
    interval was degenerate.
    """
    rng = np.random.default_rng(7)
    units = np.arange(60)
    truth = rng.normal(size=80)
    loading = rng.uniform(0.3, 1.0, units.size)

    def score_fn(subset):
        w = loading[subset].mean()
        return truth * w + rng.normal(0, 0.35, truth.size)

    out = repeated_split_half(score_fn, units, n_rep=40, seed=0)
    assert out["n_rep"] == 40
    assert out["ci_hi"] > out["ci_lo"]
    assert -1.0 <= out["r_sb"] <= 1.0


# --------------------------------------------------------------------------
# The synthetic pipeline
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic():
    from make_synthetic_long import build

    long, tissue_of = build(seed=0)
    long["tissue"] = long["slide"].map(tissue_of)
    return long


def test_synthetic_table_has_the_real_shape(synthetic):
    assert len(synthetic) == 61 * 6 * 15
    assert synthetic["slide"].nunique() == 61
    assert synthetic["predictor"].nunique() == 6
    assert synthetic["operator"].nunique() == 15


def test_loader_rejects_a_table_missing_columns(tmp_path):
    from audit.io import load_long

    p = tmp_path / "bad.csv"
    pd.DataFrame({"slide": ["a"], "score": [1.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_long(p)


def test_loader_accepts_the_legacy_model_column(tmp_path):
    """Earlier drafts called the predictor column `model`.

    analyses.py read `model` while run_audit.py wrote `predictor`, so the
    reordering analysis raised KeyError on the table the audit produced. The
    loader now normalises, and this pins it.
    """
    from audit.io import load_long

    p = tmp_path / "legacy.csv"
    pd.DataFrame({
        "slide": ["a"], "model": ["M"], "operator": ["op"], "family": ["f"],
        "score": [1.0], "p": [0.01], "matched_pcc": [0.5],
    }).to_csv(p, index=False)
    assert "predictor" in load_long(p).columns


def test_loader_raises_on_unmapped_slides(tmp_path):
    from audit.io import load_long

    p = tmp_path / "x.csv"
    pd.DataFrame({
        "slide": ["a", "b"], "predictor": ["M", "M"], "operator": ["op", "op"],
        "family": ["f", "f"], "score": [1.0, 2.0], "p": [0.01, 0.02],
        "matched_pcc": [0.5, 0.6],
    }).to_csv(p, index=False)
    with pytest.raises(ValueError, match="no tissue"):
        load_long(p, tissue_map={"a": "ccRCC"})


def test_synthetic_pipeline_recovers_the_published_partition(synthetic):
    """The end-to-end assertion the synthetic path exists for.

    The generated couplings are set from the published D values, so the
    analysis half of the pipeline must sort the fifteen operators into the
    same three groups with the same membership. The values will not match and
    are not meant to.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from analyze import operator_table

    t = operator_table(synthetic, n_boot=25, seed=0)
    group = dict(zip(t["operator"], t["group"]))

    expected_A = {"spatialdm", "coexpr", "domain_boundary", "lisa",
                  "lr_hotspot", "mcp_hotspot"}
    expected_B = {"lr_magnitude", "progeny", "mcp"}
    expected_C = {"svg_top200", "domain_ari", "domain_nmi", "moran",
                  "moran_pea", "neighborhood"}

    assert {o for o, g in group.items() if g.startswith("A")} == expected_A
    assert {o for o, g in group.items() if g.startswith("B")} == expected_B
    assert {o for o, g in group.items() if g.startswith("C")} == expected_C


def test_redundant_operators_reorder_fewer_pairs(synthetic):
    """The claim the health framing rests on.

    An operator that is redundant with accuracy must reorder fewer model pairs
    than one that is distinct from it. If this ordering ever inverted, the
    reordering table in the responses would be reporting noise.
    """
    from audit.analyses import rank_reversals

    redundant = max(rank_reversals(synthetic, op)["reversal"]
                    for op in ("mcp", "progeny", "lr_magnitude"))
    distinct = min(rank_reversals(synthetic, op)["reversal"]
                   for op in ("lr_hotspot", "mcp_hotspot", "lisa", "coexpr"))
    assert redundant < distinct
