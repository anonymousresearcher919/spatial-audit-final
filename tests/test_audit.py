"""Invariants that the audit depends on. Run with `pytest tests/`.

These are the properties whose violation silently changes a reported number:
the p-value floor, the bootstrap relabel, the direction of D on known inputs,
whether the domain graph actually uses coordinates, and whether each null
destroys the structure its operator measures.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NG = 300
GENES = [f"G{i:04d}" for i in range(NG)]


def _stub_resources():
    """Synthetic panels, so the tests need no licensed resource files."""
    rng = np.random.default_rng(0)
    m = types.ModuleType("audit.resources")
    m.LINEAGE_ORDER = [f"L{i}" for i in range(8)]
    m.lr_pairs = lambda: tuple((GENES[i], GENES[i + 1]) for i in range(0, 120, 2))
    m.mcp_markers = lambda: {
        f"L{i}": tuple(GENES[200 + i * 8 : 208 + i * 8]) for i in range(8)
    }
    weights = {f"P{j}": {g: float(rng.normal()) for g in GENES[:60]} for j in range(6)}
    m.progeny_weights = lambda: weights
    m.gene_index = lambda genes, wanted: np.array(
        [{g: i for i, g in enumerate(genes)}[w] for w in wanted if w in set(genes)],
        dtype=int,
    )
    m.select_hvg = lambda X, genes, n_top=1611, min_detect=0.10: (
        np.arange(X.shape[1]),
        genes,
    )
    sys.modules["audit.resources"] = m


_stub_resources()

from audit.core import REGISTRY, knn_graph, mc_pvalue, run_null, z_score  # noqa: E402
from audit.distinctness import (  # noqa: E402
    boot_slides,
    bootstrap_D,
    fit_D,
    paired_delta_D,
)
import audit.operators as O  # noqa: E402


# --------------------------------------------------------------------------
# Monte Carlo estimator
# --------------------------------------------------------------------------

def test_p_never_zero():
    p, _, _ = mc_pvalue(1e9, np.random.default_rng(0).normal(size=99))
    assert p > 0


@pytest.mark.parametrize("B,floor", [(99, 0.01), (39, 0.025)])
def test_p_floor(B, floor):
    p, _, _ = mc_pvalue(10.0, np.zeros(B))
    assert p == pytest.approx(floor)
    assert floor < 0.05, "B too small for the test to be able to reject at alpha=0.05"


def test_p_ceiling():
    assert mc_pvalue(0.0, np.zeros(39))[0] == pytest.approx(1.0)


def test_z_guards_zero_variance():
    assert np.isnan(z_score(1.0, np.ones(50)))


# --------------------------------------------------------------------------
# Distinctness
# --------------------------------------------------------------------------

def _frame(fn, n_slides=40, n_models=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_slides):
        base = rng.uniform(0.1, 0.8)
        for m in range(n_models):
            pcc = float(np.clip(base + rng.normal(0, 0.05), 0, 1))
            rows.append(
                {"slide": f"S{s}", "predictor": f"M{m}", "matched_pcc": pcc,
                 "score": fn(pcc, rng)}
            )
    return pd.DataFrame(rows)


def test_D_near_zero_for_smooth_function_of_accuracy():
    assert fit_D(_frame(lambda p, r: p ** 2))["D"] < 0.08


def test_D_near_one_for_score_independent_of_accuracy():
    assert fit_D(_frame(lambda p, r: r.normal()))["D"] > 0.9


def test_D_is_clipped():
    d = fit_D(_frame(lambda p, r: r.normal()))
    assert 0.0 <= d["D"] <= 1.0
    assert d["D_untruncated"] >= d["D"] - 1e-9


def test_bootstrap_relabels_duplicate_slides():
    d = _frame(lambda p, r: p ** 2)
    b = boot_slides(d, np.random.default_rng(1))
    assert len(b) == len(d)
    # a slide drawn twice must become two distinct CV groups
    assert b["slide"].nunique() == d["slide"].nunique()
    assert b["slide"].str.contains("__b").all()


def test_bootstrap_interval_brackets_estimate():
    r = bootstrap_D(_frame(lambda p, r: p ** 2), n_boot=40)
    assert r["ci_lo"] <= r["D"] <= r["ci_hi"]


def test_paired_delta_detects_the_noisier_member():
    r = paired_delta_D(
        _frame(lambda p, rr: p ** 2),
        _frame(lambda p, rr: rr.normal()),
        n_boot=40,
    )
    assert r["delta_D"] > 0.5 and r["p_gt0"] == 1.0


def test_fit_D_refuses_too_few_slides():
    d = _frame(lambda p, r: p ** 2)
    assert fit_D(d[d["slide"].isin(["S0", "S1"])]) is None


# --------------------------------------------------------------------------
# Graph and operators
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def slide():
    rng = np.random.default_rng(5)
    side = 20
    C = np.stack(np.meshgrid(np.arange(side), np.arange(side)), -1).reshape(-1, 2).astype(float)
    n = len(C)
    quad = (C[:, 0] >= side / 2).astype(int) * 2 + (C[:, 1] >= side / 2).astype(int)
    prog = rng.normal(size=(4, NG)) * 1.5
    Xm = np.abs(prog[quad] + rng.normal(0, 0.6, (n, NG))).astype(np.float32)
    Xm[rng.random(Xm.shape) < 0.1] = 0
    Xh = (Xm + rng.normal(0, 0.25, Xm.shape)).astype(np.float32).clip(0)
    Xbad = rng.random((n, NG)).astype(np.float32)
    return C, Xm, Xh, Xbad, quad


def test_knn_graph_is_row_standardised_and_loopless():
    C = np.random.default_rng(0).random((120, 2)) * 100
    W = knn_graph(C, k=8)
    assert np.allclose(np.asarray(W.sum(1)).ravel(), 1.0, atol=1e-5)
    assert W.diagonal().sum() == 0


def test_all_fifteen_operators_registered():
    assert len(REGISTRY) == 15


def test_agreement_arity_ignores_defaulted_hyperparameters():
    # mean_column_jaccard(Zhat, Z, q=0.90) must not be handed coordinates as q
    assert REGISTRY["lr_hotspot"]._arity == 2
    assert REGISTRY["domain_boundary"]._arity == 3


@pytest.mark.parametrize("name", [k for k, v in REGISTRY.items() if v.family != "domain"])
def test_operator_prefers_the_better_prediction(name, slide):
    C, Xm, Xh, Xbad, _ = slide
    op = REGISTRY[name]
    Z = op.fn(Xm, C, GENES)
    good = op.score(op.fn(Xh, C, GENES), Z, C)
    bad = op.score(op.fn(Xbad, C, GENES), Z, C)
    assert np.isfinite(good) and good > bad


def test_hotspot_jaccard_sits_at_the_analytic_floor():
    rng = np.random.default_rng(0)
    j = O.mean_column_jaccard(rng.random((400, 50)), rng.random((400, 50)), q=0.90)
    assert j == pytest.approx((1 - 0.90) / (1 + 0.90), abs=0.02)


def test_shared_row_shuffle_is_not_a_null_for_coexpression(slide):
    """The motivation for the per-gene perturbation."""
    C, Xm, _, _, _ = slide
    rng = np.random.default_rng(0)
    Z = O.phi_coexpr(Xm, C, GENES)
    shared = O.phi_coexpr(Xm[rng.permutation(len(C))], C, GENES)
    pergene = O.phi_coexpr(O.perturb_per_gene(Xm, C, rng), C, GENES)
    assert O.upper_triangle_spearman(shared, Z) > 0.999
    assert abs(O.upper_triangle_spearman(pergene, Z)) < 0.15


def test_bivariate_moran_guards_a_constant_field(slide):
    C, Xm, _, _, _ = slide
    W = knn_graph(C, k=8)
    assert np.isfinite(O._bivariate_moran_R(np.ones(len(C)), Xm[:, 0], W))


# --------------------------------------------------------------------------
# Domain family
# --------------------------------------------------------------------------

igraph = pytest.importorskip("igraph")
pytest.importorskip("leidenalg")


def test_domain_graph_actually_uses_coordinates(slide):
    """Clustering an expression-space neighbor graph ignores geometry: the
    partition would be identical after shuffling the coordinates."""
    from sklearn.metrics import adjusted_rand_score

    C, Xm, _, _, _ = slide
    rng = np.random.default_rng(0)
    real = O._domain_labels(Xm, C)
    shuffled = O._domain_labels(Xm, C[rng.permutation(len(C))])
    assert adjusted_rand_score(real, shuffled) < 0.10


def test_domain_clustering_recovers_planted_regions(slide):
    from sklearn.metrics import adjusted_rand_score

    C, Xm, _, _, quad = slide
    assert adjusted_rand_score(quad, O._domain_labels(Xm, C, res=0.25)) > 0.9


def test_boundary_agreement_invariant_to_label_renaming(slide):
    C, Xm, _, _, _ = slide
    lab = O._domain_labels(Xm, C)
    perm = np.random.default_rng(0).permutation(lab.max() + 1)
    assert O.boundary_agreement(perm[lab], lab, C) == pytest.approx(1.0)


def test_recluster_null_does_not_certify_noise(slide):
    """Permuting fitted labels instead puts pure noise above z=100."""
    C, Xm, _, Xbad, _ = slide
    r = run_null(REGISTRY["domain_boundary"], Xbad, Xm, C, GENES, B=19, seed=0)
    assert abs(r["z"]) < 10
    assert r["p"] > 0.05
