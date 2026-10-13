"""Operator registry, spatial graphs, and permutation-null machinery."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    torch = None
    HAS_CUDA = False


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

@dataclass
class Operator:
    """A map Phi: (n_spots, n_genes) -> Z, plus an agreement function on Z.

    fn(X, C, genes) -> Z
        X     (n_spots, n_genes) log1p expression, columns aligned to `genes`
        C     (n_spots, 2) spot coordinates
        genes list of HGNC symbols for the columns of X

    agreement(Zhat, Z) or agreement(Zhat, Z, C) -> float
        Arity is detected at registration; the 3-arg form is for agreements
        that need geometry (boundary overlap, neighborhood topology).

    null_on
        "input"  permute spot rows of Xhat before applying Phi
        "output" permute the entries of Zhat after applying Phi

    perturb
        Optional replacement for the default input permutation, for nulls
        that are not a single shared row shuffle (per-gene independent
        permutation for co-expression, re-clustering for the domain family).
        Signature perturb(X, C, rng) -> X_null.
    """

    name: str
    family: str
    fn: Callable
    agreement: Callable
    null_on: str = "input"
    n_perm: int = 99
    gene_set: str = "hvg"
    perturb: Optional[Callable] = None
    _arity: int = field(init=False, default=2)

    def __post_init__(self):
        if self.null_on not in ("input", "output"):
            raise ValueError(f"{self.name}: null_on must be 'input' or 'output'")
        # Count required positional parameters only. An agreement with a
        # defaulted hyperparameter (q for the hotspot Jaccard) must not be
        # mistaken for one that wants coordinates.
        params = inspect.signature(self.agreement).parameters.values()
        self._arity = sum(
            1
            for p in params
            if p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        )
        if self._arity not in (2, 3):
            raise ValueError(
                f"{self.name}: agreement must take 2 or 3 required arguments, "
                f"got {self._arity}"
            )

    def score(self, Zhat, Z, C) -> float:
        v = self.agreement(Zhat, Z, C) if self._arity == 3 else self.agreement(Zhat, Z)
        return float(v)


REGISTRY: Dict[str, Operator] = {}


def register(
    name, family, agreement, null_on="input", n_perm=99, gene_set="hvg", perturb=None
):
    def wrap(fn):
        if name in REGISTRY:
            raise KeyError(f"operator {name} already registered")
        REGISTRY[name] = Operator(
            name=name,
            family=family,
            fn=fn,
            agreement=agreement,
            null_on=null_on,
            n_perm=n_perm,
            gene_set=gene_set,
            perturb=perturb,
        )
        return fn

    return wrap


# --------------------------------------------------------------------------
# Spatial graph
# --------------------------------------------------------------------------

def knn_graph(C, k=8, row_standardise=True):
    """Symmetric k-NN adjacency on physical coordinates, self excluded."""
    n = C.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(C)
    _, idx = nn.kneighbors(C)
    idx = idx[:, 1:]
    rows = np.repeat(np.arange(n), idx.shape[1])
    W = csr_matrix(
        (np.ones(rows.size), (rows, idx.ravel())), shape=(n, n), dtype=np.float32
    )
    W = W.maximum(W.T)
    if row_standardise:
        deg = np.asarray(W.sum(1)).ravel()
        deg[deg == 0] = 1.0
        W = csr_matrix(W.multiply(1.0 / deg[:, None]))
    return W


def neighbor_mean(X, W):
    """Row-standardised neighborhood average of every gene field."""
    return np.asarray(W @ X, dtype=np.float32)


# --------------------------------------------------------------------------
# Monte Carlo significance
# --------------------------------------------------------------------------

def mc_pvalue(observed, null_draws, tail="greater"):
    """Finite-sample permutation p-value, Phipson & Smyth (2010).

    p = (1 + b) / (1 + B). The naive b/B admits p=0, which is not a valid
    estimate of a positive probability. Smallest attainable p is 1/(1+B),
    so B must satisfy 1/(1+B) <= alpha for the test to be able to reject.
    """
    null = np.asarray(null_draws, dtype=float)
    null = null[np.isfinite(null)]
    B = int(null.size)
    if B == 0 or not np.isfinite(observed):
        return np.nan, 0, np.nan
    if tail == "greater":
        b = int((null >= observed).sum())
    elif tail == "less":
        b = int((null <= observed).sum())
    else:
        raise ValueError("tail must be 'greater' or 'less'")
    sd = float(null.std(ddof=1)) if B > 1 else np.nan
    return (1.0 + b) / (1.0 + B), B, sd


def z_score(observed, null_draws):
    """Reported for contrast with mc_pvalue; not used as a gate.

    Null widths in this audit span two orders of magnitude, so a fixed z
    threshold is a different test on every operator.
    """
    null = np.asarray(null_draws, dtype=float)
    null = null[np.isfinite(null)]
    if null.size < 2:
        return np.nan
    sd = null.std(ddof=1)
    if sd < 1e-12:
        return np.nan
    return float((observed - null.mean()) / sd)


# --------------------------------------------------------------------------
# Null drivers
# --------------------------------------------------------------------------

def run_null(op, Xhat, Xmeas, C, genes, B=None, seed=0, Z_meas=None):
    """Observed score plus B null draws for one slide-model-operator cell.

    Returns dict with score, null array, p, z, null_sd.
    """
    B = op.n_perm if B is None else B
    rng = np.random.default_rng(seed)
    n = Xhat.shape[0]

    if Z_meas is None:
        Z_meas = op.fn(Xmeas, C, genes)
    Z_hat = op.fn(Xhat, C, genes)
    obs = op.score(Z_hat, Z_meas, C)

    draws = np.empty(B, dtype=float)
    if op.null_on == "input":
        for b in range(B):
            Xn = op.perturb(Xhat, C, rng) if op.perturb else Xhat[rng.permutation(n)]
            draws[b] = op.score(op.fn(Xn, C, genes), Z_meas, C)
    else:
        flat = np.asarray(Z_hat)
        for b in range(B):
            shuffled = flat.copy()
            rng.shuffle(shuffled.reshape(-1))
            draws[b] = op.score(shuffled.reshape(flat.shape), Z_meas, C)

    p, B_eff, sd = mc_pvalue(obs, draws)
    return {
        "operator": op.name,
        "score": obs,
        "p": p,
        "z": z_score(obs, draws),
        "null_sd": sd,
        "B": B_eff,
        "null": draws,
    }


# --------------------------------------------------------------------------
# GPU batching
# --------------------------------------------------------------------------
# Permutation nulls at B=99 across 61 slides x 6 models x 15 operators are
# dominated by repeated matrix products on the same operands. Operators whose
# Phi is expressible as dense linear algebra run all B permutations as one
# batched tensor op; measured 26.6x over the numpy loop on an A100.

def to_device(A, device="cuda"):
    if torch is None:
        raise RuntimeError("torch not available")
    return torch.as_tensor(np.ascontiguousarray(A), dtype=torch.float32, device=device)


def batched_permutations(n, B, seed, device="cuda"):
    """(B, n) permutation index matrix on device."""
    g = torch.Generator(device=device).manual_seed(int(seed))
    return torch.stack([torch.randperm(n, generator=g, device=device) for _ in range(B)])


def batched_pearson(A, B_):
    """Column-wise Pearson correlation of stacked batches.

    A  (B, n, p)   B_ (n, p) or (B, n, p)  ->  (B, p)
    """
    if B_.dim() == 2:
        B_ = B_.unsqueeze(0).expand_as(A)
    a = A - A.mean(dim=1, keepdim=True)
    b = B_ - B_.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = a.pow(2).sum(dim=1).sqrt() * b.pow(2).sum(dim=1).sqrt()
    return torch.where(den > 1e-12, num / den, torch.zeros_like(num))


def batched_jaccard(A, B_, q=0.90):
    """Hotspot Jaccard per column after within-column quantile thresholding."""
    if B_.dim() == 2:
        B_ = B_.unsqueeze(0).expand_as(A)
    ta = torch.quantile(A, q, dim=1, keepdim=True)
    tb = torch.quantile(B_, q, dim=1, keepdim=True)
    ha, hb = A >= ta, B_ >= tb
    inter = (ha & hb).sum(dim=1).float()
    union = (ha | hb).sum(dim=1).float()
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


def batched_lr_null(Xhat, W, lig_idx, rec_idx, Z_meas, B, seed, mode="magnitude", q=0.90):
    """All B permutations of the ligand-receptor operator in one pass.

    mode="magnitude" -> mean per-pair Pearson
    mode="hotspot"   -> mean per-pair Jaccard at quantile q
    """
    device = "cuda" if HAS_CUDA else "cpu"
    X = to_device(Xhat, device)
    Wt = to_device(W.toarray(), device)
    Zm = to_device(Z_meas, device)
    perms = batched_permutations(X.shape[0], B, seed, device)

    out = torch.empty(B, device=device)
    chunk = max(1, int(2e8 // (X.shape[0] * len(lig_idx) + 1)))
    for s in range(0, B, chunk):
        P = perms[s : s + chunk]
        Xp = X[P]
        L = Xp[:, :, lig_idx]
        R = torch.einsum("ij,bjp->bip", Wt, Xp[:, :, rec_idx])
        Zp = L * R
        agree = (
            batched_pearson(Zp, Zm)
            if mode == "magnitude"
            else batched_jaccard(Zp, Zm, q)
        )
        out[s : s + P.shape[0]] = agree.mean(dim=1)
    return out.cpu().numpy()
