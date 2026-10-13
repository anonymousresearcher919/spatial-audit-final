"""The fifteen downstream operators.

Each is Phi(X, C, genes) -> Z with an agreement function. None contains
learned parameters. Matched families share Phi and differ only in agreement:

    lr_magnitude / lr_hotspot       ligand-receptor product field
    mcp / mcp_hotspot               lineage marker-mean field
    moran / lisa                    spatial autocorrelation of HVGs
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from .core import knn_graph, neighbor_mean, register
from .resources import LINEAGE_ORDER, lr_pairs, mcp_markers, progeny_weights

K_NEIGHBORS = 8
HOTSPOT_Q = 0.90
COEXPR_PANEL = 1200
SVG_TOP = 200
LEIDEN_RES = 1.0
N_PCS = 30


# --------------------------------------------------------------------------
# Agreement functions
# --------------------------------------------------------------------------

def mean_column_pearson(Zhat, Z):
    """Mean per-column Pearson, skipping columns that are constant in either."""
    out = []
    for j in range(Z.shape[1]):
        a, b = Zhat[:, j], Z[:, j]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        out.append(pearsonr(a, b)[0])
    return float(np.nanmean(out)) if out else np.nan


def mean_column_jaccard(Zhat, Z, q=HOTSPOT_Q):
    """Mean per-column Jaccard of within-column top-(1-q) hotspot sets.

    The analytic floor for independent sets at q=0.90 is (1-q)/(1+q) = 0.053.
    """
    out = []
    for j in range(Z.shape[1]):
        a, b = Zhat[:, j], Z[:, j]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        ha = a >= np.quantile(a, q)
        hb = b >= np.quantile(b, q)
        u = (ha | hb).sum()
        out.append((ha & hb).sum() / u if u else np.nan)
    return float(np.nanmean(out)) if out else np.nan


def spearman_flat(Zhat, Z):
    a, b = np.asarray(Zhat).ravel(), np.asarray(Z).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return np.nan
    return float(spearmanr(a[m], b[m])[0])


def pearson_flat(Zhat, Z):
    a, b = np.asarray(Zhat).ravel(), np.asarray(Z).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return np.nan
    return float(pearsonr(a[m], b[m])[0])


def upper_triangle_spearman(Zhat, Z):
    iu = np.triu_indices_from(Z, k=1)
    return spearman_flat(Zhat[iu], Z[iu])


def set_jaccard(Zhat, Z):
    a, b = set(np.asarray(Zhat).ravel().tolist()), set(np.asarray(Z).ravel().tolist())
    u = len(a | b)
    return len(a & b) / u if u else np.nan


def label_ari(Zhat, Z):
    return float(adjusted_rand_score(np.asarray(Z).ravel(), np.asarray(Zhat).ravel()))


def label_nmi(Zhat, Z):
    return float(
        normalized_mutual_info_score(np.asarray(Z).ravel(), np.asarray(Zhat).ravel())
    )


def boundary_agreement(Zhat, Z, C):
    """Jaccard of boundary spot sets.

    A spot is a boundary spot if any of its k spatial neighbors carries a
    different domain label. This scores where the tissue is partitioned, not
    how the partitions are named, so it is invariant to label permutation in
    a way ARI and NMI are not.
    """
    W = knn_graph(C, k=K_NEIGHBORS, row_standardise=False)
    a, b = np.asarray(Zhat).ravel(), np.asarray(Z).ravel()
    ba, bb = np.zeros(len(a), bool), np.zeros(len(b), bool)
    indptr, indices = W.indptr, W.indices
    for i in range(len(a)):
        nb = indices[indptr[i] : indptr[i + 1]]
        if nb.size:
            ba[i] = np.any(a[nb] != a[i])
            bb[i] = np.any(b[nb] != b[i])
    u = (ba | bb).sum()
    return float((ba & bb).sum() / u) if u else np.nan


def neighborhood_agreement(Zhat, Z, C):
    """Frobenius-normalised similarity of lineage co-occurrence matrices."""
    a = np.asarray(Zhat, float)
    b = np.asarray(Z, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return float(np.sum((a / na) * (b / nb)))


# --------------------------------------------------------------------------
# Shared field constructors
# --------------------------------------------------------------------------

def _gene_index(genes, wanted):
    pos = {g: i for i, g in enumerate(genes)}
    return np.array([pos[g] for g in wanted if g in pos], dtype=int)


def _lr_field(X, C, genes, k=K_NEIGHBORS):
    """Phi_k(X)[i, p] = X[i, lig_p] * mean over k spatial neighbors of X[:, rec_p]."""
    pairs = [(l, r) for l, r in lr_pairs() if l in set(genes) and r in set(genes)]
    if not pairs:
        return np.zeros((X.shape[0], 0), dtype=np.float32)
    pos = {g: i for i, g in enumerate(genes)}
    li = np.array([pos[l] for l, _ in pairs])
    ri = np.array([pos[r] for _, r in pairs])
    W = knn_graph(C, k=k)
    Rbar = neighbor_mean(X[:, ri], W)
    return (X[:, li] * Rbar).astype(np.float32)


def _mcp_field(X, C, genes, min_expr=0.02, min_markers=3):
    """Mean expression of each lineage's marker panel, per spot."""
    detect = (X > 0).mean(axis=0)
    keep = set(np.asarray(genes)[detect >= min_expr].tolist())
    cols = []
    for pop in LINEAGE_ORDER:
        idx = _gene_index(genes, [g for g in mcp_markers()[pop] if g in keep])
        cols.append(
            X[:, idx].mean(axis=1) if idx.size >= min_markers else np.full(X.shape[0], np.nan)
        )
    Z = np.column_stack(cols).astype(np.float32)
    return Z[:, ~np.all(np.isnan(Z), axis=0)]


def _morans_i(X, W):
    """Moran's I per gene under a row-standardised weight matrix."""
    Xc = X - X.mean(axis=0, keepdims=True)
    num = np.einsum("ij,ij->j", Xc, np.asarray(W @ Xc))
    den = np.einsum("ij,ij->j", Xc, Xc)
    return np.where(den > 1e-12, num / den, 0.0)


def _local_morans_i(X, W):
    """Local Moran's I_i per spot per gene."""
    Xc = X - X.mean(axis=0, keepdims=True)
    s2 = (Xc ** 2).mean(axis=0, keepdims=True)
    s2 = np.where(s2 > 1e-12, s2, 1.0)
    return (Xc / s2) * np.asarray(W @ Xc)


def _bivariate_moran_R(A, B, W):
    """Bivariate Moran's R between two gene fields.

    Both fields are standardised explicitly rather than relying on a library
    flag; dividing by an unguarded standard deviation returns inf for the
    constant columns that permutation nulls routinely produce.
    """
    def z(v):
        s = v.std()
        return (v - v.mean()) / s if s > 1e-12 else np.zeros_like(v)

    Az, Bz = z(A), z(B)
    return float(np.mean(Az * np.asarray(W @ Bz)))


# --------------------------------------------------------------------------
# Null perturbations
# --------------------------------------------------------------------------

def perturb_per_gene(X, C, rng):
    """Permute each gene's spot vector independently.

    For co-expression the shared row shuffle is not a valid null: it leaves
    every gene-gene correlation exactly intact.
    """
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        out[:, j] = X[rng.permutation(X.shape[0]), j]
    return out


def perturb_recluster(X, C, rng):
    """Shuffle expression rows, then let the domain operator re-cluster.

    Permuting the fitted domain labels instead is not a null for this
    operator: it destroys spatial contiguity, which real domains have and
    shuffled labels never do, so pure noise certifies at z above 100. Shuffling
    the expression and re-running the clustering keeps the contiguity that the
    graph construction imposes, and leaves only the biological signal to test.
    """
    return X[rng.permutation(X.shape[0])]


# --------------------------------------------------------------------------
# Ligand-receptor
# --------------------------------------------------------------------------

@register("lr_magnitude", "lr", mean_column_pearson, gene_set="lr", n_perm=99)
def phi_lr_magnitude(X, C, genes):
    return _lr_field(X, C, genes)


@register("lr_hotspot", "lr", mean_column_jaccard, gene_set="lr", n_perm=99)
def phi_lr_hotspot(X, C, genes):
    return _lr_field(X, C, genes)


@register("spatialdm", "lr", spearman_flat, gene_set="lr", n_perm=50)
def phi_spatialdm(X, C, genes):
    """Bivariate Moran's R for each ligand-receptor pair."""
    pairs = [(l, r) for l, r in lr_pairs() if l in set(genes) and r in set(genes)]
    if not pairs:
        return np.zeros(0, dtype=np.float32)
    pos = {g: i for i, g in enumerate(genes)}
    W = knn_graph(C, k=K_NEIGHBORS)
    return np.array(
        [_bivariate_moran_R(X[:, pos[l]], X[:, pos[r]], W) for l, r in pairs],
        dtype=np.float32,
    )


# --------------------------------------------------------------------------
# Pathway activity
# --------------------------------------------------------------------------

@register("progeny", "pathway", mean_column_pearson, gene_set="pathway", n_perm=99)
def phi_progeny(X, C, genes):
    """Signed weighted gene sums over PROGENy pathway signatures.

    A signature is kept only if at least 30% of its genes and at least 20
    absolute survive the expression filter; the matched accuracy baseline
    follows the same narrowing.
    """
    W_df = progeny_weights()
    cols, gset = [], set(genes)
    pos = {g: i for i, g in enumerate(genes)}
    for pw, w in W_df.items():
        present = [g for g in w if g in gset]
        if len(present) < 20 or len(present) / len(w) < 0.30:
            continue
        idx = np.array([pos[g] for g in present])
        wts = np.array([w[g] for g in present], dtype=np.float32)
        cols.append(X[:, idx] @ wts)
    if not cols:
        return np.zeros((X.shape[0], 0), dtype=np.float32)
    return np.column_stack(cols).astype(np.float32)


# --------------------------------------------------------------------------
# Cell lineage
# --------------------------------------------------------------------------

@register("mcp", "lineage", mean_column_pearson, gene_set="marker", n_perm=99)
def phi_mcp(X, C, genes):
    return _mcp_field(X, C, genes)


@register("mcp_hotspot", "lineage", mean_column_jaccard, gene_set="marker", n_perm=99)
def phi_mcp_hotspot(X, C, genes):
    return _mcp_field(X, C, genes)


@register("neighborhood", "lineage", neighborhood_agreement, gene_set="marker", n_perm=99)
def phi_neighborhood(X, C, genes):
    """Lineage co-occurrence in k-nearest spatial neighborhoods.

    Each spot is assigned its dominant lineage; the operator returns the
    lineage-by-lineage count of adjacent pairs, row-normalised.
    """
    Z = _mcp_field(X, C, genes)
    if Z.shape[1] < 2:
        return np.zeros((1, 1), dtype=np.float32)
    lab = np.argmax(np.nan_to_num(Z, nan=-np.inf), axis=1)
    L = Z.shape[1]
    W = knn_graph(C, k=K_NEIGHBORS, row_standardise=False)
    M = np.zeros((L, L), dtype=np.float32)
    indptr, indices = W.indptr, W.indices
    for i in range(len(lab)):
        for j in indices[indptr[i] : indptr[i + 1]]:
            M[lab[i], lab[j]] += 1
    rs = M.sum(axis=1, keepdims=True)
    return M / np.where(rs > 0, rs, 1.0)


# --------------------------------------------------------------------------
# Spatial autocorrelation
# --------------------------------------------------------------------------

@register("moran", "autocorr", spearman_flat, gene_set="hvg", n_perm=99)
def phi_moran(X, C, genes):
    return _morans_i(X, knn_graph(C, k=K_NEIGHBORS)).astype(np.float32)


@register("moran_pea", "autocorr", pearson_flat, gene_set="hvg", n_perm=99)
def phi_moran_pea(X, C, genes):
    return _morans_i(X, knn_graph(C, k=K_NEIGHBORS)).astype(np.float32)


@register("svg_top200", "autocorr", set_jaccard, gene_set="hvg", n_perm=99)
def phi_svg_top200(X, C, genes):
    """Identity of the K most spatially variable genes."""
    I = _morans_i(X, knn_graph(C, k=K_NEIGHBORS))
    k = min(SVG_TOP, len(I))
    return np.argsort(-I)[:k]


@register("lisa", "autocorr", mean_column_jaccard, gene_set="hvg", n_perm=99)
def phi_lisa(X, C, genes):
    """Local indicators of spatial association, per spot per gene."""
    return _local_morans_i(X, knn_graph(C, k=K_NEIGHBORS)).astype(np.float32)


# --------------------------------------------------------------------------
# Tissue domains
# --------------------------------------------------------------------------

def _domain_labels(X, C, n_pcs=N_PCS, k=K_NEIGHBORS, res=LEIDEN_RES, seed=0):
    """Leiden communities on the physical k-NN graph, weighted by PCA similarity.

    Edges come from spatial adjacency and carry expression similarity as their
    weight. Clustering on an expression-space neighbor graph instead produces
    partitions whose ARI against a coordinate-shuffled copy of the same slide
    stays at the real value, because the coordinates were never used.
    """
    import igraph as ig
    import leidenalg

    p = min(n_pcs, min(X.shape) - 1)
    P = PCA(n_components=p, random_state=seed).fit_transform(
        X - X.mean(axis=0, keepdims=True)
    )
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(C))).fit(C)
    _, idx = nn.kneighbors(C)
    idx = idx[:, 1:]

    edges, weights = [], []
    scale = np.median(np.linalg.norm(P[:, None, :][:, 0, :] - P[idx[:, 0]], axis=1)) + 1e-8
    for i in range(len(C)):
        for j in idx[i]:
            if i < j:
                d = np.linalg.norm(P[i] - P[j])
                edges.append((int(i), int(j)))
                weights.append(float(np.exp(-(d ** 2) / (2 * scale ** 2))))

    g = ig.Graph(n=len(C), edges=edges)
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=res,
        seed=seed,
    )
    return np.asarray(part.membership, dtype=np.int32)


@register(
    "domain_ari", "domain", label_ari, gene_set="hvg", n_perm=39, perturb=perturb_recluster
)
def phi_domain_ari(X, C, genes):
    return _domain_labels(X, C)


@register(
    "domain_nmi", "domain", label_nmi, gene_set="hvg", n_perm=39, perturb=perturb_recluster
)
def phi_domain_nmi(X, C, genes):
    return _domain_labels(X, C)


@register(
    "domain_boundary",
    "domain",
    boundary_agreement,
    gene_set="hvg",
    n_perm=39,
    perturb=perturb_recluster,
)
def phi_domain_boundary(X, C, genes):
    return _domain_labels(X, C)


# --------------------------------------------------------------------------
# Co-expression
# --------------------------------------------------------------------------

@register(
    "coexpr",
    "coexpr",
    upper_triangle_spearman,
    gene_set="coexpr",
    n_perm=99,
    perturb=perturb_per_gene,
)
def phi_coexpr(X, C, genes):
    """Gene-by-gene Pearson correlation over a panel fixed across all slides."""
    p = min(COEXPR_PANEL, X.shape[1])
    Xs = X[:, :p]
    Xc = Xs - Xs.mean(axis=0, keepdims=True)
    sd = Xc.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Xz = Xc / sd
    return (Xz.T @ Xz / Xs.shape[0]).astype(np.float32)
