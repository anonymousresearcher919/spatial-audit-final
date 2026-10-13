from .core import REGISTRY, Operator, knn_graph, mc_pvalue, register, run_null, z_score
from .distinctness import (
    bootstrap_D,
    boot_slides,
    fit_D,
    paired_delta_D,
    recoverability,
)
from .errors import error_conditioned_D, error_features, error_morans_i
from .reliability import (
    corrected_D,
    corrected_delta,
    repeated_split_half,
    residualized_split_half,
    spearman_brown,
    worst_case_delta,
)
from .spatial_nulls import (
    block_perturb,
    block_report,
    block_size_sweep,
    random_set_jaccard_floor,
    square_blocks,
)
from . import operators  # noqa: F401  populates REGISTRY

__all__ = [
    "REGISTRY",
    "Operator",
    "register",
    "run_null",
    "mc_pvalue",
    "z_score",
    "knn_graph",
    "fit_D",
    "boot_slides",
    "bootstrap_D",
    "paired_delta_D",
    "recoverability",
    "block_perturb",
    "block_report",
    "block_size_sweep",
    "square_blocks",
    "random_set_jaccard_floor",
    "error_features",
    "error_morans_i",
    "error_conditioned_D",
    "spearman_brown",
    "repeated_split_half",
    "residualized_split_half",
    "corrected_D",
    "corrected_delta",
    "worst_case_delta",
]

__version__ = "1.1.0"
