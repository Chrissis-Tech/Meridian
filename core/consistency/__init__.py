"""Meridian Consistency Package - Formalized Consistency Analysis"""

from .normalize import (
    NormalizationConfig,
    normalize,
    canonicalize_json,
    normalize_numbers,
    extract_answer,
)
from .similarity import (
    SimilarityMethod,
    SimilarityResult,
    compute_similarity,
    exact_similarity,
    edit_similarity,
    embedding_similarity,
    jaccard_similarity,
)
from .cluster import (
    Cluster,
    ClusteringResult,
    cluster_outputs,
    compute_consistency_metrics,
)

__all__ = [
    # Normalization
    "NormalizationConfig",
    "normalize",
    "canonicalize_json",
    "normalize_numbers",
    "extract_answer",
    # Similarity
    "SimilarityMethod",
    "SimilarityResult",
    "compute_similarity",
    "exact_similarity",
    "edit_similarity",
    "embedding_similarity",
    "jaccard_similarity",
    # Clustering
    "Cluster",
    "ClusteringResult",
    "cluster_outputs",
    "compute_consistency_metrics",
]
