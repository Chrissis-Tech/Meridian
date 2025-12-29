"""
Meridian Consistency - Clustering

Clusters outputs to identify modes for consistency analysis.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from .similarity import SimilarityMethod, compute_similarity


@dataclass
class Cluster:
    """A cluster of similar outputs."""
    id: int
    representative: str
    members: list[str]
    size: int


@dataclass
class ClusteringResult:
    """Result of output clustering."""
    clusters: list[Cluster]
    num_clusters: int
    dominant_cluster: Cluster
    mode_consistency: float  # size of dominant / total
    pairwise_agreement: float  # fraction of pairs that match


def cluster_outputs(
    outputs: list[str],
    similarity_method: SimilarityMethod = SimilarityMethod.EXACT,
    threshold: float = 0.9,
) -> ClusteringResult:
    """
    Cluster outputs by similarity to identify modes.
    
    Uses greedy single-linkage clustering:
    1. Each output starts as its own cluster
    2. Merge clusters if any pair is similar above threshold
    
    Args:
        outputs: List of output strings
        similarity_method: How to compare outputs
        threshold: Similarity threshold for clustering
    
    Returns:
        ClusteringResult with clusters and consistency metrics
    """
    if not outputs:
        return ClusteringResult(
            clusters=[],
            num_clusters=0,
            dominant_cluster=Cluster(0, "", [], 0),
            mode_consistency=0.0,
            pairwise_agreement=0.0,
        )
    
    n = len(outputs)
    
    # Initialize: each output is its own cluster
    cluster_ids = list(range(n))
    
    # Compute pairwise similarities and cluster
    similarity_matrix = {}
    for i in range(n):
        for j in range(i + 1, n):
            result = compute_similarity(
                outputs[i], 
                outputs[j], 
                similarity_method, 
                threshold
            )
            similarity_matrix[(i, j)] = result.score
            
            # Merge if similar
            if result.is_match:
                # Union-find merge
                old_id = cluster_ids[j]
                new_id = cluster_ids[i]
                for k in range(n):
                    if cluster_ids[k] == old_id:
                        cluster_ids[k] = new_id
    
    # Build cluster objects
    cluster_map = {}
    for idx, cid in enumerate(cluster_ids):
        if cid not in cluster_map:
            cluster_map[cid] = {
                "representative": outputs[idx],
                "members": [],
            }
        cluster_map[cid]["members"].append(outputs[idx])
    
    clusters = [
        Cluster(
            id=cid,
            representative=data["representative"],
            members=data["members"],
            size=len(data["members"]),
        )
        for cid, data in cluster_map.items()
    ]
    
    # Sort by size descending
    clusters.sort(key=lambda c: -c.size)
    dominant = clusters[0] if clusters else Cluster(0, "", [], 0)
    
    # Calculate pairwise agreement
    if n > 1:
        num_pairs = n * (n - 1) // 2
        matching_pairs = sum(
            1 for i in range(n) for j in range(i + 1, n)
            if cluster_ids[i] == cluster_ids[j]
        )
        pairwise_agreement = matching_pairs / num_pairs
    else:
        pairwise_agreement = 1.0
    
    return ClusteringResult(
        clusters=clusters,
        num_clusters=len(clusters),
        dominant_cluster=dominant,
        mode_consistency=dominant.size / n if n > 0 else 0.0,
        pairwise_agreement=pairwise_agreement,
    )


def compute_consistency_metrics(
    outputs: list[str],
    similarity_method: SimilarityMethod = SimilarityMethod.EXACT,
    threshold: float = 0.9,
) -> dict:
    """
    Compute comprehensive consistency metrics for a set of outputs.
    
    Returns:
        Dictionary with mode_consistency, pairwise_agreement, 
        num_clusters, and cluster_distribution
    """
    result = cluster_outputs(outputs, similarity_method, threshold)
    
    cluster_sizes = [c.size for c in result.clusters]
    total = sum(cluster_sizes)
    
    return {
        "mode_consistency": result.mode_consistency,
        "pairwise_agreement": result.pairwise_agreement,
        "num_clusters": result.num_clusters,
        "num_outputs": total,
        "dominant_cluster_size": result.dominant_cluster.size,
        "cluster_distribution": [s / total if total > 0 else 0 for s in cluster_sizes],
        "is_consistent": result.mode_consistency >= 0.8,
    }
