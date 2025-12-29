"""
Meridian Stats - Bootstrap Confidence Intervals
"""

import numpy as np
from typing import Optional


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    statistic: str = "mean",
) -> tuple[float, tuple[float, float]]:
    """Calculate bootstrap confidence interval."""
    values = np.array(values)
    
    if len(values) < 2:
        mean = np.mean(values) if len(values) > 0 else 0
        return float(mean), (float(mean), float(mean))
    
    stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        if statistic == "mean":
            stats.append(np.mean(sample))
        elif statistic == "median":
            stats.append(np.median(sample))
        elif statistic == "std":
            stats.append(np.std(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(stats, alpha / 2 * 100)
    upper = np.percentile(stats, (1 - alpha / 2) * 100)
    center = np.mean(values) if statistic == "mean" else np.median(values)
    
    return float(center), (float(lower), float(upper))


def bootstrap_difference(
    values_a: list[float],
    values_b: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> tuple[float, tuple[float, float]]:
    """Bootstrap CI for difference between two groups."""
    a = np.array(values_a)
    b = np.array(values_b)
    
    diffs = []
    for _ in range(n_bootstrap):
        sample_a = np.random.choice(a, size=len(a), replace=True)
        sample_b = np.random.choice(b, size=len(b), replace=True)
        diffs.append(np.mean(sample_b) - np.mean(sample_a))
    
    alpha = 1 - confidence
    lower = np.percentile(diffs, alpha / 2 * 100)
    upper = np.percentile(diffs, (1 - alpha / 2) * 100)
    center = np.mean(b) - np.mean(a)
    
    return float(center), (float(lower), float(upper))
