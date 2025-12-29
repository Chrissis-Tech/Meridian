"""
Meridian Scoring - Calibration Metrics

Measures model calibration (ECE, MCE) for uncertainty quantification.
"""

import numpy as np
from ..types import ScoringResult, CalibrationResult


def expected_calibration_error(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> CalibrationResult:
    """Calculate ECE and MCE."""
    if len(confidences) == 0:
        return CalibrationResult(ece=0.0, mce=0.0, bin_accuracies=[], bin_confidences=[])
    
    confidences = np.array(confidences)
    correct = np.array(correct, dtype=float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies, bin_confidences, bin_counts = [], [], []
    
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences >= lower) & (confidences < upper) if i < n_bins - 1 else (confidences >= lower) & (confidences <= upper)
        count = np.sum(in_bin)
        bin_counts.append(int(count))
        bin_accuracies.append(float(np.mean(correct[in_bin])) if count > 0 else 0.0)
        bin_confidences.append(float(np.mean(confidences[in_bin])) if count > 0 else (lower + upper) / 2)
    
    total = len(confidences)
    ece = sum((bin_counts[i] / total) * abs(bin_accuracies[i] - bin_confidences[i]) for i in range(n_bins) if bin_counts[i] > 0)
    gaps = [abs(bin_accuracies[i] - bin_confidences[i]) for i in range(n_bins) if bin_counts[i] > 0]
    
    return CalibrationResult(
        ece=float(ece), mce=max(gaps) if gaps else 0.0,
        reliability_diagram={"bin_edges": bin_boundaries.tolist(), "bin_counts": bin_counts},
        bin_accuracies=bin_accuracies, bin_confidences=bin_confidences,
    )


def calibration_score(confidences: list[float], correct: list[bool], ece_threshold: float = 0.1) -> ScoringResult:
    """Score model calibration."""
    result = expected_calibration_error(confidences, correct)
    return ScoringResult(
        passed=result.ece <= ece_threshold, score=max(0.0, 1.0 - result.ece),
        method="calibration", details={"ece": result.ece, "mce": result.mce}
    )
