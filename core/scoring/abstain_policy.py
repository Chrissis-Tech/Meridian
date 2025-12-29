"""
Meridian Scoring - Abstain Policy

Selective prediction: when to refuse/abstain instead of answering.
"""

from typing import Optional
from ..types import ScoringResult


def should_abstain(
    confidence: Optional[float] = None,
    consistency_score: Optional[float] = None,
    contradiction_detected: bool = False,
    entropy: Optional[float] = None,
    confidence_threshold: float = 0.5,
    consistency_threshold: float = 0.6,
    entropy_threshold: float = 1.5,
) -> tuple[bool, str]:
    """
    Determine if model should abstain from answering.
    
    Returns (should_abstain, reason)
    """
    if contradiction_detected:
        return True, "contradiction_detected"
    if confidence is not None and confidence < confidence_threshold:
        return True, f"low_confidence_{confidence:.2f}"
    if consistency_score is not None and consistency_score < consistency_threshold:
        return True, f"low_consistency_{consistency_score:.2f}"
    if entropy is not None and entropy > entropy_threshold:
        return True, f"high_entropy_{entropy:.2f}"
    return False, "confident"


def abstention_score(
    abstained: bool,
    should_have_abstained: bool,
) -> ScoringResult:
    """Score abstention decision correctness."""
    if abstained == should_have_abstained:
        return ScoringResult(
            passed=True, score=1.0, method="abstention",
            details={"correct_decision": True, "abstained": abstained}
        )
    return ScoringResult(
        passed=False, score=0.0, method="abstention",
        details={"correct_decision": False, "abstained": abstained, "should_have": should_have_abstained}
    )
