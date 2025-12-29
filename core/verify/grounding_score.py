"""
Meridian Verify - Grounding Score

Determines if claims are supported by evidence.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .claims import Claim
from .retrieval import Evidence, RetrievalResult


class SupportLevel(Enum):
    """Level of evidence support for a claim."""
    SUPPORTED = "supported"
    REFUTED = "refuted"
    NOT_ENOUGH_INFO = "not_enough_info"


@dataclass
class GroundingResult:
    """Result of grounding a claim against evidence."""
    claim: Claim
    support_level: SupportLevel
    confidence: float
    best_evidence: Optional[Evidence]
    all_evidences: list[Evidence]
    explanation: str


def compute_grounding_score(
    claim: Claim,
    retrieval_result: RetrievalResult,
    support_threshold: float = 0.7,
    refute_threshold: float = 0.3,
) -> GroundingResult:
    """
    Determine if a claim is supported by retrieved evidence.
    
    Uses heuristic approach (no LLM judge required):
    - High keyword/entity overlap = likely supported
    - Contradicting patterns = likely refuted
    - Low overlap = not enough info
    """
    if not retrieval_result.evidences:
        return GroundingResult(
            claim=claim,
            support_level=SupportLevel.NOT_ENOUGH_INFO,
            confidence=0.0,
            best_evidence=None,
            all_evidences=[],
            explanation="No relevant evidence found.",
        )
    
    best_evidence = retrieval_result.evidences[0]
    
    # Compute overlap score
    overlap_score = compute_overlap(claim.text, best_evidence.text)
    
    # Check for contradiction patterns
    contradiction_score = check_contradictions(claim.text, best_evidence.text)
    
    # Determine support level
    if contradiction_score > 0.5:
        support_level = SupportLevel.REFUTED
        confidence = contradiction_score
        explanation = f"Evidence contradicts claim (score: {contradiction_score:.2f})."
    elif overlap_score >= support_threshold:
        support_level = SupportLevel.SUPPORTED
        confidence = overlap_score
        explanation = f"Claim supported by evidence (overlap: {overlap_score:.2f})."
    elif overlap_score >= refute_threshold:
        support_level = SupportLevel.NOT_ENOUGH_INFO
        confidence = 1 - overlap_score
        explanation = f"Partial evidence found, insufficient to confirm (overlap: {overlap_score:.2f})."
    else:
        support_level = SupportLevel.NOT_ENOUGH_INFO
        confidence = 0.8
        explanation = f"No strong evidence found (overlap: {overlap_score:.2f})."
    
    return GroundingResult(
        claim=claim,
        support_level=support_level,
        confidence=confidence,
        best_evidence=best_evidence,
        all_evidences=retrieval_result.evidences,
        explanation=explanation,
    )


def compute_overlap(claim_text: str, evidence_text: str) -> float:
    """Compute word overlap between claim and evidence."""
    import re
    
    def tokenize(text):
        return set(re.findall(r'\b\w+\b', text.lower()))
    
    claim_tokens = tokenize(claim_text)
    evidence_tokens = tokenize(evidence_text)
    
    if not claim_tokens:
        return 0.0
    
    overlap = len(claim_tokens & evidence_tokens)
    return overlap / len(claim_tokens)


def check_contradictions(claim_text: str, evidence_text: str) -> float:
    """
    Check for contradiction patterns between claim and evidence.
    
    Simple heuristic: look for negation patterns near shared entities.
    """
    import re
    
    # Negation words
    negations = {'not', 'never', 'no', 'none', 'neither', 'nobody', 'nothing', "n't", 'false', 'incorrect'}
    
    claim_lower = claim_text.lower()
    evidence_lower = evidence_text.lower()
    
    claim_tokens = set(re.findall(r'\b\w+\b', claim_lower))
    evidence_tokens = set(re.findall(r'\b\w+\b', evidence_lower))
    
    # Check if evidence has negations where claim doesn't (or vice versa)
    claim_negations = claim_tokens & negations
    evidence_negations = evidence_tokens & negations
    
    # If one has negations and the other doesn't, might be contradiction
    if bool(claim_negations) != bool(evidence_negations):
        # Check if they share key entities
        shared = claim_tokens & evidence_tokens - negations - {'the', 'a', 'an', 'is', 'was', 'are', 'were'}
        if len(shared) >= 2:  # At least 2 shared content words
            return 0.6
    
    # Check for numeric contradictions
    claim_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', claim_text)
    evidence_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', evidence_text)
    
    if claim_numbers and evidence_numbers:
        # If same context but different numbers
        claim_nums = set(float(n) for n in claim_numbers)
        evidence_nums = set(float(n) for n in evidence_numbers)
        if claim_nums and evidence_nums and claim_nums.isdisjoint(evidence_nums):
            # Check context overlap
            non_numeric_claim = re.sub(r'\d+(?:\.\d+)?', '', claim_lower)
            non_numeric_evidence = re.sub(r'\d+(?:\.\d+)?', '', evidence_lower)
            context_overlap = compute_overlap(non_numeric_claim, non_numeric_evidence)
            if context_overlap > 0.5:
                return 0.7
    
    return 0.0


@dataclass
class FactualityReport:
    """Aggregate report of factuality for a text."""
    total_claims: int
    supported: int
    refuted: int
    not_enough_info: int
    grounding_rate: float  # supported / total
    refutation_rate: float  # refuted / total
    results: list[GroundingResult]


def generate_factuality_report(
    grounding_results: list[GroundingResult],
) -> FactualityReport:
    """Generate aggregate factuality report."""
    total = len(grounding_results)
    
    supported = sum(1 for r in grounding_results if r.support_level == SupportLevel.SUPPORTED)
    refuted = sum(1 for r in grounding_results if r.support_level == SupportLevel.REFUTED)
    not_enough = sum(1 for r in grounding_results if r.support_level == SupportLevel.NOT_ENOUGH_INFO)
    
    return FactualityReport(
        total_claims=total,
        supported=supported,
        refuted=refuted,
        not_enough_info=not_enough,
        grounding_rate=supported / max(total, 1),
        refutation_rate=refuted / max(total, 1),
        results=grounding_results,
    )
