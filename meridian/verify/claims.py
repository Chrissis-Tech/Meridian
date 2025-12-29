"""
Meridian Verify - Claim Extraction

Extracts factual claims from text for verification.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Claim:
    """A factual claim extracted from text."""
    text: str
    claim_type: str  # statistic, citation, entity_fact, temporal, causal
    confidence: float
    source_span: tuple[int, int]
    entities: list[str]


def extract_claims(text: str) -> list[Claim]:
    """
    Extract verifiable factual claims from text.
    
    Focuses on:
    - Statistics and numbers
    - Citations and attributions
    - Entity-based facts
    - Temporal claims
    - Causal claims
    """
    claims = []
    
    # 1. Statistics (X% of Y, N million, etc.)
    stat_patterns = [
        (r'(\d+(?:\.\d+)?%?\s+(?:of|percent|percentage)[^.]*\.)', 'statistic'),
        (r'(\d+(?:\.\d+)?\s*(?:million|billion|thousand|hundred)[^.]*\.)', 'statistic'),
        (r'(approximately\s+\d+[^.]*\.)', 'statistic'),
        (r'(more than\s+\d+[^.]*\.)', 'statistic'),
        (r'(less than\s+\d+[^.]*\.)', 'statistic'),
    ]
    
    for pattern, claim_type in stat_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            claims.append(Claim(
                text=match.group(1).strip(),
                claim_type=claim_type,
                confidence=0.8,
                source_span=(match.start(), match.end()),
                entities=extract_entities(match.group(1)),
            ))
    
    # 2. Citations (Author (YYYY), [N], etc.)
    citation_patterns = [
        (r'([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?\s*\(\d{4}\)[^.]*\.)', 'citation'),
        (r'(according to [^.]+\.)', 'citation'),
        (r'(\[[^\]]+\]\s*(?:reported|showed|found|demonstrated)[^.]*\.)', 'citation'),
    ]
    
    for pattern, claim_type in citation_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            claims.append(Claim(
                text=match.group(1).strip(),
                claim_type=claim_type,
                confidence=0.9,
                source_span=(match.start(), match.end()),
                entities=extract_entities(match.group(1)),
            ))
    
    # 3. Entity facts (X is Y, X was born in, etc.)
    entity_patterns = [
        (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:is|was|are|were)\s+(?:the|a|an)\s+[^.]+\.)', 'entity_fact'),
        (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:invented|discovered|founded|created)\s+[^.]+\.)', 'entity_fact'),
        (r'(the capital of\s+[A-Z][a-z]+\s+is\s+[^.]+\.)', 'entity_fact'),
    ]
    
    for pattern, claim_type in entity_patterns:
        for match in re.finditer(pattern, text):
            claims.append(Claim(
                text=match.group(1).strip(),
                claim_type=claim_type,
                confidence=0.7,
                source_span=(match.start(), match.end()),
                entities=extract_entities(match.group(1)),
            ))
    
    # 4. Temporal claims (in YYYY, during the, etc.)
    temporal_patterns = [
        (r'(in\s+\d{4}[^.]*\.)', 'temporal'),
        (r'(since\s+\d{4}[^.]*\.)', 'temporal'),
        (r'(from\s+\d{4}\s+to\s+\d{4}[^.]*\.)', 'temporal'),
    ]
    
    for pattern, claim_type in temporal_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            claims.append(Claim(
                text=match.group(1).strip(),
                claim_type=claim_type,
                confidence=0.6,
                source_span=(match.start(), match.end()),
                entities=extract_entities(match.group(1)),
            ))
    
    # Deduplicate overlapping claims
    claims = dedupe_claims(claims)
    
    return claims


def extract_entities(text: str) -> list[str]:
    """Extract named entities from text (simple heuristic)."""
    # Simple capitalized word extraction
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    # Filter common words
    stopwords = {'The', 'This', 'That', 'These', 'Those', 'It', 'He', 'She', 'They'}
    return [w for w in words if w not in stopwords]


def dedupe_claims(claims: list[Claim]) -> list[Claim]:
    """Remove overlapping claims, keeping highest confidence."""
    if not claims:
        return []
    
    # Sort by confidence descending
    sorted_claims = sorted(claims, key=lambda c: -c.confidence)
    result = []
    used_spans = set()
    
    for claim in sorted_claims:
        start, end = claim.source_span
        # Check for overlap
        overlaps = any(
            not (end <= s or start >= e)
            for s, e in used_spans
        )
        if not overlaps:
            result.append(claim)
            used_spans.add((start, end))
    
    return result


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Simple sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]
