"""
Meridian Consistency - Similarity Measures

Defines similarity functions for output comparison.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SimilarityMethod(Enum):
    """Available similarity methods."""
    EXACT = "exact"
    EDIT_DISTANCE = "edit_distance"
    EMBEDDING = "embedding"
    JACCARD = "jaccard"


@dataclass
class SimilarityResult:
    """Result of similarity comparison."""
    method: SimilarityMethod
    score: float  # 0-1, higher = more similar
    is_match: bool
    details: dict


def exact_similarity(text1: str, text2: str) -> SimilarityResult:
    """Exact string match (binary)."""
    is_match = text1 == text2
    return SimilarityResult(
        method=SimilarityMethod.EXACT,
        score=1.0 if is_match else 0.0,
        is_match=is_match,
        details={},
    )


def edit_similarity(
    text1: str,
    text2: str,
    threshold: float = 0.1,
) -> SimilarityResult:
    """
    Edit distance similarity (Levenshtein).
    
    Score = 1 - (edit_distance / max_length)
    """
    try:
        from Levenshtein import distance as levenshtein_distance
    except ImportError:
        # Fallback to difflib
        import difflib
        ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
        return SimilarityResult(
            method=SimilarityMethod.EDIT_DISTANCE,
            score=ratio,
            is_match=ratio >= (1 - threshold),
            details={"fallback": "difflib"},
        )
    
    max_len = max(len(text1), len(text2), 1)
    edit_dist = levenshtein_distance(text1, text2)
    score = 1 - (edit_dist / max_len)
    
    return SimilarityResult(
        method=SimilarityMethod.EDIT_DISTANCE,
        score=score,
        is_match=score >= (1 - threshold),
        details={
            "edit_distance": edit_dist,
            "max_length": max_len,
        },
    )


def embedding_similarity(
    text1: str,
    text2: str,
    threshold: float = 0.9,
    model_name: str = "all-MiniLM-L6-v2",
    _model_cache: dict = {},
) -> SimilarityResult:
    """
    Semantic similarity using sentence embeddings.
    
    Score = cosine similarity of embeddings.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        # Fallback to edit distance
        return edit_similarity(text1, text2, 1 - threshold)
    
    # Cache model
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    
    model = _model_cache[model_name]
    embeddings = model.encode([text1, text2])
    
    # Cosine similarity
    cos_sim = float(np.dot(embeddings[0], embeddings[1]) / 
                   (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
    
    return SimilarityResult(
        method=SimilarityMethod.EMBEDDING,
        score=cos_sim,
        is_match=cos_sim >= threshold,
        details={"model": model_name},
    )


def jaccard_similarity(text1: str, text2: str, threshold: float = 0.8) -> SimilarityResult:
    """
    Jaccard similarity on word tokens.
    
    Score = |intersection| / |union|
    """
    import re
    
    tokens1 = set(re.findall(r'\b\w+\b', text1.lower()))
    tokens2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    if not tokens1 and not tokens2:
        return SimilarityResult(
            method=SimilarityMethod.JACCARD,
            score=1.0,
            is_match=True,
            details={},
        )
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    score = intersection / union if union > 0 else 0.0
    
    return SimilarityResult(
        method=SimilarityMethod.JACCARD,
        score=score,
        is_match=score >= threshold,
        details={
            "intersection": intersection,
            "union": union,
        },
    )


def compute_similarity(
    text1: str,
    text2: str,
    method: SimilarityMethod = SimilarityMethod.EXACT,
    threshold: float = 0.9,
) -> SimilarityResult:
    """
    Compute similarity between two texts.
    
    Args:
        text1, text2: Texts to compare
        method: Similarity method to use
        threshold: Match threshold (method-specific)
    
    Returns:
        SimilarityResult with score and match status
    """
    if method == SimilarityMethod.EXACT:
        return exact_similarity(text1, text2)
    elif method == SimilarityMethod.EDIT_DISTANCE:
        return edit_similarity(text1, text2, threshold)
    elif method == SimilarityMethod.EMBEDDING:
        return embedding_similarity(text1, text2, threshold)
    elif method == SimilarityMethod.JACCARD:
        return jaccard_similarity(text1, text2, threshold)
    else:
        raise ValueError(f"Unknown similarity method: {method}")
