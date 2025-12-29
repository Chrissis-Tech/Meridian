"""
Meridian Reliability - Verbosity Penalty

Detects length gaming where models pad outputs to game metrics.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VerbosityResult:
    """Result of verbosity analysis."""
    score: float  # 0-1, higher = more appropriate verbosity
    word_count: int
    expected_range: tuple[int, int]
    is_verbose: bool
    is_terse: bool
    penalty: float
    details: dict


def count_content_words(text: str) -> int:
    """Count content words (excluding filler)."""
    import re
    
    # Common filler patterns
    filler_patterns = [
        r'\b(um|uh|like|you know|basically|actually|literally)\b',
        r'\b(well|so|anyway|however|therefore|furthermore)\b',
        r'\b(in order to|due to the fact that|as a matter of fact)\b',
    ]
    
    cleaned = text.lower()
    for pattern in filler_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    words = cleaned.split()
    return len([w for w in words if len(w) > 1])


def estimate_expected_length(prompt: str, task_type: str = "general") -> tuple[int, int]:
    """
    Estimate expected output length range for a prompt.
    
    Returns (min_words, max_words) tuple.
    """
    prompt_len = len(prompt.split())
    
    # Task-specific heuristics
    length_profiles = {
        "classification": (1, 10),
        "short_answer": (5, 50),
        "explanation": (30, 200),
        "essay": (100, 500),
        "code": (10, 300),
        "json": (5, 100),
    }
    
    if task_type in length_profiles:
        return length_profiles[task_type]
    
    # General heuristic: output ~0.5-2x prompt length
    min_len = max(5, prompt_len // 2)
    max_len = max(50, prompt_len * 2)
    
    return (min_len, max_len)


def calculate_verbosity_penalty(
    output: str,
    prompt: str,
    expected_range: Optional[tuple[int, int]] = None,
    task_type: str = "general",
) -> VerbosityResult:
    """
    Calculate penalty for inappropriate verbosity.
    
    Penalizes:
    - Excessive length (padding to game metrics)
    - Insufficient length (truncated/incomplete)
    """
    word_count = count_content_words(output)
    
    if expected_range is None:
        expected_range = estimate_expected_length(prompt, task_type)
    
    min_words, max_words = expected_range
    
    # Calculate penalty
    penalty = 0.0
    is_verbose = False
    is_terse = False
    
    if word_count > max_words:
        # Penalize verbosity (potential gaming)
        excess_ratio = (word_count - max_words) / max_words
        penalty = min(1.0, excess_ratio * 0.5)  # 50% penalty per 100% excess
        is_verbose = True
    elif word_count < min_words:
        # Penalize terseness (incomplete)
        deficit_ratio = (min_words - word_count) / max(min_words, 1)
        penalty = min(1.0, deficit_ratio * 0.3)  # 30% penalty per 100% deficit
        is_terse = True
    
    # Score: 1 - penalty
    score = max(0.0, 1.0 - penalty)
    
    return VerbosityResult(
        score=score,
        word_count=word_count,
        expected_range=expected_range,
        is_verbose=is_verbose,
        is_terse=is_terse,
        penalty=penalty,
        details={
            "min_expected": min_words,
            "max_expected": max_words,
            "content_words": word_count,
        }
    )


def detect_padding_patterns(text: str) -> dict:
    """
    Detect common padding/filler patterns used for length gaming.
    """
    import re
    
    patterns = {
        "repetition": r'(.{20,})\1+',  # Repeated phrases
        "filler_phrases": r'\b(in conclusion|to summarize|as mentioned|as I said)\b',
        "hedge_words": r'\b(perhaps|maybe|possibly|potentially|arguably)\b',
        "excessive_qualifiers": r'\b(very|really|quite|extremely|absolutely)\b',
        "redundant_intro": r'^(I think that|It is important to note that|As we know)',
    }
    
    results = {}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        results[name] = len(matches)
    
    return results
