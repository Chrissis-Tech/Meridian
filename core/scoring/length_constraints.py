"""
Meridian Scoring - Length Constraints

Validates output length (words, sentences, characters).
"""

import re
from typing import Optional

from ..types import ScoringResult
from ..utils import count_words, count_sentences


def check_length_constraints(
    output: str,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    min_sentences: Optional[int] = None,
    max_sentences: Optional[int] = None,
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> ScoringResult:
    """
    Check that output meets length constraints.
    
    Args:
        output: Model output
        min_words: Minimum word count
        max_words: Maximum word count
        min_sentences: Minimum sentence count
        max_sentences: Maximum sentence count
        min_chars: Minimum character count
        max_chars: Maximum character count
        
    Returns:
        ScoringResult with constraint check details
    """
    violations = []
    checks = []
    
    # Word count
    word_count = count_words(output)
    if min_words is not None:
        checks.append(("min_words", min_words))
        if word_count < min_words:
            violations.append(f"Too few words: {word_count} < {min_words}")
    
    if max_words is not None:
        checks.append(("max_words", max_words))
        if word_count > max_words:
            violations.append(f"Too many words: {word_count} > {max_words}")
    
    # Sentence count
    sentence_count = count_sentences(output)
    if min_sentences is not None:
        checks.append(("min_sentences", min_sentences))
        if sentence_count < min_sentences:
            violations.append(f"Too few sentences: {sentence_count} < {min_sentences}")
    
    if max_sentences is not None:
        checks.append(("max_sentences", max_sentences))
        if sentence_count > max_sentences:
            violations.append(f"Too many sentences: {sentence_count} > {max_sentences}")
    
    # Character count
    char_count = len(output)
    if min_chars is not None:
        checks.append(("min_chars", min_chars))
        if char_count < min_chars:
            violations.append(f"Too few characters: {char_count} < {min_chars}")
    
    if max_chars is not None:
        checks.append(("max_chars", max_chars))
        if char_count > max_chars:
            violations.append(f"Too many characters: {char_count} > {max_chars}")
    
    passed = len(violations) == 0
    score = 1.0 - (len(violations) / len(checks)) if checks else 1.0
    
    return ScoringResult(
        passed=passed,
        score=max(0.0, score),
        method="length_constraints",
        details={
            "word_count": word_count,
            "sentence_count": sentence_count,
            "char_count": char_count,
            "violations": violations,
            "constraints": dict(checks),
        }
    )


def check_single_sentence(output: str) -> ScoringResult:
    """
    Check that output is exactly one sentence.
    
    Args:
        output: Model output
        
    Returns:
        ScoringResult
    """
    sentence_count = count_sentences(output)
    passed = sentence_count == 1
    
    return ScoringResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        method="single_sentence",
        details={
            "sentence_count": sentence_count,
            "expected": 1,
        }
    )


def check_single_word(output: str) -> ScoringResult:
    """
    Check that output is exactly one word.
    
    Args:
        output: Model output
        
    Returns:
        ScoringResult
    """
    # Strip and check
    stripped = output.strip()
    words = stripped.split()
    word_count = len(words)
    passed = word_count == 1
    
    return ScoringResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        method="single_word",
        details={
            "word_count": word_count,
            "expected": 1,
            "output": stripped[:50],
        }
    )


def check_bullet_points(
    output: str,
    min_bullets: int = 1,
    max_bullets: Optional[int] = None,
    bullet_patterns: Optional[list[str]] = None,
) -> ScoringResult:
    """
    Check that output contains bullet points.
    
    Args:
        output: Model output
        min_bullets: Minimum number of bullet points
        max_bullets: Maximum number of bullet points
        bullet_patterns: Patterns that count as bullets (default: -, *, •, numbers)
        
    Returns:
        ScoringResult
    """
    if bullet_patterns is None:
        bullet_patterns = [
            r'^[\s]*[-*•][\s]+',  # - * • bullets
            r'^[\s]*\d+[.)]\s',    # 1. 2) numbered lists
        ]
    
    # Count bullet points
    lines = output.split('\n')
    bullet_count = 0
    
    for line in lines:
        for pattern in bullet_patterns:
            if re.match(pattern, line):
                bullet_count += 1
                break
    
    violations = []
    
    if bullet_count < min_bullets:
        violations.append(f"Too few bullets: {bullet_count} < {min_bullets}")
    
    if max_bullets is not None and bullet_count > max_bullets:
        violations.append(f"Too many bullets: {bullet_count} > {max_bullets}")
    
    passed = len(violations) == 0
    
    return ScoringResult(
        passed=passed,
        score=1.0 if passed else 0.5,
        method="bullet_points",
        details={
            "bullet_count": bullet_count,
            "min_bullets": min_bullets,
            "max_bullets": max_bullets,
            "violations": violations,
        }
    )


def check_word_restrictions(
    output: str,
    forbidden_words: Optional[list[str]] = None,
    required_words: Optional[list[str]] = None,
    case_sensitive: bool = False,
) -> ScoringResult:
    """
    Check word inclusion/exclusion constraints.
    
    Args:
        output: Model output
        forbidden_words: Words that must NOT appear
        required_words: Words that MUST appear
        case_sensitive: Whether matching is case-sensitive
        
    Returns:
        ScoringResult
    """
    violations = []
    
    check_output = output if case_sensitive else output.lower()
    
    # Check forbidden words
    if forbidden_words:
        for word in forbidden_words:
            check_word = word if case_sensitive else word.lower()
            # Use word boundary matching
            pattern = r'\b' + re.escape(check_word) + r'\b'
            if re.search(pattern, check_output, 0 if case_sensitive else re.IGNORECASE):
                violations.append(f"Forbidden word found: '{word}'")
    
    # Check required words
    missing_required = []
    if required_words:
        for word in required_words:
            check_word = word if case_sensitive else word.lower()
            pattern = r'\b' + re.escape(check_word) + r'\b'
            if not re.search(pattern, check_output, 0 if case_sensitive else re.IGNORECASE):
                missing_required.append(word)
                violations.append(f"Required word missing: '{word}'")
    
    passed = len(violations) == 0
    
    # Calculate partial score
    total_checks = len(forbidden_words or []) + len(required_words or [])
    score = 1.0 - (len(violations) / total_checks) if total_checks > 0 else 1.0
    
    return ScoringResult(
        passed=passed,
        score=max(0.0, score),
        method="word_restrictions",
        details={
            "violations": violations,
            "missing_required": missing_required,
            "forbidden_found": [v.split("'")[1] for v in violations if "Forbidden" in v],
        }
    )
