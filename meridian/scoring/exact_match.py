"""
Meridian Scoring - Exact Match

Exact string matching scorer with normalization options.
"""

import re
from typing import Optional

from ..types import ScoringResult


def exact_match(
    output: str,
    expected: str,
    normalize: bool = True,
    case_sensitive: bool = False,
    strip_punctuation: bool = False,
) -> ScoringResult:
    """
    Score output by exact match with expected value.
    
    Args:
        output: Model output
        expected: Expected value
        normalize: Whether to normalize whitespace
        case_sensitive: Whether to use case-sensitive matching
        strip_punctuation: Whether to remove punctuation
        
    Returns:
        ScoringResult with pass/fail and details
    """
    # Prepare strings
    out = output.strip()
    exp = expected.strip()
    
    if normalize:
        out = " ".join(out.split())
        exp = " ".join(exp.split())
    
    if not case_sensitive:
        out = out.lower()
        exp = exp.lower()
    
    if strip_punctuation:
        out = re.sub(r'[^\w\s]', '', out)
        exp = re.sub(r'[^\w\s]', '', exp)
    
    passed = out == exp
    
    return ScoringResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        method="exact_match",
        details={
            "output": output[:200],
            "expected": expected[:200],
            "normalized": normalize,
            "case_sensitive": case_sensitive,
        }
    )


def contains_match(
    output: str,
    expected: str,
    case_sensitive: bool = False,
) -> ScoringResult:
    """
    Score output by checking if it contains the expected value.
    
    Args:
        output: Model output
        expected: Value that should be contained
        case_sensitive: Whether to use case-sensitive matching
        
    Returns:
        ScoringResult with pass/fail
    """
    out = output.strip()
    exp = expected.strip()
    
    if not case_sensitive:
        out = out.lower()
        exp = exp.lower()
    
    passed = exp in out
    
    return ScoringResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        method="contains_match",
        details={
            "output_preview": output[:200],
            "expected": expected,
            "found": passed,
        }
    )


def numeric_match(
    output: str,
    expected: float,
    tolerance: float = 1e-6,
) -> ScoringResult:
    """
    Score output by extracting and comparing numeric values.
    
    Args:
        output: Model output
        expected: Expected numeric value
        tolerance: Absolute tolerance for comparison
        
    Returns:
        ScoringResult with pass/fail
    """
    from ..utils import extract_number
    
    extracted = extract_number(output)
    
    if extracted is None:
        return ScoringResult(
            passed=False,
            score=0.0,
            method="numeric_match",
            details={
                "output": output[:200],
                "expected": expected,
                "extracted": None,
                "error": "No number found in output",
            }
        )
    
    passed = abs(extracted - expected) <= tolerance
    
    return ScoringResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        method="numeric_match",
        details={
            "expected": expected,
            "extracted": extracted,
            "difference": abs(extracted - expected),
            "tolerance": tolerance,
        }
    )


def multi_choice_match(
    output: str,
    correct_answer: str,
    choices: Optional[list[str]] = None,
) -> ScoringResult:
    """
    Score multiple choice responses.
    
    Args:
        output: Model output
        correct_answer: The correct choice (e.g., "A", "B", "C", "D")
        choices: Optional list of valid choices
        
    Returns:
        ScoringResult with pass/fail
    """
    # Normalize
    out = output.strip().upper()
    correct = correct_answer.strip().upper()
    
    # Try to extract just the letter
    letter_match = re.search(r'\b([A-D])\b', out)
    if letter_match:
        out = letter_match.group(1)
    
    passed = out == correct
    
    return ScoringResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        method="multi_choice_match",
        details={
            "output": output[:100],
            "extracted": out,
            "correct": correct,
        }
    )
