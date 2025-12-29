"""
Meridian Scoring - Regex Match

Pattern-based scoring using regular expressions.
"""

import re
from typing import Optional, Union

from ..types import ScoringResult


def regex_match(
    output: str,
    pattern: str,
    flags: int = 0,
    capture_group: Optional[int] = None,
    expected_capture: Optional[str] = None,
) -> ScoringResult:
    """
    Score output by regex pattern matching.
    
    Args:
        output: Model output
        pattern: Regex pattern to match
        flags: Regex flags (e.g., re.IGNORECASE)
        capture_group: Optional capture group to extract
        expected_capture: Expected value for capture group
        
    Returns:
        ScoringResult with pass/fail and captured groups
    """
    try:
        compiled = re.compile(pattern, flags)
        match = compiled.search(output)
        
        if match is None:
            return ScoringResult(
                passed=False,
                score=0.0,
                method="regex_match",
                details={
                    "pattern": pattern,
                    "matched": False,
                    "output_preview": output[:200],
                }
            )
        
        # If we need to check a capture group
        if capture_group is not None and expected_capture is not None:
            try:
                captured = match.group(capture_group)
                passed = captured == expected_capture
                return ScoringResult(
                    passed=passed,
                    score=1.0 if passed else 0.0,
                    method="regex_match",
                    details={
                        "pattern": pattern,
                        "matched": True,
                        "capture_group": capture_group,
                        "captured": captured,
                        "expected": expected_capture,
                    }
                )
            except IndexError:
                return ScoringResult(
                    passed=False,
                    score=0.0,
                    method="regex_match",
                    details={
                        "pattern": pattern,
                        "error": f"Capture group {capture_group} not found",
                    }
                )
        
        # Pattern matched
        return ScoringResult(
            passed=True,
            score=1.0,
            method="regex_match",
            details={
                "pattern": pattern,
                "matched": True,
                "match": match.group(0)[:100],
                "groups": match.groups()[:5] if match.groups() else None,
            }
        )
        
    except re.error as e:
        return ScoringResult(
            passed=False,
            score=0.0,
            method="regex_match",
            details={
                "pattern": pattern,
                "error": f"Invalid regex: {str(e)}",
            }
        )


def regex_fullmatch(
    output: str,
    pattern: str,
    flags: int = 0,
) -> ScoringResult:
    """
    Score output by requiring full regex match.
    
    Args:
        output: Model output
        pattern: Regex pattern that must match entire output
        flags: Regex flags
        
    Returns:
        ScoringResult with pass/fail
    """
    try:
        compiled = re.compile(pattern, flags)
        # Strip whitespace for fullmatch
        match = compiled.fullmatch(output.strip())
        
        passed = match is not None
        
        return ScoringResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            method="regex_fullmatch",
            details={
                "pattern": pattern,
                "matched": passed,
                "output_preview": output[:200],
            }
        )
        
    except re.error as e:
        return ScoringResult(
            passed=False,
            score=0.0,
            method="regex_fullmatch",
            details={
                "pattern": pattern,
                "error": f"Invalid regex: {str(e)}",
            }
        )


def multi_pattern_match(
    output: str,
    patterns: list[str],
    require_all: bool = True,
    flags: int = 0,
) -> ScoringResult:
    """
    Score output by matching multiple patterns.
    
    Args:
        output: Model output
        patterns: List of regex patterns
        require_all: If True, all patterns must match. If False, any match passes.
        flags: Regex flags
        
    Returns:
        ScoringResult with details on which patterns matched
    """
    matches = {}
    for pattern in patterns:
        try:
            compiled = re.compile(pattern, flags)
            match = compiled.search(output)
            matches[pattern] = match is not None
        except re.error:
            matches[pattern] = False
    
    if require_all:
        passed = all(matches.values())
    else:
        passed = any(matches.values())
    
    match_count = sum(matches.values())
    score = match_count / len(patterns) if patterns else 0.0
    
    return ScoringResult(
        passed=passed,
        score=score,
        method="multi_pattern_match",
        details={
            "patterns_matched": matches,
            "match_count": match_count,
            "total_patterns": len(patterns),
            "require_all": require_all,
        }
    )


def forbidden_patterns(
    output: str,
    patterns: list[str],
    flags: int = re.IGNORECASE,
) -> ScoringResult:
    """
    Check that output does NOT match any forbidden patterns.
    
    Args:
        output: Model output
        patterns: List of forbidden regex patterns
        flags: Regex flags
        
    Returns:
        ScoringResult (passes if no patterns match)
    """
    violations = []
    
    for pattern in patterns:
        try:
            compiled = re.compile(pattern, flags)
            match = compiled.search(output)
            if match:
                violations.append({
                    "pattern": pattern,
                    "matched": match.group(0)[:50],
                })
        except re.error:
            pass
    
    passed = len(violations) == 0
    
    return ScoringResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        method="forbidden_patterns",
        details={
            "violations": violations,
            "violation_count": len(violations),
        }
    )
