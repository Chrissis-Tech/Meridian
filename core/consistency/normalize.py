"""
Meridian Consistency - Normalization

Normalizes outputs for consistent comparison.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class NormalizationConfig:
    """Configuration for output normalization."""
    strip_whitespace: bool = True
    collapse_whitespace: bool = True
    lowercase: bool = False
    canonical_json: bool = True
    numeric_tolerance: Optional[float] = 1e-6
    remove_punctuation: bool = False
    stem_words: bool = False


def normalize(text: str, config: Optional[NormalizationConfig] = None) -> str:
    """
    Normalize text for comparison.
    
    Args:
        text: Input text
        config: Normalization options
    
    Returns:
        Normalized text
    """
    config = config or NormalizationConfig()
    result = text
    
    # 1. Strip leading/trailing whitespace
    if config.strip_whitespace:
        result = result.strip()
    
    # 2. Collapse internal whitespace
    if config.collapse_whitespace:
        result = re.sub(r'\s+', ' ', result)
    
    # 3. Lowercase
    if config.lowercase:
        result = result.lower()
    
    # 4. Canonical JSON
    if config.canonical_json:
        result = canonicalize_json(result)
    
    # 5. Normalize numbers (round to tolerance)
    if config.numeric_tolerance is not None:
        result = normalize_numbers(result, config.numeric_tolerance)
    
    # 6. Remove punctuation
    if config.remove_punctuation:
        result = re.sub(r'[^\w\s]', '', result)
    
    return result


def canonicalize_json(text: str) -> str:
    """
    Convert JSON to canonical form (sorted keys, minimal spacing).
    
    If text is not valid JSON, returns as-is.
    """
    # Try to find and canonicalize JSON in text
    try:
        parsed = json.loads(text)
        return json.dumps(parsed, sort_keys=True, separators=(',', ':'))
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON within text
    json_pattern = r'\{[^{}]*\}|\[[^\[\]]*\]'
    
    def replace_json(match):
        try:
            parsed = json.loads(match.group())
            return json.dumps(parsed, sort_keys=True, separators=(',', ':'))
        except json.JSONDecodeError:
            return match.group()
    
    return re.sub(json_pattern, replace_json, text)


def normalize_numbers(text: str, tolerance: float = 1e-6) -> str:
    """
    Normalize floating point numbers to consistent precision.
    """
    def round_number(match):
        try:
            num = float(match.group())
            # Round to significant figures based on tolerance
            if abs(num) < tolerance:
                return "0"
            rounded = round(num, -int(f"{tolerance:e}".split('e')[1]))
            # Format without trailing zeros
            if rounded == int(rounded):
                return str(int(rounded))
            return f"{rounded:g}"
        except ValueError:
            return match.group()
    
    # Match numbers (integers and floats)
    return re.sub(r'-?\d+\.?\d*(?:e[+-]?\d+)?', round_number, text, flags=re.IGNORECASE)


def extract_answer(text: str, answer_patterns: Optional[list[str]] = None) -> str:
    """
    Extract the answer portion from model output.
    
    Useful for consistency checking when preamble varies.
    """
    if answer_patterns is None:
        answer_patterns = [
            r'(?:answer|result|solution):\s*(.+)',
            r'(?:therefore|thus|so),?\s*(.+)',
            r'=\s*(.+)',
        ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Fallback: return last line or sentence
    lines = text.strip().split('\n')
    return lines[-1].strip() if lines else text
