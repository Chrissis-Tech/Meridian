"""
Meridian Core Package - DEPRECATED

This module is deprecated. Please import from 'meridian' instead:
    from meridian.runner import run_suite
    from meridian.types import RunConfig

This compatibility wrapper will be removed in v1.0.0.
"""

import warnings

# Issue deprecation warning on import
warnings.warn(
    "The 'core' package is deprecated and will be removed in v1.0.0. "
    "Please use 'from meridian import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from meridian
from meridian.types import (
    TestCase,
    TestSuite,
    RunResult,
    SuiteResult,
    ComparisonResult,
    GenerationResult,
    ScoringResult,
    CausalTrace,
    ComponentImportance,
    FeatureActivation,
    AttackResult,
    SecurityReport,
    RunConfig,
    BaselineThresholds,
)

from meridian.config import config, get_config, get_available_models
from meridian.utils import (
    generate_run_id,
    hash_prompt,
    Timer,
    estimate_tokens,
    normalize_text,
    extract_json,
    load_jsonl,
    save_jsonl,
)

__all__ = [
    # Types
    "TestCase",
    "TestSuite",
    "RunResult",
    "SuiteResult",
    "ComparisonResult",
    "GenerationResult",
    "ScoringResult",
    "CausalTrace",
    "ComponentImportance",
    "FeatureActivation",
    "AttackResult",
    "SecurityReport",
    "RunConfig",
    "BaselineThresholds",
    # Config
    "config",
    "get_config",
    "get_available_models",
    # Utils
    "generate_run_id",
    "hash_prompt",
    "Timer",
    "estimate_tokens",
    "normalize_text",
    "extract_json",
    "load_jsonl",
    "save_jsonl",
]
