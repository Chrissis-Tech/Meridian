"""
Meridian Core Types

Dataclasses and type definitions for the entire system.
Includes types for test cases, results, causal tracing, and SAE features.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional



class TaskType(Enum):
    """Types of evaluation tasks."""
    INSTRUCTION_FOLLOWING = "instruction_following"
    CONSISTENCY = "consistency"
    MATH_SHORT = "math_short"
    HALLUCINATION_CONTROL = "hallucination_control"
    ROBUSTNESS_NOISE = "robustness_noise"
    NEEDLE_HAYSTACK = "needle_haystack"
    SECURITY_ADVERSARIAL = "security_adversarial"


class ScoringMethod(Enum):
    """Available scoring methods."""
    EXACT_MATCH = "exact_match"
    REGEX_MATCH = "regex_match"
    JSON_SCHEMA = "json_schema"
    LENGTH_CONSTRAINT = "length_constraint"
    HEURISTIC_HALLUCINATION = "heuristic_hallucination"
    SELF_CONSISTENCY = "self_consistency"
    LLM_JUDGE = "llm_judge"


class ModelType(Enum):
    """Types of model adapters."""
    LOCAL_TRANSFORMERS = "local_transformers"
    OPENAI_API = "openai_api"
    ANTHROPIC_API = "anthropic_api"



@dataclass
class ExpectedOutput:
    """Expected output specification for a test case."""
    type: str  # "exact", "regex", "json_schema", "contains", "length"
    value: Optional[str] = None
    schema: Optional[dict] = None
    pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    forbidden_words: Optional[list[str]] = None
    required_words: Optional[list[str]] = None


@dataclass
class ScoringConfig:
    """Scoring configuration for a test case."""
    method: str
    weight: float = 1.0
    params: dict = field(default_factory=dict)


@dataclass
class TestCase:
    """A single test case in a suite."""
    id: str
    task: str
    prompt: str
    expected: ExpectedOutput
    scoring: ScoringConfig
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    # For consistency tests
    num_runs: int = 1
    
    # For needle-in-haystack
    needle: Optional[str] = None
    haystack_size: Optional[int] = None


@dataclass
class TestSuite:
    """A collection of test cases."""
    name: str
    description: str
    task_type: TaskType
    test_cases: list[TestCase]
    version: str = "0.3.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())



@dataclass
class GenerationResult:
    """Result from a model generation call."""
    output: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    finish_reason: str
    raw_response: Optional[dict] = None
    logprobs: Optional[list[float]] = None
    
    # For local models with interpretability
    hidden_states: Optional[Any] = None  # torch.Tensor
    attentions: Optional[Any] = None  # torch.Tensor


@dataclass
class RunResult:
    """Result of running a single test case."""
    test_id: str
    model_id: str
    output: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_estimate: float
    timestamp: str
    run_hash: str
    
    # Scoring results
    passed: bool = False
    score: float = 0.0
    scoring_method: str = ""
    scoring_details: dict = field(default_factory=dict)
    
    # Confidence/uncertainty
    confidence: Optional[float] = None
    entropy: Optional[float] = None
    
    # For consistency runs
    run_index: int = 0
    
    # Error tracking
    error: Optional[str] = None
    
    # Interpretability data (stored separately, referenced by hash)
    interpretability_ref: Optional[str] = None


@dataclass
class ConsistencyResult:
    """Result of multiple consistency runs."""
    test_id: str
    model_id: str
    num_runs: int
    outputs: list[str]
    
    # Consistency metrics
    self_consistency_score: float = 0.0
    entropy: float = 0.0
    contradiction_detected: bool = False
    variance: float = 0.0
    
    # Individual run results
    run_results: list[RunResult] = field(default_factory=list)



@dataclass
class ScoringResult:
    """Result from a scorer."""
    passed: bool
    score: float  # 0.0 to 1.0
    method: str
    details: dict = field(default_factory=dict)
    
    # For statistical significance
    confidence_interval: Optional[tuple[float, float]] = None
    p_value: Optional[float] = None


@dataclass
class CalibrationResult:
    """Calibration metrics for a model."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    reliability_diagram: dict = field(default_factory=dict)
    bin_accuracies: list[float] = field(default_factory=list)
    bin_confidences: list[float] = field(default_factory=list)



@dataclass
class SuiteResult:
    """Aggregate result for an entire test suite."""
    suite_name: str
    model_id: str
    run_id: str
    timestamp: str
    
    # Aggregate metrics
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    
    # Performance
    accuracy: float = 0.0
    mean_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    
    # With confidence intervals
    accuracy_ci: Optional[tuple[float, float]] = None
    latency_ci: Optional[tuple[float, float]] = None
    
    # Individual results
    results: list[RunResult] = field(default_factory=list)
    
    # Failure analysis
    common_failure_patterns: list[str] = field(default_factory=list)
    failure_by_tag: dict = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result of comparing two runs/models."""
    run_a_id: str
    run_b_id: str
    model_a: str
    model_b: str
    suite_name: str
    
    # Delta metrics
    accuracy_delta: float = 0.0
    accuracy_delta_ci: Optional[tuple[float, float]] = None
    accuracy_significant: bool = False
    accuracy_p_value: Optional[float] = None
    
    latency_delta: float = 0.0
    cost_delta: float = 0.0
    
    # Regression detection
    regressions: list[str] = field(default_factory=list)  # test IDs that got worse
    improvements: list[str] = field(default_factory=list)  # test IDs that got better
    
    # Effect size
    cohens_d: Optional[float] = None



@dataclass
class AttentionPattern:
    """Attention pattern for a single head."""
    layer: int
    head: int
    attention_matrix: Any  # numpy array [seq_len, seq_len]
    tokens: list[str]


@dataclass
class LogitLensResult:
    """Logit lens results across layers."""
    layers: list[int]
    positions: list[int]
    tokens: list[str]
    top_k_predictions: list[list[tuple[str, float]]]  # [layer][position] -> [(token, prob), ...]


@dataclass
class CausalTrace:
    """Causal importance of a component."""
    layer: int
    head: Optional[int]  # None for MLP
    position: int
    component_type: str  # "attention", "mlp", "residual"
    
    # Causal impact
    impact: float  # Change in probability when patched
    baseline_prob: float
    patched_prob: float


@dataclass
class ComponentImportance:
    """Importance ranking of model components."""
    task_type: str
    components: list[CausalTrace]
    
    # Top-k summaries
    top_heads: list[tuple[int, int, float]]  # (layer, head, impact)
    top_layers: list[tuple[int, float]]  # (layer, impact)
    critical_positions: list[tuple[int, float]]  # (position, impact)


@dataclass
class AblationResult:
    """Result of ablating a component."""
    component_type: str
    layer: int
    head: Optional[int]
    
    # Performance change
    original_accuracy: float
    ablated_accuracy: float
    accuracy_drop: float
    
    # Per-task breakdown
    task_impacts: dict = field(default_factory=dict)



@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder."""
    input_dim: int
    hidden_dim: int
    sparsity_coef: float = 1e-3
    layer: int = 6  # Which layer to train on


@dataclass
class FeatureActivation:
    """Activation of an SAE feature for a test case."""
    feature_id: int
    activation: float
    interpretation: Optional[str] = None
    
    # Correlation with outcomes
    failure_correlation: Optional[float] = None
    hallucination_correlation: Optional[float] = None


@dataclass
class FeatureAnalysis:
    """Analysis of SAE features for a suite."""
    suite_name: str
    model_id: str
    
    # Top features
    top_failure_features: list[tuple[int, float]]  # (feature_id, MI with failure)
    top_hallucination_features: list[tuple[int, float]]
    stable_features: list[int]  # Features that don't change with noise
    
    # Feature-Failure Mutual Information
    feature_failure_mi: dict = field(default_factory=dict)



@dataclass
class AttackResult:
    """Result of an adversarial attack test."""
    attack_type: str  # "prompt_injection", "jailbreak", "format_break"
    test_id: str
    attack_payload: str
    
    # Outcomes
    attack_succeeded: bool
    refusal_correct: bool  # Did model correctly refuse?
    format_broken: bool
    
    output: str
    details: dict = field(default_factory=dict)


@dataclass
class SecurityReport:
    """Security analysis report."""
    model_id: str
    timestamp: str
    
    # Attack success rates
    prompt_injection_rate: float = 0.0
    jailbreak_rate: float = 0.0
    format_break_rate: float = 0.0
    
    # Refusal accuracy
    refusal_accuracy: float = 0.0
    
    # Details
    attack_results: list[AttackResult] = field(default_factory=list)



@dataclass
class ModelInfo:
    """Information about a model."""
    id: str
    name: str
    model_type: ModelType
    
    # Capabilities
    supports_interpretability: bool = False
    supports_logprobs: bool = False
    max_context_length: int = 2048
    
    # Cost (per 1K tokens)
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0



@dataclass
class RunConfig:
    """Configuration for a test run."""
    model_id: str
    temperature: float = 0.0
    max_tokens: int = 256
    top_p: float = 1.0
    
    # For consistency tests
    num_consistency_runs: int = 5
    consistency_temperature: float = 0.7
    
    # Execution
    timeout_seconds: int = 60
    max_retries: int = 3
    
    # Interpretability
    capture_attention: bool = False
    capture_hidden_states: bool = False
    run_causal_tracing: bool = False



@dataclass
class BaselineThresholds:
    """Thresholds for CI regression detection."""
    model_id: str
    suite_name: str
    
    # Minimum acceptable values
    accuracy_min: float = 0.5
    pass_rate_min: float = 0.5
    
    # Maximum acceptable values
    latency_max_ms: float = 5000
    hallucination_rate_max: float = 0.3
    attack_success_rate_max: float = 0.1
    
    # Tolerance for regression
    accuracy_tolerance: float = 0.05  # Fail if drops more than 5%


@dataclass
class CICheckResult:
    """Result of a CI baseline check."""
    passed: bool
    model_id: str
    suite_name: str
    
    # Violations
    violations: list[str] = field(default_factory=list)
    
    # Metrics vs thresholds
    metrics: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
