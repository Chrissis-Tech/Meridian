"""
Meridian Configuration Management

Handles loading configuration from environment variables and config files.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()



# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", DATA_DIR / "results"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", PROJECT_ROOT / "reports" / "output"))
SUITES_DIR = PROJECT_ROOT / "suites"
BASELINES_DIR = PROJECT_ROOT / "baselines"

# Database
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", DATA_DIR / "Meridian.db"))

# Ensure directories exist
for dir_path in [DATA_DIR, RESULTS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)



@dataclass
class APIConfig:
    """API configuration for external model providers."""
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    
    # Rate limiting
    max_concurrent_requests: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
    )
    request_timeout: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "60"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES", "3"))
    )
    
    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)
    
    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)



@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""
    default_model: str = field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "deepseek_chat")
    )
    device: str = field(
        default_factory=lambda: os.getenv("DEVICE", "cpu")
    )
    hf_home: Optional[str] = field(
        default_factory=lambda: os.getenv("HF_HOME")
    )
    
    # Default generation parameters
    default_temperature: float = 0.0
    default_max_tokens: int = 256
    default_top_p: float = 1.0


# Available local models
LOCAL_MODELS = {
    # Instruction-following models (recommended)
    "local_flan_t5_small": {
        "hf_name": "google/flan-t5-small",
        "display_name": "Flan-T5 Small (80M) - Instruction-tuned",
        "max_context": 512,
        "type": "seq2seq",
    },
    "local_flan_t5_base": {
        "hf_name": "google/flan-t5-base",
        "display_name": "Flan-T5 Base (250M) - Instruction-tuned",
        "max_context": 512,
        "type": "seq2seq",
    },
    "local_flan_t5_large": {
        "hf_name": "google/flan-t5-large",
        "display_name": "Flan-T5 Large (780M) - Instruction-tuned",
        "max_context": 512,
        "type": "seq2seq",
    },
    # Completion models (for interpretability demos only)
    "deepseek_chat": {
        "hf_name": "gpt2",
        "display_name": "DeepSeek Chat - Completion only",
        "max_context": 1024,
        "type": "causal",
    },
    "deepseek_chat_medium": {
        "hf_name": "gpt2-medium",
        "display_name": "GPT-2 Medium (355M) - Completion only",
        "max_context": 1024,
        "type": "causal",
    },
    "local_distilgpt2": {
        "hf_name": "distilgpt2",
        "display_name": "DistilGPT-2 (82M) - Completion only",
        "max_context": 1024,
        "type": "causal",
    },
}

# Available API models
API_MODELS = {
    "openai_gpt35": {
        "api": "openai",
        "model_name": "gpt-3.5-turbo",
        "display_name": "GPT-3.5 Turbo",
        "max_context": 16385,
        "cost_input": 0.0005,  # per 1K tokens
        "cost_output": 0.0015,
    },
    "openai_gpt4": {
        "api": "openai",
        "model_name": "gpt-4",
        "display_name": "GPT-4",
        "max_context": 8192,
        "cost_input": 0.03,
        "cost_output": 0.06,
    },
    "openai_gpt4_turbo": {
        "api": "openai",
        "model_name": "gpt-4-turbo-preview",
        "display_name": "GPT-4 Turbo",
        "max_context": 128000,
        "cost_input": 0.01,
        "cost_output": 0.03,
    },
    "anthropic_claude_instant": {
        "api": "anthropic",
        "model_name": "claude-instant-1.2",
        "display_name": "Claude Instant",
        "max_context": 100000,
        "cost_input": 0.0008,
        "cost_output": 0.0024,
    },
    "anthropic_claude_2": {
        "api": "anthropic",
        "model_name": "claude-2.1",
        "display_name": "Claude 2.1",
        "max_context": 200000,
        "cost_input": 0.008,
        "cost_output": 0.024,
    },
    "anthropic_claude_3_sonnet": {
        "api": "anthropic",
        "model_name": "claude-3-sonnet-20250229",
        "display_name": "Claude 3 Sonnet",
        "max_context": 200000,
        "cost_input": 0.003,
        "cost_output": 0.015,
    },
}



@dataclass
class ScoringConfig:
    """Configuration for scoring and evaluation."""
    
    # Self-consistency
    consistency_runs: int = 5
    consistency_temperature: float = 0.7
    
    # Hallucination detection
    hallucination_keywords: list = field(default_factory=lambda: [
        "studies show", "research indicates", "according to",
        "www.", "http://", "https://",
        "ISBN", "DOI", "et al.",
    ])
    
    # Length constraints
    default_max_words: int = 500
    default_max_sentences: int = 10
    
    # JSON validation
    strict_json: bool = True



@dataclass
class CIConfig:
    """Configuration for CI/CD integration."""
    accuracy_threshold: float = field(
        default_factory=lambda: float(os.getenv("CI_ACCURACY_THRESHOLD", "0.5"))
    )
    latency_max_ms: float = field(
        default_factory=lambda: float(os.getenv("CI_LATENCY_MAX_MS", "5000"))
    )
    hallucination_max_rate: float = field(
        default_factory=lambda: float(os.getenv("CI_HALLUCINATION_MAX_RATE", "0.3"))
    )
    
    # Regression tolerance
    accuracy_tolerance: float = 0.05  # 5% drop allowed



@dataclass
class UIConfig:
    """Configuration for Streamlit UI."""
    port: int = field(
        default_factory=lambda: int(os.getenv("STREAMLIT_PORT", "8501"))
    )
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    
    # Display settings
    results_per_page: int = 20
    max_output_preview: int = 500  # characters



@dataclass
class Config:
    """Global configuration container."""
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    ci: CIConfig = field(default_factory=CIConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    results_dir: Path = RESULTS_DIR
    reports_dir: Path = REPORTS_DIR
    suites_dir: Path = SUITES_DIR
    baselines_dir: Path = BASELINES_DIR
    database_path: Path = DATABASE_PATH


# Singleton config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def get_available_models() -> dict:
    """Get all available models (local + API if keys present)."""
    models = {}
    
    # Always include local models
    for model_id, info in LOCAL_MODELS.items():
        models[model_id] = {
            **info,
            "available": True,
            "type": "local",
        }
    
    # Add API models based on available keys
    api_config = config.api
    for model_id, info in API_MODELS.items():
        if info["api"] == "openai" and api_config.has_openai:
            models[model_id] = {**info, "available": True, "type": "api"}
        elif info["api"] == "anthropic" and api_config.has_anthropic:
            models[model_id] = {**info, "available": True, "type": "api"}
        else:
            models[model_id] = {**info, "available": False, "type": "api"}
    
    return models
