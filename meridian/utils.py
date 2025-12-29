"""
Meridian Utility Functions

Common utilities for hashing, timing, token counting, and data manipulation.
"""

import hashlib
import json
import re
import time
import uuid
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Optional

import numpy as np



def generate_run_id() -> str:
    """Generate a unique run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:8]
    return f"run_{timestamp}_{unique}"


def hash_prompt(prompt: str) -> str:
    """Generate a hash for a prompt (for versioning)."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def hash_config(config: dict) -> str:
    """Generate a hash for a configuration dict."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def hash_result(result: dict) -> str:
    """Generate a hash for a result (for deduplication)."""
    result_str = json.dumps(result, sort_keys=True, default=str)
    return hashlib.sha256(result_str.encode()).hexdigest()[:16]



class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000


def timed(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"{func.__name__} took {elapsed:.2f}ms")
        return result
    return wrapper



def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.
    Uses a simple heuristic: ~4 characters per token for English.
    For accurate counts, use the model's tokenizer.
    """
    # Rough estimate: 4 chars per token
    return max(1, len(text) // 4)


def count_words(text: str) -> int:
    """Count words in a text string."""
    return len(text.split())


def count_sentences(text: str) -> int:
    """Count sentences in a text string."""
    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])



def estimate_cost(
    tokens_in: int,
    tokens_out: int,
    cost_per_1k_input: float,
    cost_per_1k_output: float
) -> float:
    """Estimate cost based on tokens and pricing."""
    input_cost = (tokens_in / 1000) * cost_per_1k_input
    output_cost = (tokens_out / 1000) * cost_per_1k_output
    return input_cost + output_cost



def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove punctuation (optional - controlled by config)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def extract_json(text: str) -> Optional[dict]:
    """Extract JSON from a text string."""
    # Try to find JSON in the text
    # Look for {...} or [...]
    json_patterns = [
        r'\{[^{}]*\}',  # Simple object
        r'\{.*\}',       # Nested object (greedy)
        r'\[.*\]',       # Array
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # Try parsing the entire text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_number(text: str) -> Optional[float]:
    """Extract a number from text."""
    # Look for numbers (including decimals and negatives)
    matches = re.findall(r'-?\d+\.?\d*', text)
    if matches:
        try:
            return float(matches[-1])  # Return last number found
        except ValueError:
            return None
    return None


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix



def entropy(probs: list[float]) -> float:
    """Calculate Shannon entropy of a probability distribution."""
    probs = np.array(probs)
    probs = probs[probs > 0]  # Remove zeros to avoid log(0)
    return -np.sum(probs * np.log2(probs))


def softmax(logits: list[float], temperature: float = 1.0) -> np.ndarray:
    """Apply softmax with temperature."""
    logits = np.array(logits) / temperature
    exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return exp_logits / np.sum(exp_logits)


def mean_with_ci(values: list[float], confidence: float = 0.95) -> tuple[float, tuple[float, float]]:
    """
    Calculate mean with confidence interval using bootstrap.
    Returns (mean, (lower, upper))
    """
    from scipy import stats
    
    values = np.array(values)
    mean = np.mean(values)
    
    if len(values) < 2:
        return mean, (mean, mean)
    
    # Bootstrap confidence interval
    n_bootstrap = 1000
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    
    return mean, (lower, upper)



def ensure_dir(path: str) -> None:
    """Ensure a directory exists."""
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely load JSON with a default value on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file."""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict], path: str) -> None:
    """Save items to a JSONL file."""
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item) + '\n')



def now_iso() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def format_duration(ms: float) -> str:
    """Format duration in a human-readable way."""
    if ms < 1000:
        return f"{ms:.1f}ms"
    elif ms < 60000:
        return f"{ms/1000:.2f}s"
    else:
        minutes = int(ms // 60000)
        seconds = (ms % 60000) / 1000
        return f"{minutes}m {seconds:.1f}s"



def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a value as a percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_delta(value: float, decimals: int = 1, show_sign: bool = True) -> str:
    """Format a delta value with sign."""
    formatted = f"{value * 100:.{decimals}f}%"
    if show_sign and value > 0:
        formatted = "+" + formatted
    return formatted


def format_ci(mean: float, ci: tuple[float, float], decimals: int = 1) -> str:
    """Format mean with confidence interval."""
    return f"{mean*100:.{decimals}f}% [{ci[0]*100:.{decimals}f}%, {ci[1]*100:.{decimals}f}%]"
