"""
Basic tests for Meridian core functionality.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestScoring:
    """Test scoring functions."""
    
    def test_exact_match(self):
        from core.scoring import exact_match
        
        result = exact_match("hello", "hello")
        assert result.passed == True
        assert result.score == 1.0
        
        result = exact_match("hello", "world")
        assert result.passed == False
        assert result.score == 0.0
    
    def test_exact_match_case_insensitive(self):
        from core.scoring import exact_match
        
        result = exact_match("HELLO", "hello", case_sensitive=False)
        assert result.passed == True
    
    def test_contains_match(self):
        from core.scoring import contains_match
        
        result = contains_match("The answer is 42", "42")
        assert result.passed == True
        
        result = contains_match("No number here", "42")
        assert result.passed == False
    
    def test_numeric_match(self):
        from core.scoring import numeric_match
        
        result = numeric_match("The answer is 42", 42)
        assert result.passed == True
        
        result = numeric_match("Approximately 3.14", 3.14, tolerance=0.01)
        assert result.passed == True
    
    def test_regex_match(self):
        from core.scoring import regex_match
        
        result = regex_match("Email: test@example.com", r"\w+@\w+\.\w+")
        assert result.passed == True
        
        result = regex_match("No email here", r"\w+@\w+\.\w+")
        assert result.passed == False
    
    def test_json_validation(self):
        from core.scoring import validate_json
        
        result = validate_json('{"name": "John", "age": 30}')
        assert result.passed == True
        
        result = validate_json('not json')
        assert result.passed == False
    
    def test_length_constraints(self):
        from core.scoring import check_length_constraints
        
        result = check_length_constraints("one two three", max_words=5)
        assert result.passed == True
        
        result = check_length_constraints("one two three four five six", max_words=5)
        assert result.passed == False


class TestUtils:
    """Test utility functions."""
    
    def test_hash_prompt(self):
        from core.utils import hash_prompt
        
        h1 = hash_prompt("test prompt")
        h2 = hash_prompt("test prompt")
        h3 = hash_prompt("different prompt")
        
        assert h1 == h2
        assert h1 != h3
    
    def test_normalize_text(self):
        from core.utils import normalize_text
        
        assert normalize_text("  Hello   World  ") == "hello world"
    
    def test_count_words(self):
        from core.utils import count_words
        
        assert count_words("one two three") == 3
        assert count_words("") == 0
    
    def test_count_sentences(self):
        from core.utils import count_sentences
        
        assert count_sentences("Hello. World.") == 2
        assert count_sentences("One sentence") == 1
    
    def test_extract_json(self):
        from core.utils import extract_json
        
        data = extract_json('{"key": "value"}')
        assert data == {"key": "value"}
        
        data = extract_json('Some text {"key": "value"} more text')
        assert data == {"key": "value"}
    
    def test_extract_number(self):
        from core.utils import extract_number
        
        assert extract_number("The answer is 42") == 42
        assert extract_number("Pi is 3.14159") == 3.14159
        assert extract_number("No numbers") is None


class TestConfig:
    """Test configuration."""
    
    def test_config_loads(self):
        from core.config import config
        
        assert config is not None
        assert hasattr(config, 'model')
        assert hasattr(config, 'api')
    
    def test_available_models(self):
        from core.config import get_available_models
        
        models = get_available_models()
        assert "deepseek_chat" in models
        assert models["deepseek_chat"]["available"] == True


class TestTypes:
    """Test type definitions."""
    
    def test_run_result(self):
        from core.types import RunResult
        
        result = RunResult(
            test_id="test-001",
            model_id="gpt2",
            output="Hello",
            tokens_in=10,
            tokens_out=5,
            latency_ms=100,
            cost_estimate=0.0,
            timestamp="2025-01-01T00:00:00",
            run_hash="abc123",
        )
        
        assert result.test_id == "test-001"
        assert result.passed == False  # Default
    
    def test_scoring_result(self):
        from core.types import ScoringResult
        
        result = ScoringResult(
            passed=True,
            score=1.0,
            method="exact_match",
            details={"key": "value"},
        )
        
        assert result.passed == True
        assert result.score == 1.0


class TestStatistics:
    """Test statistical functions."""
    
    def test_bootstrap_ci(self):
        from core.stats import bootstrap_ci
        
        values = [0.8, 0.85, 0.9, 0.75, 0.82]
        mean, (lower, upper) = bootstrap_ci(values)
        
        assert 0.7 < mean < 0.95
        assert lower < mean < upper
    
    def test_cohens_d(self):
        from core.stats import cohens_d
        
        a = [1, 2, 3, 4, 5]
        b = [6, 7, 8, 9, 10]
        
        d = cohens_d(a, b)
        assert d > 2.0  # Large effect size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
