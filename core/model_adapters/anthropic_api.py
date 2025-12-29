"""
Meridian Model Adapters - Anthropic API

Adapter for Anthropic models (Claude family)
"""

from typing import Any, Optional

from .base import ModelAdapter, GenerationConfig
from ..types import GenerationResult, ModelInfo, ModelType
from ..config import API_MODELS, config as app_config
from ..utils import Timer


class AnthropicAdapter(ModelAdapter):
    """
    Adapter for Anthropic API models.
    
    Supports:
    - Claude Instant
    - Claude 2.1
    - Claude 3 Sonnet/Opus/Haiku
    
    Features:
    - Automatic cost estimation
    - Long context support (up to 200K tokens)
    """
    
    def __init__(
        self,
        model_name: str = "anthropic_claude_3_sonnet",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Anthropic adapter.
        
        Args:
            model_name: Model identifier (from API_MODELS)
            api_key: Anthropic API key (or from environment)
        """
        self._model_name = model_name
        
        # Get model config
        if model_name in API_MODELS and API_MODELS[model_name]["api"] == "anthropic":
            model_config = API_MODELS[model_name]
            self._api_model = model_config["model_name"]
            self._display_name = model_config["display_name"]
            self._max_context = model_config["max_context"]
            self._cost_input = model_config["cost_input"]
            self._cost_output = model_config["cost_output"]
        else:
            # Assume direct model name
            self._api_model = model_name
            self._display_name = model_name
            self._max_context = 100000
            self._cost_input = 0.0
            self._cost_output = 0.0
        
        # Get API key
        self._api_key = api_key or app_config.api.anthropic_api_key
        
        if not self._api_key:
            raise ValueError(
                "Anthropic API key not provided. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
            )
        
        # Initialize client
        from anthropic import Anthropic
        self._client = Anthropic(api_key=self._api_key)
    
    @property
    def model_id(self) -> str:
        return self._model_name
    
    @property
    def supports_interpretability(self) -> bool:
        return False
    
    @property
    def supports_logprobs(self) -> bool:
        return False  # Anthropic doesn't expose logprobs
    
    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            id=self._model_name,
            name=self._display_name,
            model_type=ModelType.ANTHROPIC_API,
            supports_interpretability=False,
            supports_logprobs=False,
            max_context_length=self._max_context,
            cost_per_1k_input=self._cost_input,
            cost_per_1k_output=self._cost_output,
        )
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """Generate text using Anthropic API."""
        config = config or GenerationConfig()
        
        # Build request
        request_params = {
            "model": self._api_model,
            "max_tokens": config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        # Temperature (Anthropic uses 0-1 scale)
        if config.temperature > 0:
            request_params["temperature"] = min(config.temperature, 1.0)
        
        if config.top_p < 1.0:
            request_params["top_p"] = config.top_p
        
        if config.stop_sequences:
            request_params["stop_sequences"] = config.stop_sequences
        
        # Make request with timing
        with Timer() as timer:
            response = self._client.messages.create(**request_params)
        
        # Extract response
        output_text = ""
        if response.content:
            output_text = response.content[0].text
        
        # Token counts
        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        
        # Map stop reason
        finish_reason = "stop"
        if response.stop_reason == "max_tokens":
            finish_reason = "length"
        elif response.stop_reason == "stop_sequence":
            finish_reason = "stop"
        
        # Calculate cost
        cost = self.estimate_cost(tokens_in, tokens_out)
        
        return GenerationResult(
            output=output_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=timer.elapsed_ms,
            finish_reason=finish_reason,
            raw_response={
                "id": response.id,
                "model": response.model,
                "stop_reason": response.stop_reason,
            },
        )
    
    def estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        """Calculate cost based on token counts."""
        input_cost = (tokens_in / 1000) * self._cost_input
        output_cost = (tokens_out / 1000) * self._cost_output
        return input_cost + output_cost


def create_anthropic_adapter(model_name: str = "anthropic_claude_3_sonnet") -> Optional[AnthropicAdapter]:
    """
    Factory function to create Anthropic adapter if API key is available.
    
    Returns None if API key is not configured.
    """
    if not app_config.api.has_anthropic:
        return None
    
    try:
        return AnthropicAdapter(model_name)
    except ValueError:
        return None
