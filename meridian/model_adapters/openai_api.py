"""
Meridian Model Adapters - OpenAI API

Adapter for OpenAI models (GPT-3.5, GPT-4, etc.)
"""

import time
from typing import Any, Optional

from .base import ModelAdapter, GenerationConfig
from ..types import GenerationResult, ModelInfo, ModelType
from ..config import API_MODELS, config as app_config
from ..utils import Timer


class OpenAIAdapter(ModelAdapter):
    """
    Adapter for OpenAI API models.
    
    Supports:
    - GPT-3.5 Turbo
    - GPT-4
    - GPT-4 Turbo
    
    Features:
    - Automatic cost estimation
    - Log probabilities (when available)
    - Retry with exponential backoff
    """
    
    def __init__(
        self,
        model_name: str = "openai_gpt35",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the OpenAI adapter.
        
        Args:
            model_name: Model identifier (from API_MODELS)
            api_key: OpenAI API key (or from environment)
        """
        self._model_name = model_name
        
        # Get model config
        if model_name in API_MODELS and API_MODELS[model_name]["api"] == "openai":
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
            self._max_context = 4096
            self._cost_input = 0.0
            self._cost_output = 0.0
        
        # Get API key
        self._api_key = api_key or app_config.api.openai_api_key
        
        if not self._api_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        # Initialize client
        from openai import OpenAI
        self._client = OpenAI(api_key=self._api_key)
    
    @property
    def model_id(self) -> str:
        return self._model_name
    
    @property
    def supports_interpretability(self) -> bool:
        return False
    
    @property
    def supports_logprobs(self) -> bool:
        return True  # GPT-3.5/4 support logprobs
    
    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            id=self._model_name,
            name=self._display_name,
            model_type=ModelType.OPENAI_API,
            supports_interpretability=False,
            supports_logprobs=True,
            max_context_length=self._max_context,
            cost_per_1k_input=self._cost_input,
            cost_per_1k_output=self._cost_output,
        )
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """Generate text using OpenAI API."""
        config = config or GenerationConfig()
        
        # Build request
        messages = [{"role": "user", "content": prompt}]
        
        request_params = {
            "model": self._api_model,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        
        if config.stop_sequences:
            request_params["stop"] = config.stop_sequences
        
        if config.return_logprobs:
            request_params["logprobs"] = True
            request_params["top_logprobs"] = 5
        
        # Make request with timing
        with Timer() as timer:
            response = self._client.chat.completions.create(**request_params)
        
        # Extract response
        choice = response.choices[0]
        output_text = choice.message.content or ""
        
        # Token counts
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        
        # Log probabilities
        logprobs = None
        if config.return_logprobs and choice.logprobs:
            logprobs = [
                lp.logprob 
                for lp in choice.logprobs.content
            ]
        
        # Calculate cost
        cost = self.estimate_cost(tokens_in, tokens_out)
        
        return GenerationResult(
            output=output_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=timer.elapsed_ms,
            finish_reason=choice.finish_reason,
            raw_response=response.model_dump(),
            logprobs=logprobs,
        )
    
    def estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        """Calculate cost based on token counts."""
        input_cost = (tokens_in / 1000) * self._cost_input
        output_cost = (tokens_out / 1000) * self._cost_output
        return input_cost + output_cost


def create_openai_adapter(model_name: str = "openai_gpt35") -> Optional[OpenAIAdapter]:
    """
    Factory function to create OpenAI adapter if API key is available.
    
    Returns None if API key is not configured.
    """
    if not app_config.api.has_openai:
        return None
    
    try:
        return OpenAIAdapter(model_name)
    except ValueError:
        return None
