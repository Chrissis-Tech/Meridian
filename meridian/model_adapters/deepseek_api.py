"""
Meridian Model Adapters - DeepSeek API

Adapter for DeepSeek models (cheap and effective).
"""

import os
from typing import Optional

from .base import ModelAdapter, GenerationConfig
from ..types import GenerationResult, ModelInfo, ModelType
from ..utils import Timer


class DeepSeekAdapter(ModelAdapter):
    """
    Adapter for DeepSeek API.
    
    DeepSeek is a cost-effective alternative to OpenAI:
    - deepseek-chat: $0.14/M input, $0.28/M output
    - Very capable for math, reasoning, instruction-following
    """
    
    def __init__(
        self,
        model_name: str = "deepseek_chat",
        api_key: Optional[str] = None,
    ):
        self._model_name = model_name
        
        # Model configs
        model_configs = {
            "deepseek_chat": {
                "api_model": "deepseek-chat",
                "display_name": "DeepSeek Chat",
                "max_context": 32768,
                "cost_input": 0.00014,  # $0.14/M = $0.00014/K
                "cost_output": 0.00028,
            },
            "deepseek_coder": {
                "api_model": "deepseek-coder",
                "display_name": "DeepSeek Coder",
                "max_context": 16384,
                "cost_input": 0.00014,
                "cost_output": 0.00028,
            },
        }
        
        config = model_configs.get(model_name, model_configs["deepseek_chat"])
        self._api_model = config["api_model"]
        self._display_name = config["display_name"]
        self._max_context = config["max_context"]
        self._cost_input = config["cost_input"]
        self._cost_output = config["cost_output"]
        
        # Get API key
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        
        if not self._api_key:
            raise ValueError(
                "DeepSeek API key not provided. "
                "Set DEEPSEEK_API_KEY environment variable."
            )
        
        # Initialize client (OpenAI-compatible)
        from openai import OpenAI
        self._client = OpenAI(
            api_key=self._api_key,
            base_url="https://api.deepseek.com"
        )
    
    @property
    def model_id(self) -> str:
        return self._model_name
    
    @property
    def supports_interpretability(self) -> bool:
        return False
    
    @property
    def supports_logprobs(self) -> bool:
        return False
    
    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            id=self._model_name,
            name=self._display_name,
            model_type=ModelType.OPENAI_API,
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
        """Generate text using DeepSeek API."""
        config = config or GenerationConfig()
        
        messages = [{"role": "user", "content": prompt}]
        
        request_params = {
            "model": self._api_model,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }
        
        if config.stop_sequences:
            request_params["stop"] = config.stop_sequences
        
        with Timer() as timer:
            response = self._client.chat.completions.create(**request_params)
        
        choice = response.choices[0]
        output_text = choice.message.content or ""
        
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        
        cost = self.estimate_cost(tokens_in, tokens_out)
        
        return GenerationResult(
            output=output_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=timer.elapsed_ms,
            finish_reason=choice.finish_reason,
            raw_response=response.model_dump(),
        )
    
    def estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        """Calculate cost based on token counts."""
        input_cost = (tokens_in / 1000) * self._cost_input
        output_cost = (tokens_out / 1000) * self._cost_output
        return input_cost + output_cost
