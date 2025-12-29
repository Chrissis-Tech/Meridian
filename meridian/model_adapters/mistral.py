"""
Meridian Model Adapter - Mistral AI
"""

import os
from typing import Optional

from .base import ModelAdapter, GenerationConfig
from ..types import GenerationResult, ModelInfo, ModelType


class MistralAdapter(ModelAdapter):
    """Adapter for Mistral AI API"""
    
    def __init__(self, model_name: str = "mistral-medium"):
        self._model_name = model_name
        self._model_id = f"mistral_{model_name.replace('-', '_')}"
        self._client = None
        self._init_client()
    
    def _init_client(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            return
        
        try:
            from mistralai.client import MistralClient
            self._client = MistralClient(api_key=api_key)
        except ImportError:
            pass
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        if not self._client:
            raise RuntimeError("Mistral client not initialized. Set MISTRAL_API_KEY.")
        
        config = config or GenerationConfig()
        
        from mistralai.models.chat_completion import ChatMessage
        
        messages = [ChatMessage(role="user", content=prompt)]
        
        response = self._client.chat(
            model=self._model_name,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        
        choice = response.choices[0]
        output = choice.message.content
        
        return GenerationResult(
            output=output,
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            latency_ms=0,  # Not provided by API
            finish_reason=choice.finish_reason or "stop",
            raw_response={"model": self._model_name}
        )
    
    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self._model_id,
            model_type=ModelType.OPENAI_API,
            context_length=32000,
            supports_system_prompt=True,
            supports_functions=False,
            cost_per_1k_input=0.0027,  # ~$2.70/M
            cost_per_1k_output=0.0081  # ~$8.10/M
        )
    
    def estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * 0.0027 + tokens_out * 0.0081) / 1000


class MistralSmallAdapter(MistralAdapter):
    def __init__(self):
        super().__init__("mistral-small")


class MistralMediumAdapter(MistralAdapter):
    def __init__(self):
        super().__init__("mistral-medium")


class MistralLargeAdapter(MistralAdapter):
    def __init__(self):
        super().__init__("mistral-large-latest")
