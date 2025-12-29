"""
Meridian Model Adapter - Groq
Ultra-fast inference
"""

import os
from typing import Optional

from .base import ModelAdapter, GenerationConfig
from ..types import GenerationResult, ModelInfo, ModelType


class GroqAdapter(ModelAdapter):
    """Adapter for Groq API (ultra-fast inference)"""
    
    def __init__(self, model_name: str = "llama2-70b-4096"):
        self._model_name = model_name
        self._model_id = f"groq_{model_name.replace('-', '_')}"
        self._client = None
        self._init_client()
    
    def _init_client(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return
        
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
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
            raise RuntimeError("Groq client not initialized. Set GROQ_API_KEY.")
        
        config = config or GenerationConfig()
        
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        
        choice = response.choices[0]
        output = choice.message.content
        
        return GenerationResult(
            output=output,
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            latency_ms=0,
            finish_reason=choice.finish_reason or "stop",
            raw_response={"model": self._model_name}
        )
    
    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self._model_id,
            model_type=ModelType.OPENAI_API,
            context_length=4096,
            supports_system_prompt=True,
            supports_functions=False,
            cost_per_1k_input=0.00027,  # Very cheap
            cost_per_1k_output=0.00027
        )
    
    def estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in + tokens_out) * 0.00027 / 1000


class GroqLlama70BAdapter(GroqAdapter):
    def __init__(self):
        super().__init__("llama2-70b-4096")


class GroqLlama8BAdapter(GroqAdapter):
    def __init__(self):
        super().__init__("llama3-8b-8192")


class GroqMixtralAdapter(GroqAdapter):
    def __init__(self):
        super().__init__("mixtral-8x7b-32768")
