"""
Meridian Model Adapter - Together AI
Open models via API
"""

import os
from typing import Optional

from .base import ModelAdapter, GenerationConfig
from ..types import GenerationResult, ModelInfo, ModelType


class TogetherAdapter(ModelAdapter):
    """Adapter for Together AI API"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-70b-chat-hf"):
        self._model_name = model_name
        # Create clean model_id
        clean_name = model_name.split("/")[-1].replace("-", "_").lower()
        self._model_id = f"together_{clean_name}"
        self._client = None
        self._init_client()
    
    def _init_client(self):
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            return
        
        try:
            import together
            together.api_key = api_key
            self._client = together
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
            raise RuntimeError("Together client not initialized. Set TOGETHER_API_KEY.")
        
        config = config or GenerationConfig()
        
        response = self._client.Complete.create(
            model=self._model_name,
            prompt=prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        
        output = response["output"]["choices"][0]["text"]
        usage = response.get("usage", {})
        
        return GenerationResult(
            output=output,
            tokens_in=usage.get("prompt_tokens", len(prompt) // 4),
            tokens_out=usage.get("completion_tokens", len(output) // 4),
            latency_ms=0,
            finish_reason="stop",
            raw_response={"model": self._model_name}
        )
    
    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self._model_id,
            model_type=ModelType.OPENAI_API,
            context_length=4096,
            supports_system_prompt=True,
            supports_functions=False,
            cost_per_1k_input=0.0009,
            cost_per_1k_output=0.0009
        )
    
    def estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in + tokens_out) * 0.0009 / 1000


class TogetherLlama70BAdapter(TogetherAdapter):
    def __init__(self):
        super().__init__("meta-llama/Llama-2-70b-chat-hf")


class TogetherMixtralAdapter(TogetherAdapter):
    def __init__(self):
        super().__init__("mistralai/Mixtral-8x7B-Instruct-v0.1")


class TogetherCodeLlamaAdapter(TogetherAdapter):
    def __init__(self):
        super().__init__("codellama/CodeLlama-34b-Instruct-hf")
