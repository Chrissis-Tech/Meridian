"""
Meridian Model Adapter - Mistral AI

Uses direct HTTP API to avoid SDK version issues.
"""

import os
import time
import requests
from typing import Optional

from .base import ModelAdapter, GenerationConfig
from ..types import GenerationResult, ModelInfo, ModelType


class MistralAdapter(ModelAdapter):
    """Adapter for Mistral AI API using direct HTTP calls."""
    
    API_BASE = "https://api.mistral.ai/v1"
    
    def __init__(self, model_name: str = "mistral-medium"):
        self._model_name = model_name
        self._model_id = f"mistral_{model_name.replace('-', '_')}"
        self._api_key = os.getenv("MISTRAL_API_KEY")
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        if not self._api_key:
            return GenerationResult(
                output="",
                tokens_in=0,
                tokens_out=0,
                latency_ms=0,
                finish_reason="error",
                error="Mistral API key not set. Set MISTRAL_API_KEY.",
                raw_response={}
            )
        
        config = config or GenerationConfig()
        
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        
        start = time.perf_counter()
        
        try:
            response = requests.post(
                f"{self.API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            latency_ms = (time.perf_counter() - start) * 1000
            
            if response.status_code != 200:
                return GenerationResult(
                    output="",
                    tokens_in=0,
                    tokens_out=0,
                    latency_ms=latency_ms,
                    finish_reason="error",
                    error=f"Mistral API error {response.status_code}: {response.text[:200]}",
                    raw_response={"status": response.status_code}
                )
            
            data = response.json()
            
            choice = data["choices"][0]
            output = choice["message"]["content"]
            usage = data.get("usage", {})
            
            return GenerationResult(
                output=output,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
                latency_ms=latency_ms,
                finish_reason=choice.get("finish_reason", "stop"),
                raw_response={"model": self._model_name}
            )
            
        except requests.Timeout:
            return GenerationResult(
                output="",
                tokens_in=0,
                tokens_out=0,
                latency_ms=60000,
                finish_reason="error",
                error="Mistral API timeout",
                raw_response={}
            )
        except Exception as e:
            return GenerationResult(
                output="",
                tokens_in=0,
                tokens_out=0,
                latency_ms=0,
                finish_reason="error",
                error=f"Mistral error: {str(e)}",
                raw_response={}
            )
    
    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self._model_id,
            model_type=ModelType.OPENAI_API,
            context_length=32000,
            supports_system_prompt=True,
            supports_functions=False,
            cost_per_1k_input=0.0027,
            cost_per_1k_output=0.0081
        )
    
    def estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * 0.0027 + tokens_out * 0.0081) / 1000


class MistralSmallAdapter(MistralAdapter):
    def __init__(self):
        super().__init__("mistral-small-latest")


class MistralMediumAdapter(MistralAdapter):
    def __init__(self):
        super().__init__("mistral-medium-latest")


class MistralLargeAdapter(MistralAdapter):
    def __init__(self):
        super().__init__("mistral-large-latest")

