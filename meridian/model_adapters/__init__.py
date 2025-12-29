"""
Meridian Model Adapters Package
"""

from .base import ModelAdapter, GenerationConfig
from .local_transformers import LocalTransformersAdapter
from .seq2seq import Seq2SeqAdapter
from .openai_api import OpenAIAdapter, create_openai_adapter
from .anthropic_api import AnthropicAdapter, create_anthropic_adapter
from .deepseek_api import DeepSeekAdapter

# New providers
from .mistral import MistralSmallAdapter, MistralMediumAdapter, MistralLargeAdapter
from .groq_adapter import GroqLlama70BAdapter, GroqLlama8BAdapter, GroqMixtralAdapter
from .together import TogetherLlama70BAdapter, TogetherMixtralAdapter, TogetherCodeLlamaAdapter

from typing import Optional
import os
from ..config import get_available_models, LOCAL_MODELS


# Registry of all adapters
AVAILABLE_ADAPTERS = {
    # Local
    "local_distilgpt2": LocalTransformersAdapter,
    "local_flan_t5_small": Seq2SeqAdapter,
    # DeepSeek
    "deepseek_chat": DeepSeekAdapter,
    "deepseek_coder": DeepSeekAdapter,
    # OpenAI
    "openai_gpt35": OpenAIAdapter,
    "openai_gpt4": OpenAIAdapter,
    "openai_gpt4_turbo": OpenAIAdapter,
    # Anthropic
    "anthropic_claude_2": AnthropicAdapter,
    # Mistral
    "mistral_small": MistralSmallAdapter,
    "mistral_medium": MistralMediumAdapter,
    "mistral_large": MistralLargeAdapter,
    # Groq
    "groq_llama70b": GroqLlama70BAdapter,
    "groq_llama8b": GroqLlama8BAdapter,
    "groq_mixtral": GroqMixtralAdapter,
    # Together
    "together_llama70b": TogetherLlama70BAdapter,
    "together_mixtral": TogetherMixtralAdapter,
    "together_codellama": TogetherCodeLlamaAdapter,
}


def get_adapter(model_id: str) -> ModelAdapter:
    """
    Factory function to get the appropriate adapter for a model.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Configured ModelAdapter instance
        
    Raises:
        ValueError: If model is not available
    """
    # DeepSeek
    if model_id.startswith("deepseek_"):
        if os.getenv("DEEPSEEK_API_KEY"):
            return DeepSeekAdapter(model_id)
        else:
            raise ValueError("DeepSeek requires DEEPSEEK_API_KEY environment variable")
    
    # Mistral
    if model_id.startswith("mistral_"):
        if not os.getenv("MISTRAL_API_KEY"):
            raise ValueError("Mistral requires MISTRAL_API_KEY environment variable")
        if model_id == "mistral_small":
            return MistralSmallAdapter()
        elif model_id == "mistral_medium":
            return MistralMediumAdapter()
        elif model_id == "mistral_large":
            return MistralLargeAdapter()
    
    # Groq
    if model_id.startswith("groq_"):
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("Groq requires GROQ_API_KEY environment variable")
        if model_id == "groq_llama70b":
            return GroqLlama70BAdapter()
        elif model_id == "groq_llama8b":
            return GroqLlama8BAdapter()
        elif model_id == "groq_mixtral":
            return GroqMixtralAdapter()
    
    # Together
    if model_id.startswith("together_"):
        if not os.getenv("TOGETHER_API_KEY"):
            raise ValueError("Together requires TOGETHER_API_KEY environment variable")
        if model_id == "together_llama70b":
            return TogetherLlama70BAdapter()
        elif model_id == "together_mixtral":
            return TogetherMixtralAdapter()
        elif model_id == "together_codellama":
            return TogetherCodeLlamaAdapter()
    
    # Check config-based models
    available = get_available_models()
    
    if model_id not in available:
        all_models = list(AVAILABLE_ADAPTERS.keys())
        raise ValueError(f"Unknown model: {model_id}. Available: {all_models}")
    
    model_info = available[model_id]
    
    if not model_info.get("available", False):
        raise ValueError(
            f"Model {model_id} requires API key which is not configured. "
            f"Set the appropriate environment variable."
        )
    
    if model_info["type"] == "local":
        local_config = LOCAL_MODELS.get(model_id, {})
        if local_config.get("type") == "seq2seq":
            return Seq2SeqAdapter(model_id)
        return LocalTransformersAdapter(model_id)
    elif model_info.get("api") == "openai":
        return OpenAIAdapter(model_id)
    elif model_info.get("api") == "anthropic":
        return AnthropicAdapter(model_id)
    else:
        raise ValueError(f"Unknown model type for {model_id}")


def list_available_models() -> list[str]:
    """Get list of available model IDs."""
    available = get_available_models()
    models = [k for k, v in available.items() if v.get("available", False)]
    
    # Add API-based models if keys are available
    if os.getenv("DEEPSEEK_API_KEY"):
        models.extend(["deepseek_chat", "deepseek_coder"])
    if os.getenv("MISTRAL_API_KEY"):
        models.extend(["mistral_small", "mistral_medium", "mistral_large"])
    if os.getenv("GROQ_API_KEY"):
        models.extend(["groq_llama70b", "groq_llama8b", "groq_mixtral"])
    if os.getenv("TOGETHER_API_KEY"):
        models.extend(["together_llama70b", "together_mixtral", "together_codellama"])
    
    return models


__all__ = [
    # Base
    "ModelAdapter",
    "GenerationConfig",
    # Implementations
    "LocalTransformersAdapter",
    "Seq2SeqAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "DeepSeekAdapter",
    "MistralSmallAdapter",
    "MistralMediumAdapter",
    "MistralLargeAdapter",
    "GroqLlama70BAdapter",
    "GroqLlama8BAdapter",
    "GroqMixtralAdapter",
    "TogetherLlama70BAdapter",
    "TogetherMixtralAdapter",
    "TogetherCodeLlamaAdapter",
    # Registry
    "AVAILABLE_ADAPTERS",
    # Factory
    "get_adapter",
    "list_available_models",
    "create_openai_adapter",
    "create_anthropic_adapter",
]
