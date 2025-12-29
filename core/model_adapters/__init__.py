"""
DEPRECATED: Use 'from meridian.model_adapters import ...' instead.
This module will be removed in v1.0.0.
"""
from meridian.model_adapters import *
from meridian.model_adapters import (
    ModelAdapter,
    GenerationConfig,
    get_adapter,
    list_available_models,
    AVAILABLE_ADAPTERS,
    LocalTransformersAdapter,
    Seq2SeqAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    DeepSeekAdapter,
    MistralSmallAdapter,
    MistralMediumAdapter,
    MistralLargeAdapter,
    GroqLlama70BAdapter,
    GroqLlama8BAdapter,
    GroqMixtralAdapter,
    TogetherLlama70BAdapter,
    TogetherMixtralAdapter,
    TogetherCodeLlamaAdapter,
)
