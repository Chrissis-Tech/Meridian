"""
Meridian Model Adapters - Base Class

Abstract base class for all model adapters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from ..types import GenerationResult, ModelInfo, ModelType


@dataclass
class GenerationConfig:
    """Configuration for a generation request."""
    temperature: float = 0.0
    max_tokens: int = 256
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    
    # Interpretability options (for local models)
    return_hidden_states: bool = False
    return_attention: bool = False
    return_logprobs: bool = False


class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    
    All model adapters must implement this interface to ensure
    consistent behavior across different model backends.
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The input prompt
            config: Generation configuration
            
        Returns:
            GenerationResult with output, tokens, latency, etc.
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the model.
        
        Returns:
            ModelInfo with capabilities and metadata
        """
        pass
    
    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier for this model."""
        pass
    
    @property
    def supports_interpretability(self) -> bool:
        """Whether this model supports interpretability features."""
        return False
    
    @property
    def supports_logprobs(self) -> bool:
        """Whether this model returns log probabilities."""
        return False
    
    @property
    def supports_batching(self) -> bool:
        """Whether this model supports batched requests."""
        return False
    
    def batch_generate(
        self,
        prompts: list[str],
        config: Optional[GenerationConfig] = None
    ) -> list[GenerationResult]:
        """
        Generate responses for multiple prompts.
        
        Default implementation calls generate() sequentially.
        Override for models that support native batching.
        """
        return [self.generate(prompt, config) for prompt in prompts]
    
    def get_hidden_states(
        self,
        prompt: str,
        layer: Optional[int] = None
    ) -> Optional[Any]:
        """
        Get hidden states for a prompt.
        
        Only available for local models with interpretability.
        
        Args:
            prompt: Input prompt
            layer: Specific layer (None for all)
            
        Returns:
            Hidden states tensor or None if not supported
        """
        return None
    
    def get_attention_patterns(
        self,
        prompt: str,
        layer: Optional[int] = None,
        head: Optional[int] = None
    ) -> Optional[Any]:
        """
        Get attention patterns for a prompt.
        
        Only available for local models with interpretability.
        
        Args:
            prompt: Input prompt
            layer: Specific layer (None for all)
            head: Specific head (None for all)
            
        Returns:
            Attention tensor or None if not supported
        """
        return None
    
    def get_logits(self, prompt: str) -> Optional[Any]:
        """
        Get raw logits for a prompt.
        
        Only available for local models.
        
        Returns:
            Logits tensor or None if not supported
        """
        return None
    
    def estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        """
        Estimate the cost of a generation.
        
        Returns 0.0 for local models.
        """
        return 0.0
    
    def health_check(self) -> bool:
        """
        Check if the model is available and working.
        
        Returns True if the model can generate responses.
        """
        try:
            result = self.generate("Hello", GenerationConfig(max_tokens=5))
            return result is not None and result.output != ""
        except Exception:
            return False
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id={self.model_id})"
