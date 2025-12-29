"""
Meridian Model Adapters - Seq2Seq (Flan-T5)

Adapter for instruction-tuned seq2seq models.
"""

import time
from typing import Any, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .base import ModelAdapter, GenerationConfig
from ..types import GenerationResult, ModelInfo, ModelType
from ..config import LOCAL_MODELS, config as app_config
from ..utils import Timer


class Seq2SeqAdapter(ModelAdapter):
    """
    Adapter for seq2seq models like Flan-T5.
    
    These models are instruction-tuned and work well for:
    - Question answering
    - Math problems
    - Following instructions
    
    Unlike GPT-2, they don't just complete text - they generate answers.
    """
    
    def __init__(
        self,
        model_name: str = "local_flan_t5_base",
        device: Optional[str] = None,
    ):
        self._model_name = model_name
        
        # Get HuggingFace model name
        if model_name in LOCAL_MODELS:
            self._hf_name = LOCAL_MODELS[model_name]["hf_name"]
            self._display_name = LOCAL_MODELS[model_name]["display_name"]
            self._max_context = LOCAL_MODELS[model_name]["max_context"]
        else:
            self._hf_name = model_name
            self._display_name = model_name
            self._max_context = 512
        
        # Determine device
        if device:
            self._device = device
        elif torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        
        # Load model and tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self._hf_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._hf_name)
        self._model = self._model.to(self._device)
        self._model.eval()
    
    @property
    def model_id(self) -> str:
        return self._model_name
    
    @property
    def supports_interpretability(self) -> bool:
        return True
    
    @property
    def supports_logprobs(self) -> bool:
        return True
    
    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            id=self._model_name,
            name=self._display_name,
            model_type=ModelType.LOCAL_TRANSFORMERS,
            supports_interpretability=True,
            supports_logprobs=True,
            max_context_length=self._max_context,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
        )
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """Generate text with full metadata."""
        config = config or GenerationConfig()
        
        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_context,
        ).to(self._device)
        
        tokens_in = inputs["input_ids"].shape[1]
        
        # Generate with timing
        with Timer() as timer:
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=config.max_tokens,
                    temperature=config.temperature if config.temperature > 0 else 1.0,
                    do_sample=config.temperature > 0,
                    num_beams=1 if config.temperature > 0 else 4,
                )
        
        # Decode output
        output_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_out = outputs.shape[1]
        
        return GenerationResult(
            output=output_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=timer.elapsed_ms,
            finish_reason="stop",
            raw_response=None,
        )
    
    def get_hidden_states(
        self,
        prompt: str,
        layer: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """Get encoder hidden states."""
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_context,
        ).to(self._device)
        
        with torch.no_grad():
            outputs = self._model.encoder(
                **inputs,
                output_hidden_states=True,
            )
        
        hidden_states = outputs.hidden_states
        
        if layer is not None:
            return hidden_states[layer].cpu()
        
        stacked = torch.stack([h.cpu() for h in hidden_states])
        return stacked
    
    def get_attention_patterns(
        self,
        prompt: str,
        layer: Optional[int] = None,
        head: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """Get attention patterns."""
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_context,
        ).to(self._device)
        
        with torch.no_grad():
            outputs = self._model.encoder(
                **inputs,
                output_attentions=True,
            )
        
        attentions = outputs.attentions
        
        if layer is not None:
            attn = attentions[layer].cpu()
            if head is not None:
                return attn[:, head, :, :]
            return attn
        
        stacked = torch.stack([a.cpu() for a in attentions])
        return stacked
    
    def tokenize(self, text: str) -> list[str]:
        """Get tokens for a text string."""
        token_ids = self._tokenizer.encode(text)
        return [self._tokenizer.decode([tid]) for tid in token_ids]
    
    def estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        """Local models have no cost."""
        return 0.0
