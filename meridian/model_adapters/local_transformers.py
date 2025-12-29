"""
Meridian Model Adapters - Local Transformers

Adapter for local HuggingFace models (GPT-2, DistilGPT, etc.)
with full interpretability support.
"""

import time
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import ModelAdapter, GenerationConfig
from ..types import GenerationResult, ModelInfo, ModelType
from ..config import LOCAL_MODELS, config as app_config
from ..utils import Timer


class LocalTransformersAdapter(ModelAdapter):
    """
    Adapter for local HuggingFace transformer models.
    
    Supports:
    - GPT-2 family (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
    - DistilGPT2
    - Other causal LM models
    
    Features:
    - Full interpretability (attention, hidden states)
    - No API key required
    - Fast inference on GPU
    """
    
    def __init__(
        self,
        model_name: str = "deepseek_chat",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
    ):
        """
        Initialize the local model adapter.
        
        Args:
            model_name: Model identifier (from LOCAL_MODELS or HF name)
            device: Device to use (cuda, cpu, mps)
            load_in_8bit: Whether to use 8-bit quantization
        """
        self._model_name = model_name
        
        # Get HuggingFace model name
        if model_name in LOCAL_MODELS:
            self._hf_name = LOCAL_MODELS[model_name]["hf_name"]
            self._display_name = LOCAL_MODELS[model_name]["display_name"]
            self._max_context = LOCAL_MODELS[model_name]["max_context"]
        else:
            # Assume it's a direct HF model name
            self._hf_name = model_name
            self._display_name = model_name
            self._max_context = 1024
        
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
        
        # Set pad token if not set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Load model with appropriate settings
        model_kwargs = {
            "output_hidden_states": True,
            "output_attentions": True,
        }
        
        if load_in_8bit and self._device == "cuda":
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16 if self._device == "cuda" else torch.float32
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self._hf_name,
            **model_kwargs
        )
        
        if not load_in_8bit:
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
            max_length=self._max_context - config.max_tokens,
        ).to(self._device)
        
        tokens_in = inputs["input_ids"].shape[1]
        
        # Generate with timing
        with Timer() as timer:
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=config.max_tokens,
                    temperature=config.temperature if config.temperature > 0 else None,
                    top_p=config.top_p if config.temperature > 0 else None,
                    do_sample=config.temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    output_hidden_states=config.return_hidden_states,
                    output_attentions=config.return_attention,
                    return_dict_in_generate=True,
                )
        
        # Decode output
        generated_ids = outputs.sequences[0][tokens_in:]
        output_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        tokens_out = len(generated_ids)
        
        # Determine finish reason
        finish_reason = "length"
        if generated_ids[-1] == self._tokenizer.eos_token_id:
            finish_reason = "stop"
        
        # Prepare result
        result = GenerationResult(
            output=output_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=timer.elapsed_ms,
            finish_reason=finish_reason,
            raw_response=None,
        )
        
        # Add interpretability data if requested
        if config.return_hidden_states and hasattr(outputs, 'hidden_states'):
            result.hidden_states = outputs.hidden_states
        
        if config.return_attention and hasattr(outputs, 'attentions'):
            result.attentions = outputs.attentions
        
        return result
    
    def get_hidden_states(
        self,
        prompt: str,
        layer: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """Get hidden states for interpretability."""
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_context,
        ).to(self._device)
        
        with torch.no_grad():
            outputs = self._model(
                **inputs,
                output_hidden_states=True,
            )
        
        hidden_states = outputs.hidden_states  # Tuple of (layer+1) tensors
        
        if layer is not None:
            return hidden_states[layer].cpu()
        
        # Stack all layers: [num_layers, batch, seq_len, hidden_dim]
        stacked = torch.stack([h.cpu() for h in hidden_states])
        return stacked
    
    def get_attention_patterns(
        self,
        prompt: str,
        layer: Optional[int] = None,
        head: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """Get attention patterns for interpretability."""
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_context,
        ).to(self._device)
        
        with torch.no_grad():
            outputs = self._model(
                **inputs,
                output_attentions=True,
            )
        
        attentions = outputs.attentions  # Tuple of (layer) tensors, each [batch, heads, seq, seq]
        
        if layer is not None:
            attn = attentions[layer].cpu()
            if head is not None:
                return attn[:, head, :, :]
            return attn
        
        # Stack all layers: [num_layers, batch, heads, seq, seq]
        stacked = torch.stack([a.cpu() for a in attentions])
        return stacked
    
    def get_logits(self, prompt: str) -> torch.Tensor:
        """Get raw logits for the prompt."""
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_context,
        ).to(self._device)
        
        with torch.no_grad():
            outputs = self._model(**inputs)
        
        return outputs.logits.cpu()
    
    def get_logit_lens(
        self,
        prompt: str,
        position: int = -1,
        top_k: int = 10
    ) -> list[list[tuple[str, float]]]:
        """
        Apply logit lens to see predictions at each layer.
        
        Returns list of (layer) -> list of (token, probability) tuples
        """
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_context,
        ).to(self._device)
        
        with torch.no_grad():
            outputs = self._model(
                **inputs,
                output_hidden_states=True,
            )
        
        hidden_states = outputs.hidden_states  # (layer+1,) tuple
        lm_head = self._model.lm_head if hasattr(self._model, 'lm_head') else self._model.transformer.wte
        
        results = []
        
        for layer_idx, hidden in enumerate(hidden_states):
            # Apply layer norm if exists
            if hasattr(self._model, 'transformer') and hasattr(self._model.transformer, 'ln_f'):
                hidden = self._model.transformer.ln_f(hidden)
            
            # Project to vocabulary
            logits = lm_head(hidden)
            
            # Get probabilities at position
            probs = torch.softmax(logits[0, position, :], dim=-1)
            
            # Get top-k
            top_probs, top_ids = torch.topk(probs, top_k)
            
            layer_preds = []
            for prob, token_id in zip(top_probs.tolist(), top_ids.tolist()):
                token = self._tokenizer.decode([token_id])
                layer_preds.append((token, prob))
            
            results.append(layer_preds)
        
        return results
    
    def forward_with_hooks(
        self,
        prompt: str,
        hook_fn: callable,
        layers: Optional[list[int]] = None
    ) -> dict:
        """
        Run forward pass with custom hooks for activation patching.
        
        Args:
            prompt: Input prompt
            hook_fn: Function to call at each layer
            layers: Specific layers to hook (None for all)
            
        Returns:
            Dict with outputs and collected hook data
        """
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_context,
        ).to(self._device)
        
        hooks = []
        hook_data = {}
        
        def make_hook(name):
            def hook(module, input, output):
                hook_data[name] = {
                    'input': input[0].detach().cpu() if isinstance(input, tuple) else input.detach().cpu(),
                    'output': output[0].detach().cpu() if isinstance(output, tuple) else output.detach().cpu(),
                }
                if hook_fn:
                    return hook_fn(name, input, output)
            return hook
        
        # Register hooks on transformer blocks
        if hasattr(self._model, 'transformer'):
            blocks = self._model.transformer.h
        else:
            blocks = self._model.model.layers
        
        target_layers = layers or list(range(len(blocks)))
        
        for i in target_layers:
            hook = blocks[i].register_forward_hook(make_hook(f"layer_{i}"))
            hooks.append(hook)
        
        try:
            with torch.no_grad():
                outputs = self._model(**inputs)
        finally:
            for hook in hooks:
                hook.remove()
        
        return {
            'outputs': outputs,
            'hook_data': hook_data,
            'inputs': inputs,
        }
    
    def tokenize(self, text: str) -> list[str]:
        """Get tokens for a text string."""
        token_ids = self._tokenizer.encode(text)
        return [self._tokenizer.decode([tid]) for tid in token_ids]
    
    def estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        """Local models have no cost."""
        return 0.0
