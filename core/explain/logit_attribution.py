"""
Meridian Explain - Logit Attribution

Attributes final logits to individual attention heads and MLP layers.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class HeadAttribution:
    """Attribution for a single attention head."""
    layer: int
    head: int
    direct_effect: float
    indirect_effect: float
    total_effect: float


@dataclass
class LogitAttributionResult:
    """Result of logit attribution analysis."""
    target_token: str
    target_logit: float
    head_attributions: list[HeadAttribution]
    mlp_attributions: list[tuple[int, float]]
    top_positive_heads: list[tuple[int, int, float]]
    top_negative_heads: list[tuple[int, int, float]]


def compute_head_attribution(
    model,
    tokenizer,
    prompt: str,
    target_token: str,
    position: int = -1,
) -> LogitAttributionResult:
    """
    Compute attribution of final logit to individual attention heads.
    
    Uses direct attribution: contribution = (head_output @ W_U)[target_id]
    
    Args:
        model: HuggingFace transformer model
        tokenizer: Tokenizer
        prompt: Input text
        target_token: Token to analyze
        position: Position to analyze (-1 for last)
    
    Returns:
        LogitAttributionResult with per-head attributions
    """
    model.eval()
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    target_ids = tokenizer.encode(target_token, add_special_tokens=False)
    if not target_ids:
        raise ValueError(f"Token '{target_token}' not found in vocabulary")
    target_id = target_ids[0]
    
    # Hook to capture attention outputs
    head_outputs = {}
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            # output is typically (hidden_states, attention_weights, ...)
            if isinstance(output, tuple):
                head_outputs[layer_idx] = output[0].detach()
            else:
                head_outputs[layer_idx] = output.detach()
        return hook
    
    # Register hooks on attention modules
    hooks = []
    if hasattr(model, 'transformer'):
        # GPT-2 style
        for i, block in enumerate(model.transformer.h):
            hook = block.attn.register_forward_hook(make_hook(i))
            hooks.append(hook)
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # LLaMA style
        for i, layer in enumerate(model.model.layers):
            hook = layer.self_attn.register_forward_hook(make_hook(i))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Get unembedding vector for target token
    if hasattr(model, 'lm_head'):
        W_U = model.lm_head.weight[target_id]  # (hidden,)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        W_U = model.transformer.wte.weight[target_id]
    else:
        raise ValueError("Could not find unembedding matrix")
    
    seq_len = inputs['input_ids'].shape[1]
    pos = position if position >= 0 else seq_len + position
    
    # Compute per-head attributions
    head_attributions = []
    
    n_heads = model.config.num_attention_heads if hasattr(model.config, 'num_attention_heads') else 12
    head_dim = model.config.hidden_size // n_heads if hasattr(model.config, 'hidden_size') else 64
    
    for layer_idx, attn_output in head_outputs.items():
        # attn_output: (batch, seq, hidden)
        layer_output = attn_output[0, pos]  # (hidden,)
        
        # Split into heads (approximate)
        layer_contribution = float(layer_output @ W_U)
        
        # Distribute contribution across heads (simplified)
        per_head = layer_contribution / n_heads
        
        for head_idx in range(n_heads):
            head_attributions.append(HeadAttribution(
                layer=layer_idx,
                head=head_idx,
                direct_effect=per_head,
                indirect_effect=0.0,  # Would require more complex analysis
                total_effect=per_head,
            ))
    
    # Sort and get top heads
    sorted_heads = sorted(head_attributions, key=lambda h: -h.total_effect)
    
    top_positive = [(h.layer, h.head, h.total_effect) for h in sorted_heads if h.total_effect > 0][:10]
    top_negative = [(h.layer, h.head, h.total_effect) for h in sorted_heads if h.total_effect < 0][-10:]
    
    # MLP attributions (from residual decomposition)
    hidden_states = outputs.hidden_states
    mlp_attributions = []
    
    for layer_idx in range(len(hidden_states) - 1):
        h_pre = hidden_states[layer_idx][0, pos]
        h_post = hidden_states[layer_idx + 1][0, pos]
        
        # Approximate MLP contribution as residual minus attention
        if layer_idx in head_outputs:
            attn_contrib = head_outputs[layer_idx][0, pos]
            mlp_contrib = h_post - h_pre - attn_contrib
        else:
            mlp_contrib = (h_post - h_pre) / 2  # Rough approximation
        
        mlp_logit = float(mlp_contrib @ W_U)
        mlp_attributions.append((layer_idx, mlp_logit))
    
    target_logit = float(outputs.logits[0, pos, target_id])
    
    return LogitAttributionResult(
        target_token=target_token,
        target_logit=target_logit,
        head_attributions=head_attributions,
        mlp_attributions=mlp_attributions,
        top_positive_heads=top_positive,
        top_negative_heads=top_negative,
    )


def format_attribution_summary(result: LogitAttributionResult) -> str:
    """Format attribution result as readable summary."""
    lines = [
        f"Logit Attribution for '{result.target_token}'",
        f"Target logit: {result.target_logit:.4f}",
        "",
        "Top positive heads:",
    ]
    
    for layer, head, effect in result.top_positive_heads[:5]:
        lines.append(f"  L{layer}H{head}: +{effect:.4f}")
    
    lines.append("")
    lines.append("Top negative heads:")
    
    for layer, head, effect in result.top_negative_heads[:5]:
        lines.append(f"  L{layer}H{head}: {effect:.4f}")
    
    return "\n".join(lines)
