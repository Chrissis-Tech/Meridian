"""
Meridian Explain - Residual Stream Decomposition

Decomposes the residual stream to attribute predictions to components.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ComponentContribution:
    """Contribution of a model component to a prediction."""
    component_type: str  # "attention", "mlp", "embedding"
    layer: int
    head: Optional[int]  # None for MLP
    contribution: float
    logit_diff: float


@dataclass
class ResidualDecomposition:
    """Full decomposition of residual stream."""
    target_token: str
    target_position: int
    final_logit: float
    contributions: list[ComponentContribution]
    total_attributed: float
    attribution_coverage: float  # fraction of logit explained


def decompose_residual_stream(
    model,
    tokenizer,
    prompt: str,
    target_token: str,
    target_position: int = -1,
) -> ResidualDecomposition:
    """
    Decompose the residual stream to attribute logit to components.
    
    For each layer, computes:
    - Attention contribution: how much attention heads contribute
    - MLP contribution: how much MLP contributes
    
    Args:
        model: HuggingFace model with output_hidden_states
        tokenizer: Tokenizer
        prompt: Input prompt
        target_token: Token to analyze
        target_position: Position in sequence (-1 for last)
    
    Returns:
        ResidualDecomposition with component contributions
    """
    model.eval()
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    target_id = tokenizer.encode(target_token, add_special_tokens=False)
    if not target_id:
        raise ValueError(f"Target token '{target_token}' not in vocabulary")
    target_id = target_id[0]
    
    # Get hidden states
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,
        )
    
    hidden_states = outputs.hidden_states  # (num_layers+1, batch, seq, hidden)
    logits = outputs.logits
    
    seq_len = hidden_states[0].shape[1]
    pos = target_position if target_position >= 0 else seq_len + target_position
    
    # Final logit for target token
    final_logit = float(logits[0, pos, target_id])
    
    contributions = []
    total_attributed = 0.0
    
    num_layers = len(hidden_states) - 1
    
    # Get embedding matrix
    if hasattr(model, 'lm_head'):
        embed_matrix = model.lm_head.weight  # (vocab, hidden)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        embed_matrix = model.transformer.wte.weight
    else:
        embed_matrix = None
    
    if embed_matrix is None:
        # Fallback: approximate contribution
        for layer_idx in range(num_layers):
            h_pre = hidden_states[layer_idx][0, pos]
            h_post = hidden_states[layer_idx + 1][0, pos]
            delta = h_post - h_pre
            
            # Approximate contribution as norm of delta
            contribution = float(torch.norm(delta))
            contributions.append(ComponentContribution(
                component_type="layer",
                layer=layer_idx,
                head=None,
                contribution=contribution,
                logit_diff=0.0,
            ))
            total_attributed += contribution
    else:
        # Proper logit lens decomposition
        target_row = embed_matrix[target_id]  # (hidden,)
        
        # Embedding contribution
        h_embed = hidden_states[0][0, pos]
        embed_logit = float(h_embed @ target_row)
        contributions.append(ComponentContribution(
            component_type="embedding",
            layer=0,
            head=None,
            contribution=embed_logit,
            logit_diff=embed_logit,
        ))
        total_attributed += abs(embed_logit)
        
        # Layer contributions
        for layer_idx in range(num_layers):
            h_pre = hidden_states[layer_idx][0, pos]
            h_post = hidden_states[layer_idx + 1][0, pos]
            delta = h_post - h_pre
            
            logit_diff = float(delta @ target_row)
            contribution = abs(logit_diff)
            
            contributions.append(ComponentContribution(
                component_type="layer",
                layer=layer_idx,
                head=None,
                contribution=contribution,
                logit_diff=logit_diff,
            ))
            total_attributed += contribution
    
    # Sort by contribution magnitude
    contributions.sort(key=lambda c: -abs(c.contribution))
    
    attribution_coverage = total_attributed / abs(final_logit) if final_logit != 0 else 0
    
    return ResidualDecomposition(
        target_token=target_token,
        target_position=pos,
        final_logit=final_logit,
        contributions=contributions,
        total_attributed=total_attributed,
        attribution_coverage=min(1.0, attribution_coverage),
    )


def get_top_contributing_layers(
    decomposition: ResidualDecomposition,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Get top contributing layers from decomposition."""
    layer_contribs = [
        (c.layer, c.logit_diff)
        for c in decomposition.contributions
        if c.component_type == "layer"
    ]
    layer_contribs.sort(key=lambda x: -abs(x[1]))
    return layer_contribs[:top_k]


def format_decomposition_summary(decomposition: ResidualDecomposition) -> str:
    """Format decomposition as readable summary."""
    lines = [
        f"Residual Stream Decomposition for '{decomposition.target_token}'",
        f"Final logit: {decomposition.final_logit:.4f}",
        f"Attribution coverage: {decomposition.attribution_coverage:.1%}",
        "",
        "Top contributions:",
    ]
    
    for c in decomposition.contributions[:10]:
        sign = "+" if c.logit_diff >= 0 else ""
        lines.append(f"  Layer {c.layer}: {sign}{c.logit_diff:.4f}")
    
    return "\n".join(lines)
