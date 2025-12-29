"""
Meridian Explain - Attention Visualization

Attention pattern visualization and analysis.
"""

from typing import Optional, Any
import numpy as np

from ..types import AttentionPattern


def extract_attention_patterns(
    model_adapter,
    prompt: str,
    layer: Optional[int] = None,
    head: Optional[int] = None,
) -> list[AttentionPattern]:
    """Extract attention patterns from a model."""
    attention = model_adapter.get_attention_patterns(prompt, layer, head)
    tokens = model_adapter.tokenize(prompt)
    
    if attention is None:
        return []
    
    patterns = []
    attention = np.array(attention)
    
    # Shape: [layers, batch, heads, seq, seq] or subset
    if len(attention.shape) == 5:
        n_layers, _, n_heads, seq_len, _ = attention.shape
        for l in range(n_layers):
            for h in range(n_heads):
                patterns.append(AttentionPattern(
                    layer=l,
                    head=h,
                    attention_matrix=attention[l, 0, h],
                    tokens=tokens[:seq_len],
                ))
    elif len(attention.shape) == 4:
        _, n_heads, seq_len, _ = attention.shape
        for h in range(n_heads):
            patterns.append(AttentionPattern(
                layer=layer or 0,
                head=h,
                attention_matrix=attention[0, h],
                tokens=tokens[:seq_len],
            ))
    
    return patterns


def attention_to_plotly(pattern: AttentionPattern) -> dict:
    """Convert attention pattern to Plotly heatmap spec."""
    return {
        "data": [{
            "type": "heatmap",
            "z": pattern.attention_matrix.tolist(),
            "x": pattern.tokens,
            "y": pattern.tokens,
            "colorscale": "Viridis",
        }],
        "layout": {
            "title": f"Layer {pattern.layer}, Head {pattern.head}",
            "xaxis": {"title": "Key"},
            "yaxis": {"title": "Query"},
        }
    }


def find_important_heads(
    patterns: list[AttentionPattern],
    query_position: int = -1,
    top_k: int = 10,
) -> list[tuple[int, int, float]]:
    """Find heads with strongest attention at a position."""
    scores = []
    
    for p in patterns:
        # Entropy-based importance (lower entropy = more focused)
        attn_row = p.attention_matrix[query_position]
        entropy = -np.sum(attn_row * np.log(attn_row + 1e-10))
        max_attn = np.max(attn_row)
        
        # Score: higher max attention, lower entropy = more important
        score = max_attn - 0.1 * entropy
        scores.append((p.layer, p.head, float(score)))
    
    return sorted(scores, key=lambda x: -x[2])[:top_k]
