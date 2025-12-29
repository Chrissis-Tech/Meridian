"""
Meridian Explain - Logit Lens

Logit lens to see model predictions at each layer.
"""

from typing import Optional, Any
import numpy as np

from ..types import LogitLensResult


def apply_logit_lens(
    model_adapter,
    prompt: str,
    positions: Optional[list[int]] = None,
    top_k: int = 10,
) -> LogitLensResult:
    """
    Apply logit lens to see predictions at each layer.
    
    Shows what token the model would predict if we stopped at each layer.
    """
    if not model_adapter.supports_interpretability:
        return LogitLensResult(layers=[], positions=[], tokens=[], top_k_predictions=[])
    
    # Use model's built-in logit lens if available
    if hasattr(model_adapter, 'get_logit_lens'):
        tokens = model_adapter.tokenize(prompt)
        position = positions[-1] if positions else -1
        layer_preds = model_adapter.get_logit_lens(prompt, position, top_k)
        
        return LogitLensResult(
            layers=list(range(len(layer_preds))),
            positions=[position] * len(layer_preds),
            tokens=tokens,
            top_k_predictions=layer_preds,
        )
    
    return LogitLensResult(layers=[], positions=[], tokens=[], top_k_predictions=[])


def logit_lens_to_plotly(result: LogitLensResult) -> dict:
    """Convert logit lens result to Plotly visualization."""
    if not result.top_k_predictions:
        return {"data": [], "layout": {"title": "No data"}}
    
    # Extract top-1 prediction at each layer
    layers = result.layers
    top_tokens = [preds[0][0] if preds else "" for preds in result.top_k_predictions]
    top_probs = [preds[0][1] if preds else 0 for preds in result.top_k_predictions]
    
    return {
        "data": [{
            "type": "bar",
            "x": layers,
            "y": top_probs,
            "text": top_tokens,
            "textposition": "outside",
        }],
        "layout": {
            "title": "Top-1 Prediction Probability by Layer",
            "xaxis": {"title": "Layer"},
            "yaxis": {"title": "Probability", "range": [0, 1]},
        }
    }


def find_emergence_layer(result: LogitLensResult, target_token: str) -> Optional[int]:
    """Find the layer where a target token first appears in top-k."""
    for i, preds in enumerate(result.top_k_predictions):
        tokens = [p[0].strip() for p in preds]
        if target_token.strip() in tokens:
            return i
    return None


def layer_transition_analysis(result: LogitLensResult) -> dict:
    """Analyze how predictions change across layers."""
    if len(result.top_k_predictions) < 2:
        return {"transitions": [], "stable_from": None}
    
    transitions = []
    prev_top = None
    stable_from = None
    
    for i, preds in enumerate(result.top_k_predictions):
        top = preds[0][0] if preds else ""
        if prev_top is not None and top != prev_top:
            transitions.append({"layer": i, "from": prev_top, "to": top})
            stable_from = None
        elif stable_from is None:
            stable_from = i
        prev_top = top
    
    return {
        "transitions": transitions,
        "stable_from": stable_from,
        "final_prediction": result.top_k_predictions[-1][0][0] if result.top_k_predictions else None,
    }
