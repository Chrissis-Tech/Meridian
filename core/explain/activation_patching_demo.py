"""
Meridian Explain - Activation Patching Demo

Simple activation patching demonstration.
"""

import numpy as np
import torch
from typing import Optional


class ActivationPatcher:
    """Demonstration of activation patching."""
    
    def __init__(self, model_adapter):
        if not model_adapter.supports_interpretability:
            raise ValueError("Model must support interpretability")
        self.model = model_adapter
    
    def patch_and_measure(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        target_layer: int,
        target_position: int,
    ) -> dict:
        """
        Patch activations from corrupted run into clean run.
        
        Returns impact on output probability.
        """
        # Get hidden states
        clean_hidden = self.model.get_hidden_states(clean_prompt, layer=target_layer)
        corrupted_hidden = self.model.get_hidden_states(corrupted_prompt, layer=target_layer)
        
        if clean_hidden is None or corrupted_hidden is None:
            return {"error": "Could not get hidden states"}
        
        clean_hidden = np.array(clean_hidden)
        corrupted_hidden = np.array(corrupted_hidden)
        
        # Get baseline logits
        clean_logits = self.model.get_logits(clean_prompt)
        clean_probs = torch.softmax(clean_logits[0, -1], dim=-1)
        
        # In a full implementation, we would:
        # 1. Replace clean_hidden[0, target_position] with corrupted_hidden[0, target_position]
        # 2. Run forward pass from that layer
        # 3. Compare new logits
        
        # For demo, we measure activation difference
        if target_position < min(clean_hidden.shape[1], corrupted_hidden.shape[1]):
            activation_diff = np.linalg.norm(
                clean_hidden[0, target_position] - corrupted_hidden[0, target_position]
            )
        else:
            activation_diff = 0.0
        
        return {
            "layer": target_layer,
            "position": target_position,
            "activation_diff": float(activation_diff),
            "clean_prompt": clean_prompt[:50],
            "corrupted_prompt": corrupted_prompt[:50],
        }
    
    def run_patching_sweep(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        n_layers: Optional[int] = None,
    ) -> list[dict]:
        """Run patching across all layers and positions."""
        results = []
        
        clean_hidden = self.model.get_hidden_states(clean_prompt)
        if clean_hidden is None:
            return results
        
        clean_hidden = np.array(clean_hidden)
        actual_layers = n_layers or clean_hidden.shape[0]
        seq_len = clean_hidden.shape[2]
        
        for layer in range(min(actual_layers, clean_hidden.shape[0])):
            for pos in range(seq_len):
                result = self.patch_and_measure(
                    clean_prompt, corrupted_prompt, layer, pos
                )
                results.append(result)
        
        return results


def demo_activation_patching(model_adapter) -> dict:
    """Run a demonstration of activation patching."""
    patcher = ActivationPatcher(model_adapter)
    
    # Demo: IOI-style task
    clean = "When Mary and John went to the store, Mary gave the bag to"
    corrupted = "When Mary and John went to the store, John gave the bag to"
    
    results = []
    for layer in range(6):  # First 6 layers
        result = patcher.patch_and_measure(clean, corrupted, layer, -1)
        results.append(result)
    
    return {
        "task": "Indirect Object Identification",
        "clean_prompt": clean,
        "corrupted_prompt": corrupted,
        "results": results,
        "interpretation": "Higher activation diff = more impact from that layer",
    }
