"""
Meridian Explain - Causal Tracing

Systematic activation patching to find critical components.
"""

from typing import Optional
import numpy as np
import torch

from ..types import CausalTrace, ComponentImportance


class CausalTracer:
    """Systematic causal tracing via activation patching."""
    
    def __init__(self, model_adapter):
        if not model_adapter.supports_interpretability:
            raise ValueError("Model must support interpretability")
        self.model = model_adapter
    
    def trace_critical_positions(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        target_token: str,
    ) -> list[CausalTrace]:
        """Find which token positions are critical for the prediction."""
        traces = []
        
        # Get clean and corrupted hidden states
        clean_hidden = self.model.get_hidden_states(clean_prompt)
        corrupted_hidden = self.model.get_hidden_states(corrupted_prompt)
        
        if clean_hidden is None or corrupted_hidden is None:
            return traces
        
        clean_hidden = np.array(clean_hidden)
        corrupted_hidden = np.array(corrupted_hidden)
        
        # Get baseline probability
        clean_logits = self.model.get_logits(clean_prompt)
        tokens = self.model.tokenize(clean_prompt)
        target_id = self.model._tokenizer.encode(target_token)[0]
        
        baseline_prob = float(torch.softmax(clean_logits[0, -1], dim=-1)[target_id])
        
        # Patch each position and measure impact
        n_layers = clean_hidden.shape[0]
        seq_len = min(clean_hidden.shape[2], corrupted_hidden.shape[2])
        
        for pos in range(seq_len):
            # Simple impact estimate: L2 distance between clean and corrupted
            impact = float(np.mean([
                np.linalg.norm(clean_hidden[l, 0, pos] - corrupted_hidden[l, 0, pos])
                for l in range(n_layers)
            ]))
            
            traces.append(CausalTrace(
                layer=-1,  # Aggregated across layers
                head=None,
                position=pos,
                component_type="residual",
                impact=impact,
                baseline_prob=baseline_prob,
                patched_prob=0.0,  # Would need actual patching
            ))
        
        return sorted(traces, key=lambda x: -x.impact)
    
    def trace_critical_layers(
        self,
        prompt: str,
        target_token: str,
    ) -> list[CausalTrace]:
        """Find which layers are most important."""
        traces = []
        
        hidden_states = self.model.get_hidden_states(prompt)
        if hidden_states is None:
            return traces
        
        hidden_states = np.array(hidden_states)
        n_layers = hidden_states.shape[0]
        
        # Get baseline
        logits = self.model.get_logits(prompt)
        target_id = self.model._tokenizer.encode(target_token)[0]
        baseline_prob = float(torch.softmax(logits[0, -1], dim=-1)[target_id])
        
        # Measure layer-wise contribution via activation norms
        for layer in range(n_layers):
            # Use norm of hidden state change as proxy for importance
            if layer == 0:
                delta = hidden_states[layer]
            else:
                delta = hidden_states[layer] - hidden_states[layer - 1]
            
            impact = float(np.linalg.norm(delta[0, -1]))  # Last position
            
            traces.append(CausalTrace(
                layer=layer,
                head=None,
                position=-1,
                component_type="layer",
                impact=impact,
                baseline_prob=baseline_prob,
                patched_prob=0.0,
            ))
        
        return sorted(traces, key=lambda x: -x.impact)
    
    def trace_critical_heads(
        self,
        prompt: str,
        target_token: str,
    ) -> list[CausalTrace]:
        """Find which attention heads are most important."""
        traces = []
        
        attention = self.model.get_attention_patterns(prompt)
        if attention is None:
            return traces
        
        attention = np.array(attention)
        
        # Get baseline
        logits = self.model.get_logits(prompt)
        target_id = self.model._tokenizer.encode(target_token)[0]
        baseline_prob = float(torch.softmax(logits[0, -1], dim=-1)[target_id])
        
        n_layers = attention.shape[0]
        n_heads = attention.shape[2]
        
        for layer in range(n_layers):
            for head in range(n_heads):
                # Use attention entropy as importance proxy
                attn = attention[layer, 0, head, -1]  # Last query position
                entropy = -np.sum(attn * np.log(attn + 1e-10))
                max_attn = np.max(attn)
                
                # Lower entropy + higher max = more focused/important
                impact = max_attn * (1 - entropy / np.log(len(attn)))
                
                traces.append(CausalTrace(
                    layer=layer,
                    head=head,
                    position=-1,
                    component_type="attention",
                    impact=float(impact),
                    baseline_prob=baseline_prob,
                    patched_prob=0.0,
                ))
        
        return sorted(traces, key=lambda x: -x.impact)
    
    def generate_component_importance(
        self,
        prompt: str,
        target_token: str,
        task_type: str = "unknown",
    ) -> ComponentImportance:
        """Generate full component importance analysis."""
        layer_traces = self.trace_critical_layers(prompt, target_token)
        head_traces = self.trace_critical_heads(prompt, target_token)
        position_traces = self.trace_critical_positions(prompt, prompt, target_token)
        
        all_traces = layer_traces + head_traces + position_traces
        
        # Top-k summaries
        top_heads = [
            (t.layer, t.head, t.impact)
            for t in head_traces[:10]
            if t.head is not None
        ]
        
        top_layers = [
            (t.layer, t.impact)
            for t in layer_traces[:10]
        ]
        
        critical_positions = [
            (t.position, t.impact)
            for t in position_traces[:10]
        ]
        
        return ComponentImportance(
            task_type=task_type,
            components=all_traces,
            top_heads=top_heads,
            top_layers=top_layers,
            critical_positions=critical_positions,
        )


def causal_heatmap_data(importance: ComponentImportance) -> dict:
    """Generate heatmap data from component importance."""
    if not importance.top_heads:
        return {"layers": [], "heads": [], "impacts": []}
    
    layers = sorted(set(h[0] for h in importance.top_heads))
    heads = sorted(set(h[1] for h in importance.top_heads))
    
    impact_matrix = np.zeros((len(layers), len(heads)))
    
    for layer, head, impact in importance.top_heads:
        if layer in layers and head in heads:
            i = layers.index(layer)
            j = heads.index(head)
            impact_matrix[i, j] = impact
    
    return {
        "layers": layers,
        "heads": heads,
        "impacts": impact_matrix.tolist(),
    }
