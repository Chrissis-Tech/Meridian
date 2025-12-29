"""
Meridian Explain Package

Interpretability tools:
- Attention: triage heuristic (information flow)
- Causal: intervention-based evidence (component importance)
- Attribution: quantitative logit decomposition
"""

from .attention_viz import (
    extract_attention_patterns,
    attention_to_plotly,
    find_important_heads,
)
from .logit_lens import (
    apply_logit_lens,
    logit_lens_to_plotly,
    find_emergence_layer,
)
from .causal_tracing import (
    CausalTracer,
    causal_heatmap_data,
)
from .activation_patching_demo import (
    ActivationPatcher,
    demo_activation_patching,
)
from .residual_decomp import (
    ResidualDecomposition,
    ComponentContribution,
    decompose_residual_stream,
    get_top_contributing_layers,
    format_decomposition_summary,
)
from .logit_attribution import (
    HeadAttribution,
    LogitAttributionResult,
    compute_head_attribution,
    format_attribution_summary,
)

__all__ = [
    # Attention (triage)
    "extract_attention_patterns",
    "attention_to_plotly",
    "find_important_heads",
    # Logit Lens
    "apply_logit_lens",
    "logit_lens_to_plotly",
    "find_emergence_layer",
    # Causal Tracing (evidence)
    "CausalTracer",
    "causal_heatmap_data",
    "ActivationPatcher",
    "demo_activation_patching",
    # Residual Decomposition (attribution)
    "ResidualDecomposition",
    "ComponentContribution",
    "decompose_residual_stream",
    "get_top_contributing_layers",
    "format_decomposition_summary",
    # Head Attribution
    "HeadAttribution",
    "LogitAttributionResult",
    "compute_head_attribution",
    "format_attribution_summary",
]

