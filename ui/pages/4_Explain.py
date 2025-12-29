"""
Meridian UI - Explain Page (Interpretability)
"""

import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Explain - Meridian", page_icon="M", layout="wide")

st.title("Interpretability Analysis")

# Clear explanation of requirements
st.markdown("""
### Requirements

Interpretability features analyze the internal behavior of language models. This requires:

1. **Local model execution** - The model must run on your machine (not via API)
2. **PyTorch and Transformers** - `pip install torch transformers`
3. **Sufficient RAM** - DistilGPT-2 needs ~1GB, Flan-T5 needs ~2GB

**Why can't this work with API models like DeepSeek or GPT-4?**

API models only return text output. To analyze attention patterns, logit lens, and causal tracing, 
we need access to intermediate layer activations, which are not exposed by commercial APIs.
""")

st.markdown("---")

# Check if local models are available
try:
    import torch
    import transformers
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if not TORCH_AVAILABLE:
    st.error("""
    **PyTorch or Transformers not installed.**
    
    Install with:
    ```
    pip install torch transformers
    ```
    
    Then restart the application.
    """)
    st.stop()

# Model selection - use actually available local models
st.markdown("### Select Local Model")

LOCAL_MODELS = {
    "local_distilgpt2": {
        "name": "DistilGPT-2 (82M)",
        "size": "~300MB",
        "ram": "~1GB",
        "description": "Lightweight, fast, good for demos"
    },
    "local_flan_t5_small": {
        "name": "Flan-T5 Small (77M)",
        "size": "~300MB",
        "ram": "~1GB",
        "description": "Instruction-tuned, encoder-decoder"
    },
    "local_flan_t5_base": {
        "name": "Flan-T5 Base (250M)",
        "size": "~1GB",
        "ram": "~2GB",
        "description": "Better quality, more memory"
    },
}

model_id = st.selectbox(
    "Model",
    list(LOCAL_MODELS.keys()),
    format_func=lambda x: f"{LOCAL_MODELS[x]['name']} - {LOCAL_MODELS[x]['size']} download, {LOCAL_MODELS[x]['ram']} RAM"
)

st.caption(LOCAL_MODELS[model_id]["description"])

# Prompt input
st.markdown("### Input Prompt")
prompt = st.text_area(
    "Enter text to analyze",
    value="The capital of France is",
    height=100,
    help="The model will complete this text and we'll analyze how it made decisions"
)

if st.button("Analyze", type="primary"):
    try:
        from core.model_adapters import get_adapter
        from core.explain import (
            extract_attention_patterns, attention_to_plotly, find_important_heads,
            apply_logit_lens, logit_lens_to_plotly,
        )
        
        with st.spinner("Loading model (this may take a minute on first run)..."):
            adapter = get_adapter(model_id)
        
        # Generate output first
        st.markdown("### Model Output")
        from core.model_adapters.base import GenerationConfig
        result = adapter.generate(prompt, GenerationConfig(max_tokens=50))
        
        st.success(f"**Completion:** {result.output}")
        
        st.markdown("---")
        
        # Tabs for different analyses
        tab1, tab2 = st.tabs(["Attention Patterns", "Logit Lens"])
        
        with tab1:
            st.markdown("### Attention Patterns")
            st.caption("Shows which tokens the model 'pays attention to' when generating the next token")
            
            try:
                with st.spinner("Extracting attention weights..."):
                    patterns = extract_attention_patterns(adapter, prompt)
                
                if patterns:
                    # Find important heads
                    important = find_important_heads(patterns, top_k=5)
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("**Most Important Heads:**")
                        for i, (layer, head, score) in enumerate(important):
                            st.markdown(f"{i+1}. Layer {layer}, Head {head} — score: `{score:.3f}`")
                    
                    with col2:
                        # Show heatmap for top head
                        if important:
                            top_layer, top_head, _ = important[0]
                            pattern = next((p for p in patterns if p.layer == top_layer and p.head == top_head), None)
                            
                            if pattern:
                                fig_data = attention_to_plotly(pattern)
                                fig = go.Figure(fig_data["data"], fig_data["layout"])
                                fig.update_layout(
                                    height=400,
                                    title=f"Attention Pattern - Layer {top_layer}, Head {top_head}",
                                    paper_bgcolor='rgba(0,0,0,0)'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Attention extraction not supported for this model type")
            except Exception as e:
                st.warning(f"Attention analysis not available: {e}")
        
        with tab2:
            st.markdown("### Logit Lens")
            st.caption("Shows the model's 'early predictions' at each layer before the final output")
            
            try:
                with st.spinner("Applying logit lens..."):
                    lens_result = apply_logit_lens(adapter, prompt)
                
                if lens_result and lens_result.top_k_predictions:
                    st.markdown("**Prediction Evolution by Layer:**")
                    
                    # Create a clean visualization
                    layer_data = []
                    for i, preds in enumerate(lens_result.top_k_predictions):
                        if preds:
                            token, prob = preds[0]
                            layer_data.append({
                                "Layer": i,
                                "Prediction": repr(token),
                                "Confidence": prob
                            })
                    
                    if layer_data:
                        import pandas as pd
                        df = pd.DataFrame(layer_data)
                        
                        fig = go.Figure(go.Bar(
                            x=df["Layer"],
                            y=df["Confidence"],
                            text=df["Prediction"],
                            textposition="outside",
                            marker_color='#3498db'
                        ))
                        fig.update_layout(
                            title="Top Prediction Confidence by Layer",
                            xaxis_title="Layer",
                            yaxis_title="Confidence",
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        fig.update_xaxes(gridcolor='#ecf0f1')
                        fig.update_yaxes(gridcolor='#ecf0f1', range=[0, 1])
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Logit lens not supported for this model type")
            except Exception as e:
                st.warning(f"Logit lens not available: {e}")
    
    except Exception as e:
        st.error(f"Error during analysis: {e}")
        with st.expander("See full error"):
            import traceback
            st.code(traceback.format_exc())
