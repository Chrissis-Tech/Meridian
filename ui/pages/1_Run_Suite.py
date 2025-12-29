"""
Meridian UI - Run Suite Page
"""

import streamlit as st
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Run Suite - Meridian", page_icon="M", layout="wide")

st.title("Run Suite")

# Initialize session state for API keys
if "deepseek_key" not in st.session_state:
    st.session_state.deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
if "openai_key" not in st.session_state:
    st.session_state.openai_key = os.getenv("OPENAI_API_KEY", "")

# API Key configuration in sidebar
with st.sidebar:
    st.subheader("API Configuration")
    
    # DeepSeek
    st.markdown("**DeepSeek**")
    deepseek_key = st.text_input(
        "API Key (sk-...)",
        value=st.session_state.deepseek_key,
        key="deepseek_input",
        label_visibility="collapsed",
        placeholder="sk-xxxxxxxxxxxxxxxx",
        autocomplete="off"
    )
    
    # Validate DeepSeek key
    if deepseek_key:
        if len(deepseek_key) >= 20 and deepseek_key.startswith("sk-"):
            st.session_state.deepseek_key = deepseek_key
            os.environ["DEEPSEEK_API_KEY"] = deepseek_key
            st.success("DeepSeek: Valid")
        else:
            st.error("Invalid format")
    
    st.markdown("---")
    
    # OpenAI
    st.markdown("**OpenAI**")
    openai_key = st.text_input(
        "API Key (sk-...)",
        value=st.session_state.openai_key,
        key="openai_input",
        label_visibility="collapsed",
        placeholder="sk-proj-xxxxxxxx",
        autocomplete="off"
    )
    
    # Validate OpenAI key
    if openai_key:
        if len(openai_key) >= 20:
            st.session_state.openai_key = openai_key
            os.environ["OPENAI_API_KEY"] = openai_key
            st.success("OpenAI: Valid")
        else:
            st.error("Invalid format")

# Model info with context lengths (approximate, as of Dec 2025)
MODEL_INFO = {
    "deepseek_chat": {"name": "DeepSeek Chat", "context": "32K", "price": "$0.14/M"},
    "deepseek_coder": {"name": "DeepSeek Coder", "context": "16K", "price": "$0.14/M"},
    "openai_gpt35": {"name": "GPT-3.5 Turbo", "context": "16K", "price": "$0.50/M"},
    "openai_gpt4": {"name": "GPT-4", "context": "8K", "price": "$30/M"},
    "openai_gpt4_turbo": {"name": "GPT-4 Turbo", "context": "128K", "price": "$10/M"},
}

# Build list of working models (API-based only)
working_models = []

# DeepSeek models (if valid key)
if st.session_state.deepseek_key and len(st.session_state.deepseek_key) >= 20:
    working_models.extend(["deepseek_chat", "deepseek_coder"])

# OpenAI models (if valid key)
if st.session_state.openai_key and len(st.session_state.openai_key) >= 20:
    working_models.extend(["openai_gpt35", "openai_gpt4", "openai_gpt4_turbo"])

# Model selection
col1, col2 = st.columns(2)

with col1:
    if not working_models:
        st.error("No models available. Configure at least one API key in sidebar.")
        st.stop()
    
    def format_model(model_id):
        info = MODEL_INFO.get(model_id, {})
        return f"{info.get('name', model_id)} ({info.get('context', '?')} context, ~{info.get('price', '?')})"
    
    model_id = st.selectbox(
        "Select Model",
        working_models,
        format_func=format_model
    )

with col2:
    try:
        from core.config import SUITES_DIR
        suites = sorted([p.stem for p in SUITES_DIR.glob("*.jsonl")])
        
        # Recommended suites first
        recommended_suites = ["rag_evaluation", "code_analysis", "business_analysis", "edge_cases"]
        ordered_suites = [s for s in recommended_suites if s in suites]
        ordered_suites += [s for s in suites if s not in recommended_suites]
        
        suite_name = st.selectbox("Select Suite", ordered_suites)
    except:
        suite_name = st.selectbox("Select Suite", ["rag_evaluation", "enterprise_prompts"])

# Pricing disclaimer
st.caption(
    "Prices are estimates as of December 2025 and may change. "
    "Verify current pricing: [DeepSeek](https://platform.deepseek.com/api-docs/pricing) | "
    "[OpenAI](https://openai.com/pricing)"
)

# Configuration
st.subheader("Configuration")
col1, col2, col3 = st.columns(3)

with col1:
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
with col2:
    max_tokens = st.slider("Max Tokens", 50, 500, 256)
with col3:
    num_runs = st.number_input("Runs (for consistency)", 1, 5, 1)

# Run button
if st.button("Run Evaluation", type="primary", use_container_width=True):
    try:
        from core.runner import SuiteRunner
        from core.types import RunConfig
        from core.config import SUITES_DIR
        
        suite_path = SUITES_DIR / f"{suite_name}.jsonl"
        
        runner = SuiteRunner()
        config = RunConfig(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total):
            progress_bar.progress(current / total)
            status_text.text(f"Running: {current}/{total} tests")
        
        with st.spinner(f"Running {suite_name} with {model_id}..."):
            result = runner.run_suite(
                str(suite_path),
                model_id=model_id,
                run_config=config,
                progress_callback=update_progress
            )
        
        progress_bar.progress(1.0)
        status_text.empty()
        
        # Display results
        st.success(f"Completed: {result.passed_tests}/{result.total_tests} passed")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{result.accuracy:.1%}")
        col2.metric("Passed", result.passed_tests)
        col3.metric("Failed", result.failed_tests)
        col4.metric("Latency", f"{result.mean_latency_ms:.0f}ms")
        
        if result.accuracy_ci:
            st.info(f"95% CI: [{result.accuracy_ci[0]:.1%}, {result.accuracy_ci[1]:.1%}]")
        
        st.markdown("---")
        st.markdown(f"**Run ID:** `{result.run_id}`")
        st.markdown("View detailed results in the **Results** page.")
        
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        with st.expander("See full error"):
            st.code(traceback.format_exc())
