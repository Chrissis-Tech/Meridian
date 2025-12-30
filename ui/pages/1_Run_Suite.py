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
    # Get built-in suites
    try:
        from core.config import SUITES_DIR
        builtin_suites = sorted([p.stem for p in SUITES_DIR.glob("*.jsonl")])
    except:
        builtin_suites = ["rag_evaluation", "code_analysis", "business_analysis"]
    
    # Get custom suites
    try:
        from meridian.suites.custom import get_custom_suite_manager
        manager = get_custom_suite_manager()
        custom_suites_list = manager.list_all()
        custom_suite_names = [f"[Custom] {s['name']}" for s in custom_suites_list]
    except:
        custom_suite_names = []
    
    # Combine: recommended first, then custom, then rest
    recommended = ["rag_evaluation", "code_analysis", "business_analysis", "edge_cases"]
    ordered = [s for s in recommended if s in builtin_suites]
    ordered += custom_suite_names  # Custom suites after recommended
    ordered += [s for s in builtin_suites if s not in recommended]
    
    suite_selection = st.selectbox("Select Suite", ordered)
    
    # Parse if custom
    is_custom_suite = suite_selection.startswith("[Custom] ")
    if is_custom_suite:
        suite_name = suite_selection.replace("[Custom] ", "")
        st.caption("Custom suite from your uploaded data")
    else:
        suite_name = suite_selection

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

# Attestation section
st.markdown("---")
st.subheader("Tamper-Evident Attestation")

col1, col2 = st.columns([1, 3])
with col1:
    enable_attestation = st.checkbox("Enable Attestation", value=False)

with col2:
    with st.expander("What is Attestation?"):
        st.markdown("""
**Tamper-Evident Golden Runs** create cryptographically verified evaluation bundles.

**What it does:**
- Generates SHA256 hashes of all run artifacts
- Captures environment info (Python, OS, git commit)
- Creates a verifiable manifest
- Detects any modifications to results

**Bundle structure:**
```
results/run_xxx/
├── manifest.json      # Hashes of all files
├── config.json        # Run parameters (secrets redacted)
├── suite.jsonl        # Dataset snapshot
├── responses/         # Raw model outputs
└── attestation.json   # Verification metadata
```

**CLI verification:**
```bash
python -m meridian.cli verify --id <run_id>
```

**Use cases:**
- Compliance audits
- Reproducibility proof
- Model evaluation evidence
""")

# Run button
if st.button("Run Evaluation", type="primary", use_container_width=True):
    try:
        from meridian.runner import SuiteRunner
        from meridian.types import RunConfig
        from meridian.config import SUITES_DIR
        import tempfile
        import json
        
        runner = SuiteRunner()
        config = RunConfig(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Handle custom vs built-in suites
        if is_custom_suite:
            from meridian.suites.custom import get_custom_suite_manager
            from dataclasses import asdict
            
            manager = get_custom_suite_manager()
            pack = manager.get_by_name(suite_name)
            
            if not pack:
                st.error(f"Custom suite '{suite_name}' not found")
                st.stop()
            
            # Get holdout split
            dev_tests, holdout_tests = pack.get_holdout_split()
            
            st.info(f"Running custom suite: {len(dev_tests)} dev + {len(holdout_tests)} holdout tests")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run DEV set
            status_text.text(f"Running Dev set ({len(dev_tests)} tests)...")
            dev_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
            for t in dev_tests:
                dev_file.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
            dev_file.close()
            
            dev_result = runner.run_suite(
                dev_file.name,
                model_id=model_id,
                run_config=config,
            )
            progress_bar.progress(0.5)
            
            # Run HOLDOUT set
            status_text.text(f"Running Holdout set ({len(holdout_tests)} tests)...")
            holdout_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
            for t in holdout_tests:
                holdout_file.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
            holdout_file.close()
            
            holdout_result = runner.run_suite(
                holdout_file.name,
                model_id=model_id,
                run_config=config,
            )
            progress_bar.progress(1.0)
            status_text.empty()
            
            # Display BOTH results
            st.success(f"Completed: Dev {dev_result.passed_tests}/{dev_result.total_tests} | Holdout {holdout_result.passed_tests}/{holdout_result.total_tests}")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Dev Accuracy", f"{dev_result.accuracy:.1%}")
            col2.metric("Holdout Accuracy", f"{holdout_result.accuracy:.1%}", 
                       delta=f"{(holdout_result.accuracy - dev_result.accuracy)*100:+.1f}pp" if dev_result.accuracy else None)
            col3.metric("Total Passed", dev_result.passed_tests + holdout_result.passed_tests)
            col4.metric("Latency", f"{(dev_result.mean_latency_ms + holdout_result.mean_latency_ms)/2:.0f}ms")
            
            # Warning if holdout much worse than dev (overfitting signal)
            if dev_result.accuracy > 0 and holdout_result.accuracy < dev_result.accuracy * 0.8:
                st.warning("Holdout accuracy significantly lower than Dev. This may indicate overfitting to your test cases.")
            
            # Use holdout result for attestation (the "real" score)
            result = holdout_result
            result.suite_name = suite_name  # Override temp filename
            
            # Create combined temp for attestation
            suite_path = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
            suite_path.write(pack.to_jsonl())
            suite_path.close()
            suite_path = Path(suite_path.name)
            
        else:
            suite_path = SUITES_DIR / f"{suite_name}.jsonl"
            
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
            
            # Display results (built-in suites - no holdout split)
            st.success(f"Completed: {result.passed_tests}/{result.total_tests} passed")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{result.accuracy:.1%}")
            col2.metric("Passed", result.passed_tests)
            col3.metric("Failed", result.failed_tests)
            col4.metric("Latency", f"{result.mean_latency_ms:.0f}ms")
        
        if result.accuracy_ci:
            st.info(f"95% CI: [{result.accuracy_ci[0]:.1%}, {result.accuracy_ci[1]:.1%}]")
        
        # Generate attestation if enabled
        if enable_attestation:
            st.markdown("---")
            with st.spinner("Generating attestation bundle..."):
                from meridian.storage.attestation import get_attestation_manager
                from meridian.storage.jsonl import load_test_suite
                from dataclasses import asdict
                
                attester = get_attestation_manager()
                
                # Load suite data
                test_suite = load_test_suite(suite_path)
                suite_dicts = [asdict(tc) for tc in test_suite.test_cases]
                
                # Convert results to dicts
                responses = [asdict(r) for r in result.results]
                
                # Create config dict
                config_dict = {
                    'suite': suite_name,
                    'model': model_id,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'run_id': result.run_id,
                }
                
                attestation = attester.create_attestation(
                    run_id=result.run_id,
                    config=config_dict,
                    suite_data=suite_dicts,
                    responses=responses
                )
            
            st.success("Attestation Generated")
            
            # Display attestation info
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Hashes:**")
                st.code(f"""Manifest: {attestation.manifest_hash[:24]}...
Config:   {attestation.config_hash[:24]}...
Suite:    {attestation.suite_hash[:24]}...
Responses:{attestation.responses_hash[:24]}...""")
            
            with col2:
                st.markdown("**Environment:**")
                env = attestation.environment
                git_info = f"Git: {env.git_commit}" + (" (dirty)" if env.git_dirty else "") if env.git_commit else "Git: N/A"
                st.code(f"""Python: {env.python_version}
OS: {env.os_name}
{git_info}
Meridian: {attestation.meridian_version}""")
            
            st.info(f"Bundle saved to: data/results/{result.run_id}/")
            st.caption("Verify with: `python -m meridian.cli verify --id " + result.run_id + "`")
        
        st.markdown("---")
        st.markdown(f"**Run ID:** `{result.run_id}`")
        st.markdown("View detailed results in the **Results** page.")
        
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        with st.expander("See full error"):
            st.code(traceback.format_exc())

