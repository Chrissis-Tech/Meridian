"""
Meridian UI - Playground Page
Test your own prompts across models with LLM Judge evaluation
"""

import streamlit as st
import pandas as pd
import time
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.layout import inject_global_style, kpi_card, pills_bar, section_title, divider, callout

st.set_page_config(page_title="Playground - Meridian", page_icon="M", layout="wide")
inject_global_style()

st.markdown("# Playground")
st.markdown('<div class="ce-subtitle">Test your own prompts across models. Compare quality, speed, and cost.</div>', unsafe_allow_html=True)

# Model context limits
MODEL_LIMITS = {
    # DeepSeek
    "deepseek_chat": {"context": 32000, "chars": 128000, "name": "DeepSeek Chat"},
    "deepseek_coder": {"context": 16000, "chars": 64000, "name": "DeepSeek Coder"},
    # OpenAI
    "openai_gpt35": {"context": 16000, "chars": 64000, "name": "GPT-3.5 Turbo"},
    "openai_gpt4": {"context": 8000, "chars": 32000, "name": "GPT-4"},
    "openai_gpt4_turbo": {"context": 128000, "chars": 512000, "name": "GPT-4 Turbo"},
    # Mistral
    "mistral_small": {"context": 32000, "chars": 128000, "name": "Mistral Small"},
    "mistral_medium": {"context": 32000, "chars": 128000, "name": "Mistral Medium"},
    "mistral_large": {"context": 32000, "chars": 128000, "name": "Mistral Large"},
    # Groq (ultra-fast)
    "groq_llama70b": {"context": 4096, "chars": 16000, "name": "Groq Llama 70B"},
    "groq_llama8b": {"context": 8192, "chars": 32000, "name": "Groq Llama 8B"},
    "groq_mixtral": {"context": 32768, "chars": 128000, "name": "Groq Mixtral"},
    # Together
    "together_llama70b": {"context": 4096, "chars": 16000, "name": "Together Llama 70B"},
    "together_mixtral": {"context": 32768, "chars": 128000, "name": "Together Mixtral"},
    "together_codellama": {"context": 16000, "chars": 64000, "name": "Together CodeLlama"},
}

# Pricing per 1M tokens
PRICING = {
    # DeepSeek
    "deepseek_chat": {"input": 0.14, "output": 0.28},
    "deepseek_coder": {"input": 0.14, "output": 0.28},
    # OpenAI
    "openai_gpt35": {"input": 0.50, "output": 1.50},
    "openai_gpt4": {"input": 30.0, "output": 60.0},
    "openai_gpt4_turbo": {"input": 10.0, "output": 30.0},
    # Mistral
    "mistral_small": {"input": 1.0, "output": 3.0},
    "mistral_medium": {"input": 2.7, "output": 8.1},
    "mistral_large": {"input": 4.0, "output": 12.0},
    # Groq (very cheap)
    "groq_llama70b": {"input": 0.7, "output": 0.8},
    "groq_llama8b": {"input": 0.05, "output": 0.08},
    "groq_mixtral": {"input": 0.24, "output": 0.24},
    # Together
    "together_llama70b": {"input": 0.9, "output": 0.9},
    "together_mixtral": {"input": 0.6, "output": 0.6},
    "together_codellama": {"input": 0.78, "output": 0.78},
}

# Initialize session state
if "playground_results" not in st.session_state:
    st.session_state.playground_results = []
if "deepseek_key" not in st.session_state:
    st.session_state.deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
if "openai_key" not in st.session_state:
    st.session_state.openai_key = os.getenv("OPENAI_API_KEY", "")
if "mistral_key" not in st.session_state:
    st.session_state.mistral_key = os.getenv("MISTRAL_API_KEY", "")
if "groq_key" not in st.session_state:
    st.session_state.groq_key = os.getenv("GROQ_API_KEY", "")
if "together_key" not in st.session_state:
    st.session_state.together_key = os.getenv("TOGETHER_API_KEY", "")

# Sidebar
with st.sidebar:
    st.markdown("### API Configuration")
    
    st.markdown("**DeepSeek** (recommended)")
    deepseek_key = st.text_input(
        "DeepSeek API Key",
        value=st.session_state.deepseek_key,
        placeholder="sk-xxxxxxxxxxxxxxxx",
        label_visibility="collapsed",
        autocomplete="off"
    )
    if deepseek_key and len(deepseek_key) >= 20 and deepseek_key.startswith("sk-"):
        st.session_state.deepseek_key = deepseek_key
        os.environ["DEEPSEEK_API_KEY"] = deepseek_key
        st.caption("✓ Connected")
    elif deepseek_key:
        st.caption("⚠ Invalid format")
    
    st.markdown("**OpenAI**")
    openai_key = st.text_input(
        "OpenAI API Key",
        value=st.session_state.openai_key,
        placeholder="sk-proj-xxxxxxxx",
        label_visibility="collapsed",
        autocomplete="off"
    )
    if openai_key and len(openai_key) >= 20:
        st.session_state.openai_key = openai_key
        os.environ["OPENAI_API_KEY"] = openai_key
        st.caption("✓ Connected")
    elif openai_key:
        st.caption("⚠ Invalid format")
    
    st.markdown("**Mistral AI**")
    mistral_key = st.text_input(
        "Mistral API Key",
        value=st.session_state.mistral_key,
        placeholder="xxxxxxxxxxxxxxxx",
        label_visibility="collapsed",
        autocomplete="off"
    )
    if mistral_key and len(mistral_key) >= 20:
        st.session_state.mistral_key = mistral_key
        os.environ["MISTRAL_API_KEY"] = mistral_key
        st.caption("✓ Connected")
    elif mistral_key:
        st.caption("⚠ Invalid format")
    
    st.markdown("**Groq** (ultra-fast)")
    groq_key = st.text_input(
        "Groq API Key",
        value=st.session_state.groq_key,
        placeholder="gsk_xxxxxxxxxxxxxxxx",
        label_visibility="collapsed",
        autocomplete="off"
    )
    if groq_key and len(groq_key) >= 20:
        st.session_state.groq_key = groq_key
        os.environ["GROQ_API_KEY"] = groq_key
        st.caption("✓ Connected")
    elif groq_key:
        st.caption("⚠ Invalid format")
    
    st.markdown("**Together AI**")
    together_key = st.text_input(
        "Together API Key",
        value=st.session_state.together_key,
        placeholder="xxxxxxxxxxxxxxxx",
        label_visibility="collapsed",
        autocomplete="off"
    )
    if together_key and len(together_key) >= 20:
        st.session_state.together_key = together_key
        os.environ["TOGETHER_API_KEY"] = together_key
        st.caption("✓ Connected")
    elif together_key:
        st.caption("⚠ Invalid format")
    
    st.markdown("---")
    st.markdown("### Pricing")
    st.caption("DeepSeek: $0.14/$0.28 per 1M")
    st.caption("Groq: $0.05-0.70 per 1M (fast!)")
    st.caption("Together: $0.60-0.90 per 1M")
    st.caption("Mistral: $1-12 per 1M")
    st.caption("OpenAI: $0.50-60 per 1M")

st.markdown(divider(), unsafe_allow_html=True)

# Available models
available_models = []
if st.session_state.deepseek_key and len(st.session_state.deepseek_key) >= 20:
    available_models.extend(["deepseek_chat", "deepseek_coder"])
if st.session_state.openai_key and len(st.session_state.openai_key) >= 20:
    available_models.extend(["openai_gpt35", "openai_gpt4", "openai_gpt4_turbo"])
if st.session_state.mistral_key and len(st.session_state.mistral_key) >= 20:
    available_models.extend(["mistral_small", "mistral_medium", "mistral_large"])
if st.session_state.groq_key and len(st.session_state.groq_key) >= 20:
    available_models.extend(["groq_llama70b", "groq_llama8b", "groq_mixtral"])
if st.session_state.together_key and len(st.session_state.together_key) >= 20:
    available_models.extend(["together_llama70b", "together_mixtral", "together_codellama"])

if not available_models:
    st.markdown(callout("Configure at least one API key in the sidebar to start testing."), unsafe_allow_html=True)
    st.stop()

# === 1. SELECT MODELS ===
st.markdown(section_title("1. Select Models"), unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    selected_models = st.multiselect(
        "Models to compare",
        available_models,
        default=available_models[:2] if len(available_models) >= 2 else available_models,
        format_func=lambda x: MODEL_LIMITS.get(x, {}).get("name", x)
    )

with col2:
    if selected_models:
        min_context = min(MODEL_LIMITS.get(m, {}).get("chars", 64000) for m in selected_models)
        st.caption(f"Max context: {min_context:,} chars")

max_chars = min(MODEL_LIMITS.get(m, {}).get("chars", 64000) for m in selected_models) if selected_models else 64000

st.markdown(divider(), unsafe_allow_html=True)

# === 2. INPUT ===
st.markdown(section_title("2. Input"), unsafe_allow_html=True)

# File upload
st.markdown("**Upload Document (optional)**")
uploaded_file = st.file_uploader(
    "Upload",
    type=["txt", "md", "csv", "json", "py", "js", "html", "xml", "pdf"],
    help="Drag & drop or click. Supported: txt, md, csv, json, py, js, html, xml",
    label_visibility="collapsed"
)

file_content = ""
if uploaded_file:
    try:
        file_content = uploaded_file.read().decode("utf-8")
        if len(file_content) > max_chars:
            st.warning(f"Truncated to {max_chars:,} chars")
            file_content = file_content[:max_chars]
        st.success(f"Loaded: {uploaded_file.name} ({len(file_content):,} chars)")
    except:
        st.error("Could not read file as text")

# Context
context = st.text_area(
    "Context / Document",
    value=file_content,
    placeholder="Paste your document, knowledge base, or context here.",
    height=150,
    max_chars=max_chars,
    help=f"Max {max_chars:,} characters"
)

if context:
    st.caption(f"{len(context):,} / {max_chars:,} chars")

# Prompt
prompt = st.text_area(
    "Prompt / Task",
    placeholder="What do you want the model to do?",
    height=100,
    max_chars=10000
)

st.markdown(divider(), unsafe_allow_html=True)

# === 3. EVALUATION METHOD ===
st.markdown(section_title("3. Evaluation Method"), unsafe_allow_html=True)

eval_method = st.radio(
    "How to evaluate responses",
    ["None (just compare outputs)", "Keyword Check", "LLM Judge"],
    horizontal=True,
    help="LLM Judge uses a separate model to score correctness, completeness, format, and hallucination"
)

eval_config = {}

if eval_method == "Keyword Check":
    eval_config["keywords"] = st.text_input(
        "Required keywords (comma-separated)",
        placeholder="risk(τ), coverage(τ), McNemar, JSON",
        help="Output must contain all these keywords to pass"
    )

elif eval_method == "LLM Judge":
    col1, col2 = st.columns([2, 1])
    with col1:
        eval_config["requirements"] = st.text_area(
            "Requirements to check",
            placeholder="""Examples:
- Must be valid JSON with keys: executive_summary, gates, risk_policy
- Must mention risk(τ) and coverage(τ)
- Must include exactly 4 gates: offline_eval, shadow_traffic, canary, full_rollout
- No hallucinated facts beyond the context""",
            height=120,
            help="Specific requirements the judge will check"
        )
    with col2:
        eval_config["judge_model"] = st.selectbox(
            "Judge model",
            available_models,
            format_func=lambda x: MODEL_LIMITS.get(x, {}).get("name", x),
            help="Model used to evaluate (separate from test models)"
        )
        eval_config["pass_threshold"] = st.slider(
            "Pass threshold",
            50, 100, 70,
            help="Minimum overall score to pass"
        )

st.markdown(divider(), unsafe_allow_html=True)

# === 4. PARAMETERS ===
st.markdown(section_title("4. Parameters"), unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    max_tokens = st.slider("Max output tokens", 100, 4096, 1024)
with col2:
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

st.markdown(divider(), unsafe_allow_html=True)

# === 5. RUN ===
st.markdown(section_title("5. Run"), unsafe_allow_html=True)

can_run = prompt and selected_models
if not can_run:
    st.caption("Enter a prompt and select models")

if st.button("Run Comparison", type="primary", disabled=not can_run, use_container_width=True):
    
    full_prompt = f"{context}\n\n{prompt}" if context else prompt
    results = []
    
    progress = st.progress(0)
    status = st.empty()
    
    for i, model_id in enumerate(selected_models):
        model_name = MODEL_LIMITS.get(model_id, {}).get('name', model_id)
        status.text(f"Running {model_name}...")
        
        try:
            from core.model_adapters import get_adapter
            from core.model_adapters.base import GenerationConfig
            
            adapter = get_adapter(model_id)
            config = GenerationConfig(temperature=temperature, max_tokens=max_tokens)
            
            start = time.time()
            result = adapter.generate(full_prompt, config)
            latency = (time.time() - start) * 1000
            
            input_tokens = getattr(result, 'tokens_in', None) or len(full_prompt) // 4
            output_tokens = getattr(result, 'tokens_out', None) or len(result.output) // 4
            
            pricing = PRICING.get(model_id, {"input": 0, "output": 0})
            cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
            
            # Check if output was truncated
            truncated = output_tokens >= max_tokens - 10
            
            results.append({
                "model": model_id,
                "model_name": model_name,
                "output": result.output,
                "latency_ms": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "eval_result": None,
                "truncated": truncated
            })
            
        except Exception as e:
            results.append({
                "model": model_id,
                "model_name": model_name,
                "output": f"Error: {e}",
                "latency_ms": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0,
                "eval_result": None
            })
        
        progress.progress((i + 1) / len(selected_models))
    
    # === EVALUATION ===
    if eval_method == "Keyword Check" and eval_config.get("keywords"):
        from core.judge import quick_keyword_check
        
        for r in results:
            passed, missing = quick_keyword_check(r["output"], eval_config["keywords"])
            r["eval_result"] = {
                "passed": passed,
                "method": "keyword",
                "missing": missing,
                "score": 100 if passed else (100 - len(missing) * 10)
            }
    
    elif eval_method == "LLM Judge":
        status.text("Running LLM Judge evaluation...")
        from core.judge import evaluate_with_judge
        from core.model_adapters import get_adapter
        
        judge_adapter = get_adapter(eval_config["judge_model"])
        
        for j, r in enumerate(results):
            if r["latency_ms"] > 0:  # Only evaluate successful runs
                verdict = evaluate_with_judge(
                    context=context,
                    prompt=prompt,
                    response=r["output"],
                    requirements=eval_config.get("requirements", ""),
                    judge_adapter=judge_adapter,
                    pass_threshold=eval_config.get("pass_threshold", 70)
                )
                r["eval_result"] = {
                    "passed": verdict.passed,
                    "method": "llm_judge",
                    "score": verdict.overall_score,
                    "correctness": verdict.correctness,
                    "completeness": verdict.completeness,
                    "format": verdict.format_adherence,
                    "no_hallucination": verdict.no_hallucination,
                    "reasoning": verdict.reasoning,
                    "issues": verdict.issues
                }
            progress.progress((len(selected_models) + j + 1) / (len(selected_models) * 2))
    
    status.empty()
    progress.empty()
    st.session_state.playground_results = results

# === DISPLAY RESULTS ===
if st.session_state.playground_results:
    results = st.session_state.playground_results
    
    st.markdown(divider(), unsafe_allow_html=True)
    st.markdown(section_title("Results"), unsafe_allow_html=True)
    
    # Summary cards
    cols = st.columns(len(results))
    for col, r in zip(cols, results):
        with col:
            eval_result = r.get("eval_result")
            if eval_result:
                passed_str = " ✓" if eval_result["passed"] else " ✗"
                tone = "good" if eval_result["passed"] else "bad"
                score = eval_result.get("score", 0)
                hint = f'{score:.0f} pts · ${r["cost_usd"]:.6f}'
            else:
                passed_str = ""
                tone = "neutral"
                hint = f'${r["cost_usd"]:.6f} · {r["output_tokens"]} tokens'
            
            st.markdown(kpi_card(
                f'{r["latency_ms"]/1000:.2f}s{passed_str}',
                r["model_name"],
                hint,
                tone=tone
            ), unsafe_allow_html=True)
    
    # LLM Judge details
    has_judge = any(r.get("eval_result", {}).get("method") == "llm_judge" for r in results)
    if has_judge:
        st.markdown(section_title("Judge Evaluation"), unsafe_allow_html=True)
        
        judge_data = []
        for r in results:
            ev = r.get("eval_result", {})
            if ev.get("method") == "llm_judge":
                judge_data.append({
                    "Model": r["model_name"],
                    "Overall": f'{ev.get("score", 0):.0f}',
                    "Correctness": f'{ev.get("correctness", 0):.0f}',
                    "Completeness": f'{ev.get("completeness", 0):.0f}',
                    "Format": f'{ev.get("format", 0):.0f}',
                    "No Hallucination": f'{ev.get("no_hallucination", 0):.0f}',
                    "Passed": "✓" if ev.get("passed") else "✗"
                })
        
        if judge_data:
            st.dataframe(pd.DataFrame(judge_data), use_container_width=True, hide_index=True)
            
            # Show reasoning and issues
            for r in results:
                ev = r.get("eval_result", {})
                if ev.get("method") == "llm_judge":
                    with st.expander(f"Judge feedback: {r['model_name']}"):
                        st.markdown(f"**Reasoning:** {ev.get('reasoning', 'N/A')}")
                        issues = ev.get("issues", [])
                        if issues:
                            st.markdown("**Issues:**")
                            for issue in issues:
                                st.markdown(f"- {issue}")
    
    # Cost table
    st.markdown(section_title("Cost Breakdown"), unsafe_allow_html=True)
    
    cost_data = pd.DataFrame([{
        "Model": r["model_name"],
        "Latency": f'{r["latency_ms"]/1000:.2f}s',
        "In Tokens": f'{r["input_tokens"]:,}',
        "Out Tokens": f'{r["output_tokens"]:,}',
        "Cost": f'${r["cost_usd"]:.6f}',
        "Per 1K": f'${r["cost_usd"] * 1000:.2f}',
    } for r in results])
    
    st.dataframe(cost_data, use_container_width=True, hide_index=True)
    
    # Outputs in tabs
    st.markdown(section_title("Model Outputs"), unsafe_allow_html=True)
    
    tabs = st.tabs([r["model_name"] for r in results])
    for tab, r in zip(tabs, results):
        with tab:
            if r.get("truncated"):
                st.warning("⚠️ Output was truncated. Increase max tokens for complete response.")
            st.text_area("Output", value=r["output"], height=300, key=f"out_{r['model']}", label_visibility="collapsed")
    
    # Winner summary
    if len(results) > 1:
        st.markdown(divider(), unsafe_allow_html=True)
        
        valid = [r for r in results if r["latency_ms"] > 0]
        if valid:
            fastest = min(valid, key=lambda x: x["latency_ms"])
            cheapest = min(valid, key=lambda x: x["cost_usd"])
            
            summary = f"**Fastest:** {fastest['model_name']} ({fastest['latency_ms']/1000:.2f}s)"
            summary += f" · **Cheapest:** {cheapest['model_name']} (${cheapest['cost_usd']:.6f})"
            
            # Best by eval
            evaluated = [r for r in valid if r.get("eval_result")]
            if evaluated:
                passed = [r for r in evaluated if r["eval_result"]["passed"]]
                if passed:
                    best = max(passed, key=lambda x: x["eval_result"].get("score", 0))
                    summary += f" · **Best quality:** {best['model_name']} ({best['eval_result']['score']:.0f} pts)"
            
            st.markdown(callout(summary), unsafe_allow_html=True)
    
    # Export section
    st.markdown(divider(), unsafe_allow_html=True)
    st.markdown(section_title("Export"), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export as JSON
        import json
        export_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "context": context[:500] + "..." if len(context) > 500 else context,
            "prompt": prompt,
            "models": [r["model"] for r in results],
            "results": [{
                "model": r["model_name"],
                "latency_s": r["latency_ms"] / 1000,
                "cost_usd": r["cost_usd"],
                "tokens_in": r["input_tokens"],
                "tokens_out": r["output_tokens"],
                "passed": r.get("eval_result", {}).get("passed") if r.get("eval_result") else None,
                "score": r.get("eval_result", {}).get("score") if r.get("eval_result") else None,
                "output_preview": r["output"][:200] + "..."
            } for r in results]
        }
        
        st.download_button(
            "📄 Export JSON",
            data=json.dumps(export_data, indent=2),
            file_name="meridian_comparison.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Export as Markdown
        md_report = f"""# Meridian Comparison Report

**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}

## Prompt
```
{prompt[:500]}
```

## Results

| Model | Latency | Cost | Tokens | Passed |
|-------|---------|------|--------|--------|
"""
        for r in results:
            passed_str = "✓" if r.get("eval_result", {}).get("passed") else ("✗" if r.get("eval_result") else "—")
            md_report += f"| {r['model_name']} | {r['latency_ms']/1000:.2f}s | ${r['cost_usd']:.6f} | {r['output_tokens']} | {passed_str} |\n"
        
        md_report += "\n## Winner\n\n"
        if valid:
            md_report += f"- **Fastest:** {fastest['model_name']} ({fastest['latency_ms']/1000:.2f}s)\n"
            md_report += f"- **Cheapest:** {cheapest['model_name']} (${cheapest['cost_usd']:.6f})\n"
        
        st.download_button(
            "📝 Export Markdown",
            data=md_report,
            file_name="meridian_comparison.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    with col3:
        # Save as test case
        if st.button("💾 Save as Test Case", use_container_width=True):
            st.session_state.show_save_dialog = True
    
    # Save dialog
    if st.session_state.get("show_save_dialog"):
        with st.expander("Save as Test Case", expanded=True):
            test_id = st.text_input("Test ID", value="CUSTOM-001", placeholder="e.g., RAG-DEPLOY-001")
            suite_name = st.text_input("Suite name", value="custom_tests", placeholder="e.g., my_rag_tests")
            
            if eval_method == "Keyword Check" and eval_config.get("keywords"):
                expected_type = "contains"
                expected_value = {"required_words": [k.strip() for k in eval_config["keywords"].split(",")]}
            else:
                expected_type = "contains"
                expected_value = {"required_words": []}
            
            if st.button("Save Test Case"):
                import json
                from pathlib import Path
                
                suite_path = Path(f"suites/{suite_name}.jsonl")
                
                # Create header if new file
                if not suite_path.exists():
                    with open(suite_path, "w") as f:
                        header = {"suite_name": suite_name, "description": "Custom test suite", "version": "1.0.0"}
                        f.write(json.dumps(header) + "\n")
                
                # Append test case
                test_case = {
                    "id": test_id,
                    "prompt": (context + "\n\n" + prompt) if context else prompt,
                    "expected": {"type": expected_type, **expected_value},
                    "tags": ["custom", "playground"]
                }
                
                with open(suite_path, "a") as f:
                    f.write(json.dumps(test_case) + "\n")
                
                st.success(f"Saved to `suites/{suite_name}.jsonl`")
                st.session_state.show_save_dialog = False
