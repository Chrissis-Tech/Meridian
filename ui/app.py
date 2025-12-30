"""
Meridian UI - Main Application
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.layout import inject_global_style, pills_bar, section_title, divider, callout

st.set_page_config(
    page_title="Meridian",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_style()

# Hero
st.markdown("# Meridian")
st.markdown('<div class="ce-subtitle">Rigorous LLM evaluation for production systems</div>', unsafe_allow_html=True)

# Quick summary pills
st.markdown(pills_bar([
    "18 test suites",
    "250+ test cases",
    "Statistical confidence",
    "Tamper-evident runs"
]), unsafe_allow_html=True)

st.markdown(divider(), unsafe_allow_html=True)

# Value prop
st.markdown("""
**Why Meridian:**

Most LLM benchmarks report 90%+ accuracy on synthetic tasks. That tells you nothing about production.

Meridian tests on **real-world scenarios** and provides **cryptographically verifiable** results.
The same model scoring 95% on benchmarks scores 58% here. That's not a flaw — that's production reality.
""")

st.markdown(divider(), unsafe_allow_html=True)

# Workflow
st.markdown(section_title("Workflow"), unsafe_allow_html=True)

st.markdown("""
| Step | Page | Description |
|------|------|-------------|
| 1 | **Playground** | Test individual prompts before committing to a suite |
| 2 | **Run Suite** | Execute evaluations with optional attestation |
| 3 | **Results** | View detailed per-test results and scores |
| 4 | **Compare** | A/B test models or configurations |
| 5 | **Attestation** | Verify, export, and import evaluation bundles |
| 6 | **Explain** | Interpretability analysis (attention, logits) |
""")

st.markdown(divider(), unsafe_allow_html=True)

# Recommended suites
st.markdown(section_title("Recommended Suites"), unsafe_allow_html=True)

suites = [
    ("rag_evaluation", "RAG / Retrieval", "Extract facts from long documents", "20 tests"),
    ("code_analysis", "Code Analysis", "Debug, review, and analyze code", "20 tests"),
    ("business_analysis", "Business Logic", "Financial calculations and metrics", "20 tests"),
    ("edge_cases", "Edge Cases", "Ambiguous inputs, malformed data", "20 tests"),
]

cols = st.columns(2)
for i, (id, name, desc, size) in enumerate(suites):
    with cols[i % 2]:
        st.markdown(f"""
        <div class="ce-card" style="margin-bottom: 12px;">
            <div style="font-weight: 600; color: #111827; font-size: 14px;">{name}</div>
            <div class="ce-muted" style="margin-top: 4px;">{desc}</div>
            <div class="ce-muted" style="margin-top: 8px; font-size: 11px;">{size} | <code>{id}</code></div>
        </div>
        """, unsafe_allow_html=True)

st.markdown(divider(), unsafe_allow_html=True)

# Model support
st.markdown(section_title("Supported Models"), unsafe_allow_html=True)

st.markdown("""
| Provider | Models | Context | Setup |
|----------|--------|---------|-------|
| **DeepSeek** | deepseek_chat, deepseek_coder | 32K | `DEEPSEEK_API_KEY` |
| **OpenAI** | gpt-3.5, gpt-4, gpt-4-turbo | 16K-128K | `OPENAI_API_KEY` |
| **Mistral** | mistral-small, mistral-medium | 32K | `MISTRAL_API_KEY` |
| **Groq** | llama-70b, mixtral-8x7b | 32K | `GROQ_API_KEY` |
| **Together** | llama-2-70b, qwen-72b | 4K-32K | `TOGETHER_API_KEY` |
| **Local** | distilgpt2, flan-t5 | 1K-2K | PyTorch |
""")

st.markdown(divider(), unsafe_allow_html=True)

# Attestation feature
st.markdown(section_title("Tamper-Evident Attestation"), unsafe_allow_html=True)

st.markdown("""
Every evaluation can generate a **cryptographically verifiable bundle**:

- **SHA256 hashing** of all artifacts (responses, config, suite)
- **Environment capture** (Python version, OS, git commit)
- **Portable export** as ZIP for sharing or archival
- **Tamper detection** - any modification fails verification

Use the **Attestation** page to verify, export, or import bundles.
""")

st.markdown(divider(), unsafe_allow_html=True)

# Evidence
st.markdown(section_title("Real Results (December 2025)"), unsafe_allow_html=True)

st.markdown(callout(
    "DeepSeek Chat on production suites: <b>58% average accuracy</b>. "
    "RAG: 80% | Edge Cases: 75% | Code: 60% | Business: 30%. "
    "These are honest results with confidence intervals. "
    "See the Results page for detailed breakdowns."
), unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Workflow")
    st.markdown("""
    **1. Test**
    - [Playground](./6_Playground)
    
    **2. Evaluate**
    - [Run Suite](./1_Run_Suite)
    - [Results](./2_Results)
    
    **3. Analyze**
    - [Compare](./3_Compare)
    - [Explain](./4_Explain)
    
    **4. Verify**
    - [Attestation](./7_Attestation)
    
    **5. Learn**
    - [About](./5_About)
    """)
    
    st.markdown("---")
    st.markdown("### Meridian v0.4.0")
    st.caption("MIT License")
    st.caption("[GitHub](https://github.com/Chrissis-Tech/Meridian) | [Docs](docs/)")

