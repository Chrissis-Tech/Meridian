"""
Meridian UI - Main Application (Framework Top)
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
    "15 test suites",
    "210+ test cases",
    "Statistical confidence",
    "Regression detection"
]), unsafe_allow_html=True)

st.markdown(divider(), unsafe_allow_html=True)

# Value prop
st.markdown("""
**Why this matters:**

Most LLM benchmarks report 90%+ accuracy. That tells you nothing useful.

Meridian tests on **real production scenarios** and reports **honest results** with confidence intervals.
The same model scoring 95% elsewhere scores 58% here. That's not a flaw — that's the truth.
""")

st.markdown(divider(), unsafe_allow_html=True)

# Quick start
st.markdown(section_title("Quick Start"), unsafe_allow_html=True)

st.markdown("""
1. **Configure** — Add your API key in the sidebar (Run Suite page)
2. **Select** — Choose a suite and model to evaluate  
3. **Run** — Get accuracy, latency, and confidence intervals
4. **Compare** — A/B test different models or versions
""")

st.markdown(divider(), unsafe_allow_html=True)

# Recommended suites
st.markdown(section_title("Recommended Suites"), unsafe_allow_html=True)

suites = [
    ("rag_evaluation", "RAG / Document Retrieval", "Extract facts from long documents", "10 tests"),
    ("code_analysis", "Code Analysis", "Debug, review, and analyze code", "10 tests"),
    ("business_analysis", "Business Logic", "Financial calculations and metrics", "10 tests"),
    ("edge_cases", "Edge Cases", "Ambiguous inputs and malformed data", "12 tests"),
]

cols = st.columns(2)
for i, (id, name, desc, size) in enumerate(suites):
    with cols[i % 2]:
        st.markdown(f"""
        <div class="ce-card" style="margin-bottom: 12px;">
            <div style="font-weight: 600; color: #111827; font-size: 14px;">{name}</div>
            <div class="ce-muted" style="margin-top: 4px;">{desc}</div>
            <div class="ce-muted" style="margin-top: 8px; font-size: 11px;">{size} · <code>{id}</code></div>
        </div>
        """, unsafe_allow_html=True)

st.markdown(divider(), unsafe_allow_html=True)

# Model support
st.markdown(section_title("Supported Models"), unsafe_allow_html=True)

st.markdown("""
| Provider | Models | Context | Requires |
|----------|--------|---------|----------|
| **DeepSeek** | deepseek_chat, deepseek_coder | 32K | `DEEPSEEK_API_KEY` |
| **OpenAI** | gpt-3.5-turbo, gpt-4, gpt-4-turbo | 16K–128K | `OPENAI_API_KEY` |
| **Local** | distilgpt2, flan-t5 | 1K–2K | PyTorch + Transformers |
""")

st.markdown(divider(), unsafe_allow_html=True)

# Evidence
st.markdown(section_title("Real Results (Dec 2025)"), unsafe_allow_html=True)

st.markdown(callout(
    "DeepSeek Chat on production suites: <b>58% average accuracy</b>. "
    "RAG: 80% · Edge Cases: 75% · Code: 60% · Business: 30%. "
    "The same model scores 95% on easy benchmarks. "
    "This is the real capability level for production planning."
), unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Navigation")
    st.markdown("""
    **Test**
    - [Playground](./6_Playground) ← Test your prompts
    - [Run Suite](./1_Run_Suite)
    
    **Analyze**
    - [Results](./2_Results)
    - [Compare](./3_Compare)
    - [Explain](./4_Explain)
    
    **Docs**
    - [About](./5_About)
    """)
    
    st.markdown("---")
    st.markdown("### Meridian v0.3.0")
    st.caption("MIT License · [GitHub](https://github.com/Chrissis-Tech/Meridian)")
