"""
Meridian UI - About Page
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="About - Meridian", page_icon="M", layout="wide")

st.title("About Meridian")

st.markdown("""
## Why This Framework Exists

Most LLM benchmarks report 90%+ accuracy on carefully selected tasks.
These numbers create false confidence before production deployment.

**Meridian takes a different approach:**

- Test on **real production scenarios**, not sanitized demos
- Report **honest results** with statistical confidence intervals
- Provide **cryptographic verification** of evaluation results
- Enable **meaningful comparison** between models
- Prevent **regressions** before they reach users

---

## Methodology

### Evaluation Philosophy

We design test suites that reveal **actual capability limits**, not best-case performance:

| Category | What We Test | Why It Matters |
|----------|-------------|----------------|
| **RAG / Retrieval** | Extract facts from long documents | Real enterprise use case |
| **Edge Cases** | Ambiguous inputs, malformed data | Production traffic is messy |
| **Business Logic** | Financial calculations, metrics | Common failure mode |
| **Code Analysis** | Debug, review, optimize | Developer tool quality |
| **Security** | Prompt injection, jailbreaks | Safety-critical for deployment |

### Statistical Rigor

Every evaluation includes:

- **95% Confidence Intervals** — Know the uncertainty in your measurements
- **Bootstrap Estimation** — Robust even with small sample sizes (n=10-50)
- **Effect Size** — Understand if differences are practically meaningful
- **Regression Detection** — Statistical tests for A/B comparison

### Tamper-Evident Attestation

Every run can generate a **cryptographically verifiable bundle**:

- SHA256 hashing of all artifacts (responses, config, suite)
- Environment capture (Python version, OS, git commit)
- Portable export as ZIP for sharing or archival
- Any modification to results fails verification

---

## Real Results (December 2025)

Tested with DeepSeek Chat on production-grade suites:

| Task Type | Accuracy | Interpretation |
|-----------|----------|----------------|
| RAG / Document Retrieval | 80% | Handles context well |
| Edge Cases & Ambiguity | 75% | Manages uncertainty |
| Code Analysis | 60% | Basic debugging OK |
| Multi-step Reasoning | 60% | Simple chains work |
| Document Processing | 40% | Strict formats fail |
| Business Calculations | 30% | Financial math weak |

**Average on production tasks: 58%**

---

## Using Your Own Data

Create custom test suites with your own prompts and expected answers:

1. **Copy the template**: `suites/_template.jsonl`
2. **Add your test cases** in JSONL format
3. **Run**: `python -m meridian.cli run --suite your_suite --model deepseek_chat`
4. **Attest**: Add `--attest` flag for verifiable bundles

### Example Test Case

```json
{"id": "MY-001", "prompt": "What is 2+2?", "expected": {"type": "contains", "required_words": ["4"]}}
```

### Answer Types

| Type | Use Case | Example |
|------|----------|---------|
| `contains` | Required words in output | `["answer", "42"]` |
| `exact` | Exact match | `"APPROVED"` |
| `regex` | Pattern match | `"\\d{4}-\\d{2}-\\d{2}"` |
| `length` | Max sentences/words | `{"max_sentences": 3}` |
| `json_schema` | Structured output | Full JSON Schema |

---

## CLI Commands

```bash
# Run evaluation
python -m meridian.cli run --suite rag_evaluation --model deepseek_chat

# Run with attestation
python -m meridian.cli run --suite rag_evaluation --model deepseek_chat --attest

# Verify attestation
python -m meridian.cli verify --id <run_id>

# Replay a run
python -m meridian.cli replay --id <run_id> --mode drift

# Export/Import bundles
python -m meridian.cli export --id <run_id>
python -m meridian.cli import --bundle <file.zip>
```

---

## How to Cite

If you use Meridian in your research or product evaluation:

```bibtex
@software{meridian,
  title = {Meridian: Rigorous LLM Evaluation Framework},
  author = {Chrissis-Tech},
  year = {2025},
  url = {https://github.com/Chrissis-Tech/Meridian},
  version = {0.4.0}
}
```

---

## Links

- **Repository**: [github.com/Chrissis-Tech/Meridian](https://github.com/Chrissis-Tech/Meridian)
- **Documentation**: [README](https://github.com/Chrissis-Tech/Meridian#readme)
- **Issues**: [Report bugs or request features](https://github.com/Chrissis-Tech/Meridian/issues)
- **License**: MIT

---

## Version

**Meridian v0.4.0** — December 2025

*Honest evaluation for LLMs in production. Cryptographically verifiable.*
""")

# Sidebar info
with st.sidebar:
    st.markdown("### Quick Facts")
    st.markdown("""
    - **18 test suites**
    - **250+ test cases**
    - **6 provider families**
    - **17+ models**
    - **Tamper-evident attestation**
    """)
    
    st.markdown("---")
    st.markdown("### Model Support")
    st.markdown("""
    - DeepSeek Chat/Coder
    - OpenAI GPT-3.5/4
    - Mistral Small/Medium
    - Groq (Llama, Mixtral)
    - Together AI
    - Local (DistilGPT-2, Flan-T5)
    """)

