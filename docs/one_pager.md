# Meridian: Rigorous LLM Evaluation

## The Problem

LLM benchmarks report 90%+ accuracy on easy tasks. That tells you nothing about production readiness.

When your model scores 95% on "What is 7+8?", but only 30% on "Calculate this SaaS metric", which number matters for your business?

## The Solution

Meridian provides honest evaluation with production-grade test suites, statistical rigor, and CI/CD integration.

---

## Real Numbers (December 2025)

Tested with DeepSeek Chat on production-grade suites:

| Task Type | Accuracy | What This Means |
|-----------|----------|-----------------|
| RAG / Document Retrieval | 80% | Handles context well |
| Edge Cases & Ambiguity | 75% | Manages uncertainty |
| Code Analysis | 60% | Basic OK, complex fails |
| Multi-step Reasoning | 60% | Simple chains work |
| Document Processing | 40% | Strict formats fail |
| Business Calculations | 30% | Financial math is weak |

**Average on production tasks: 58%**

The same model that gets 95% on easy benchmarks gets 58% here. That's the real capability level.

---

## Why This Matters

| Scenario | Easy Benchmark | Meridian |
|----------|----------------|--------------|
| Model comparison | Both 95% - no difference | GPT-3.5: 58%, GPT-4: 82% |
| Detect regressions | Hidden in noise | 5% drop is visible |
| Justify model spend | No data | Clear ROI calculation |
| Pre-production check | False confidence | Honest assessment |

---

## What It Does

| Capability | Implementation |
|------------|----------------|
| **Evaluate** | 210+ tests across 15 suites |
| **Production Focus** | RAG, code, documents, business logic |
| **Statistics** | Bootstrap CIs, p-values, effect sizes |
| **Interpret** | Attention viz, logit lens, causal tracing |
| **Automate** | CI regression gates, baseline comparison |

---

## Test Suites

**Production-Grade:**
- `rag_evaluation` - Long context retrieval
- `code_analysis` - Debug, review, optimize
- `document_processing` - Extract, summarize, transform
- `multi_step_reasoning` - Logic and dependencies
- `edge_cases` - Ambiguity and error handling
- `business_analysis` - Metrics and calculations

**Standard:**
- `enterprise_prompts`, `instruction_following`, `security_adversarial`, `hallucination_control`

---

## Quick Start

```bash
git clone https://github.com/Chrissis-Tech/Meridian
pip install -e .
python -m core.cli run --suite rag_evaluation --model deepseek_chat
```

---

## Links

- Repository: github.com/Chrissis-Tech/Meridian
- License: MIT

---

*Honest evaluation for LLMs in production. Not inflated benchmarks.*
