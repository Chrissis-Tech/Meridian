# Use Cases

Real-world scenarios where Meridian provides value.

---

## 1. RAG Evaluation for Contracts

**Problem**: Your RAG system retrieves legal clauses from contracts. How do you know if it's actually finding the right information?

**Solution**: Run the `rag_evaluation` suite against your retrieval pipeline.

```python
from meridian.runner import SuiteRunner
from meridian.types import RunConfig

# Configure for your RAG model
config = RunConfig(temperature=0.0, max_tokens=500)

runner = SuiteRunner("rag_evaluation", "deepseek_chat", config)
result = runner.run()

print(f"Retrieval Accuracy: {result.accuracy:.1%}")
print(f"95% CI: [{result.accuracy_ci[0]:.1%}, {result.accuracy_ci[1]:.1%}]")

# Check specific failure patterns
for test in result.failed_tests:
    print(f"Failed: {test.id} - {test.metadata.get('category')}")
```

**Expected output**:
```
Retrieval Accuracy: 80.0%
95% CI: [70.0%, 90.0%]
Failed: rag_007 - cross_document_reference
Failed: rag_010 - temporal_clause_extraction
```

---

## 2. QA for AI Agents

**Problem**: Your AI agent handles customer support. Before production, you need to verify it doesn't hallucinate or give dangerous advice.

**Solution**: Run the `hallucination_control` and `security_adversarial` suites.

```python
from meridian.runner import run_suite

# Test hallucination resistance
hallucination = run_suite("hallucination_control", "your_agent_model")
security = run_suite("security_adversarial", "your_agent_model")

print(f"Hallucination Control: {hallucination.accuracy:.1%}")
print(f"Security Score: {security.accuracy:.1%}")

# Gate deployment on thresholds
if hallucination.accuracy < 0.80 or security.accuracy < 0.90:
    raise ValueError("Agent fails safety thresholds - do not deploy")
```

**Expected output**:
```
Hallucination Control: 85.0%
Security Score: 95.0%
```

---

## 3. CI/CD Regression Guard

**Problem**: You're updating your model or prompts. How do you catch regressions before they reach production?

**Solution**: Add Meridian to your GitHub Actions workflow.

```yaml
# .github/workflows/llm-regression.yml
name: LLM Regression Check

on:
  pull_request:
    paths:
      - 'prompts/**'
      - 'models/**'

jobs:
  regression-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install Meridian
        run: pip install -e .
      
      - name: Run regression suite
        env:
          DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
        run: |
          python -m meridian.cli check \
            --baseline baselines/production.json \
            --model deepseek_chat \
            --threshold 0.05
```

**Baseline file** (`baselines/production.json`):
```json
{
  "suite": "rag_evaluation",
  "model": "deepseek_chat",
  "accuracy": 0.80,
  "timestamp": "2025-12-28T10:00:00Z"
}
```

**CI output on regression**:
```
❌ REGRESSION DETECTED
   Baseline: 80.0%
   Current:  72.0%
   Δ: -8.0% (threshold: 5%)
   p-value: 0.023 (significant)
```

---

## 4. Safety Smoke Tests

**Problem**: Before deploying any LLM, you need to verify it resists prompt injection and jailbreaks.

**Solution**: Run the `security_adversarial` suite as part of your deployment checklist.

```python
from meridian.runner import run_suite

# Quick safety check
result = run_suite("security_adversarial", "deepseek_chat")

# Detailed breakdown
print(f"Overall Security: {result.accuracy:.1%}")

# Check specific attack categories
categories = {}
for test in result.all_tests:
    cat = test.metadata.get("attack_type", "unknown")
    if cat not in categories:
        categories[cat] = {"passed": 0, "total": 0}
    categories[cat]["total"] += 1
    if test.passed:
        categories[cat]["passed"] += 1

for cat, stats in categories.items():
    pct = stats["passed"] / stats["total"] * 100
    print(f"  {cat}: {stats['passed']}/{stats['total']} ({pct:.0f}%)")
```

**Expected output**:
```
Overall Security: 85.0%
  prompt_injection: 18/20 (90%)
  jailbreak: 8/10 (80%)
  data_extraction: 9/10 (90%)
```

---

## 5. Model Selection

**Problem**: GPT-4 costs 10x more than DeepSeek. Is it worth it for your use case?

**Solution**: Run identical suites against both models and compare statistically.

```python
from meridian.runner import run_suite

# Run same suite on both models
gpt4 = run_suite("business_analysis", "openai_gpt4")
deepseek = run_suite("business_analysis", "deepseek_chat")

# Statistical comparison
print(f"GPT-4:    {gpt4.accuracy:.1%} (CI: {gpt4.accuracy_ci})")
print(f"DeepSeek: {deepseek.accuracy:.1%} (CI: {deepseek.accuracy_ci})")

# Calculate if difference is significant
from meridian.stats import mcnemar_test

p_value = mcnemar_test(gpt4.predictions, deepseek.predictions)
print(f"Difference significant: {p_value < 0.05} (p={p_value:.3f})")

# Cost-effectiveness
gpt4_cost = 10.0  # $/M tokens
deepseek_cost = 0.14  # $/M tokens
accuracy_delta = gpt4.accuracy - deepseek.accuracy

print(f"Accuracy delta: {accuracy_delta:.1%}")
print(f"Cost delta: {gpt4_cost / deepseek_cost:.0f}x")
print(f"Worth it: {accuracy_delta > 0.15}")  # Your threshold
```

**Expected output**:
```
GPT-4:    82.0% (CI: [72.0%, 92.0%])
DeepSeek: 58.0% (CI: [48.0%, 68.0%])
Difference significant: True (p=0.012)
Accuracy delta: 24.0%
Cost delta: 71x
Worth it: True
```

---

## 6. Prompt Engineering Validation

**Problem**: You've rewritten a prompt. Did it actually improve performance?

**Solution**: A/B test with statistical rigor.

```python
from meridian.runner import SuiteRunner
from meridian.types import RunConfig

# Test original prompt
config_v1 = RunConfig(system_prompt="You are a helpful assistant.")
result_v1 = SuiteRunner("instruction_following", "deepseek_chat", config_v1).run()

# Test new prompt
config_v2 = RunConfig(system_prompt="You are a precise assistant. Follow instructions exactly.")
result_v2 = SuiteRunner("instruction_following", "deepseek_chat", config_v2).run()

# Compare
print(f"V1: {result_v1.accuracy:.1%}")
print(f"V2: {result_v2.accuracy:.1%}")
print(f"Improvement: {(result_v2.accuracy - result_v1.accuracy):.1%}")

# Check if statistically significant
from meridian.stats import bootstrap_ci, cohens_d

effect = cohens_d(result_v1.scores, result_v2.scores)
print(f"Effect size: {effect:.2f} ({'small' if effect < 0.5 else 'medium' if effect < 0.8 else 'large'})")
```

---

## 7. Interpretability Debugging

**Problem**: Your model fails on a specific test. Why?

**Solution**: Use causal tracing to identify which components are responsible.

```python
from meridian.model_adapters import get_adapter
from meridian.explain import CausalTracer

adapter = get_adapter("local_distilgpt2")  # Works best with local models
tracer = CausalTracer(adapter)

# Analyze a failing case
prompt = "The contract termination clause is found in section"
expected = "7.3"

importance = tracer.generate_component_importance(
    prompt=prompt,
    target_token=expected
)

print(f"Top 3 most important layers: {importance.top_layers[:3]}")
print(f"Top 3 most important heads: {importance.top_heads[:3]}")

# Visualize attention patterns
tracer.visualize_attention(prompt, save_path="attention_map.png")
```

**Expected output**:
```
Top 3 most important layers: [11, 10, 9]
Top 3 most important heads: [(11, 3), (10, 7), (9, 2)]
Saved attention map to attention_map.png
```

---

## Quick Reference

| Use Case | Suite(s) | Key Metric |
|----------|----------|------------|
| RAG Quality | `rag_evaluation` | Retrieval accuracy |
| Agent Safety | `security_adversarial`, `hallucination_control` | Pass rate |
| CI/CD Gate | Any + baseline | Δ accuracy, p-value |
| Model Selection | Production suites | Cost/accuracy ratio |
| Prompt Testing | `instruction_following` | Effect size |
| Debugging | N/A (interpretability) | Layer importance |

---

## Next Steps

1. **Start simple**: Run `rag_evaluation` on your current model
2. **Establish baseline**: Save results to `baselines/`
3. **Add to CI**: Use the GitHub Actions template above
4. **Iterate**: Run before every deployment

See [examples/](../examples/) for more integration scripts.
