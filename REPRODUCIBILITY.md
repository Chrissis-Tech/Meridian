# Reproducibility Guide

This document describes how to reproduce Meridian evaluation results.

## Prerequisites

```bash
# Python 3.9+
python --version

# Clone repository
git clone https://github.com/Chrissis-Tech/Meridian.git
cd Meridian

# Install dependencies
pip install -e .
```

## Reproducing a Golden Run

### 1. Using the CLI

```bash
# Reproduce the baseline golden run
python -m meridian.cli reproduce --id 2025-12-28_deepseek_chat

# Or run manually with same parameters
python -m meridian.cli run \
    --suite Meridian_core_50 \
    --model deepseek_chat \
    --temperature 0.0 \
    --output results/my_run.jsonl
```

### 2. Verification

Compare your results against the golden run:

```bash
python -m meridian.cli compare \
    --run-a golden_runs/2025-12-28_deepseek_chat/run_id.txt \
    --run-b <your_run_id>
```

Expected variance: Accuracy within 5% due to:
- Random seed differences in tokenization
- Hardware-specific floating point operations
- PyTorch version differences

## Environment Specification

### Hardware (Reference)
- CPU: Intel Core i7 or AMD Ryzen 7
- RAM: 16GB minimum
- GPU: Optional (CUDA 11.7+ for acceleration)

### Software Versions

```
python>=3.9
torch>=2.0.0
transformers>=4.30.0
streamlit>=1.28.0
numpy>=1.24.0
scipy>=1.10.0
click>=8.1.0
```

See `pyproject.toml` for complete dependency list.

### Environment Variables

```bash
# Copy template
cp .env.example .env

# Required settings
DEVICE=cpu          # or cuda, mps
DEFAULT_MODEL=deepseek_chat

# Optional (for API models)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## Reproducing Specific Experiments

### Experiment 1: Baseline GPT-2 Evaluation

```bash
# 1. Run all core suites
for suite in math_short instruction_following consistency; do
    python -m meridian.cli run --suite $suite --model deepseek_chat
done

# 2. Generate report
python scripts/make_demo_data.py
```

### Experiment 2: Model Comparison

```bash
# Requires API keys
python -m meridian.cli run --suite Meridian_core_50 --model openai_gpt35
python -m meridian.cli run --suite Meridian_core_50 --model deepseek_chat
python -m meridian.cli compare --run-a <gpt35_run> --run-b <deepseek_run>
```

### Experiment 3: Interpretability Analysis

```python
from meridian.model_adapters import get_adapter
from meridian.explain import CausalTracer

adapter = get_adapter("deepseek_chat")
tracer = CausalTracer(adapter)

# Analyze critical components
importance = tracer.generate_component_importance(
    prompt="The capital of France is",
    target_token="Paris"
)

print(f"Top layers: {importance.top_layers[:5]}")
print(f"Top heads: {importance.top_heads[:5]}")
```

## Checksum Verification

Verify dataset integrity:

```bash
# Linux/Mac
sha256sum suites/Meridian_core_50.jsonl

# Expected: <checksum will be added after initial release>
```

## Troubleshooting

### Issue: Different accuracy than golden run

1. Verify PyTorch version matches
2. Check DEVICE setting (CPU vs CUDA may differ)
3. Ensure temperature=0.0 for deterministic output
4. Confirm dataset checksum matches

### Issue: Model download fails

```bash
# Pre-download model
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('gpt2')"
```

### Issue: Out of memory

```bash
# Reduce batch size or use CPU
export DEVICE=cpu
```

## Contact

For reproducibility issues, open a GitHub issue with:
- OS and Python version
- PyTorch and transformers versions
- Full error message
- Steps to reproduce
