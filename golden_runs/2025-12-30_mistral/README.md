# Mistral Medium Golden Runs - December 30, 2025

Cryptographically verified evaluation results for Mistral Medium.

## Results Summary

| Suite | Accuracy | 95% CI | Latency |
|-------|----------|--------|---------|
| Code Analysis | 80% | [60%, 100%] | 2965ms |
| RAG Evaluation | 60% | [30%, 90%] | 3626ms |
| Multi-step Reasoning | 40% | [10%, 70%] | 2901ms |
| Document Processing | 30% | [0%, 60%] | 3844ms |
| Business Analysis | 20% | [0%, 50%] | 3874ms |

**Average Accuracy: 46%**

## Environment

- Model: Mistral Medium (mistral-medium-latest)
- Python: 3.14.0
- Git: 1e81f2368e59
- Date: December 30, 2025

## Verification

```bash
python -m meridian.cli import --bundle rag_evaluation.zip
python -m meridian.cli verify --id run_20251230_005839_a1e577f6
```
