# GPT-4 Golden Runs - December 30, 2025

Cryptographically verified evaluation results for OpenAI GPT-4.

## Results Summary

| Suite | Accuracy | 95% CI | Latency |
|-------|----------|--------|---------|
| Document Processing | 80% | [50%, 100%] | 2765ms |
| Code Analysis | 50% | [20%, 80%] | 1872ms |
| Multi-step Reasoning | 50% | [20%, 80%] | 3829ms |
| RAG Evaluation | 40% | [10%, 70%] | 2334ms |
| Business Analysis | 20% | [0%, 50%] | 5482ms |

**Average Accuracy: 48%**

## Environment

- Model: OpenAI GPT-4
- Python: 3.14.0
- Git: 474de9baa139
- Date: December 30, 2025

## Verification

```bash
python -m meridian.cli import --bundle rag_evaluation.zip
python -m meridian.cli verify --id run_20251230_002212_cb4876bb
```
