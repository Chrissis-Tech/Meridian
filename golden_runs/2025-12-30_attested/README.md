# Attested Golden Runs - December 30, 2025

Cryptographically verified evaluation results for DeepSeek Chat on production suites.

## Results Summary

| Suite | Accuracy | 95% CI | Latency | Run ID |
|-------|----------|--------|---------|--------|
| RAG Evaluation | 80% | [60%, 100%] | 4646ms | run_20251229_235321_49c59741 |
| Code Analysis | 70% | [40%, 100%] | 3670ms | run_20251229_235639_e4436f48 |
| Multi-step Reasoning | 60% | [30%, 90%] | 7265ms | run_20251230_000203_7a67f391 |
| Document Processing | 40% | [10%, 70%] | 3941ms | run_20251229_235854_35584a2b |
| Business Analysis | 30% | [10%, 60%] | 7230ms | run_20251229_235448_f5fd1013 |

**Average Accuracy: 56%**

## Environment

- Model: DeepSeek Chat (deepseek_chat)
- Python: 3.14.0
- Git: 9d22fc232a7d
- Date: December 30, 2025

## Verification

Each bundle is cryptographically signed with SHA256 hashes.

To verify any bundle:

```bash
# Import and verify
python -m meridian.cli import --bundle rag_evaluation.zip

# Or verify existing run
python -m meridian.cli verify --id run_20251229_235321_49c59741
```

## Bundle Contents

Each ZIP contains:
- `manifest.json` - SHA256 hashes of all files
- `config.json` - Exact run configuration
- `suite.jsonl` - Test cases snapshot
- `responses/` - Raw model outputs
- `attestation.json` - Verification metadata

## What This Proves

1. DeepSeek Chat achieves **56% average** on production tasks
2. Performance varies by task type (80% RAG vs 30% business)
3. Results are **tamper-evident** and reproducible
4. Any modification to files will fail verification
