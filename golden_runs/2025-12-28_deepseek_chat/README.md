# Golden Run: 2025-12-28 DeepSeek Chat

This golden run establishes the reproducible baseline for Meridian v0.4.0.

## Run Details

| Field | Value |
|-------|-------|
| **Run ID** | `golden_2025-12-28_deepseek_chat` |
| **Model** | DeepSeek Chat (`deepseek_chat`) |
| **Suite** | `Meridian_core_50` |
| **Date** | 2025-12-28 |
| **Commit** | v0.4.0 |

## Configuration

```json
{
  "temperature": 0.0,
  "max_tokens": 256,
  "device": "cpu"
}
```

## Results Summary

| Metric | Value |
|--------|-------|
| **Accuracy** | 58% |
| **95% CI** | [48%, 68%] |
| **Total Tests** | 50 |
| **Passed** | 29 |
| **Failed** | 21 |

## Breakdown by Category

| Category | Accuracy | Description |
|----------|----------|-------------|
| RAG / Retrieval | 80% | Long-context document retrieval |
| Edge Cases | 75% | Ambiguous inputs, malformed data |
| Code Analysis | 60% | Debug, review, optimize |
| Multi-step Reasoning | 60% | Logic chains, dependencies |
| Document Processing | 40% | Strict format extraction |
| Business Calculations | 30% | Financial metrics |

## Reproduction

```bash
# Exact reproduction
python -m meridian.cli run \
  --suite Meridian_core_50 \
  --model deepseek_chat \
  --temperature 0.0 \
  --output results/reproduction_check.jsonl

# Verify against this golden run
python -m meridian.cli compare \
  --run-a golden_runs/2025-12-28_deepseek_chat/run_id.txt \
  --run-b <your_run_id>
```

## Expected Variance

Results should be within **±5%** due to:
- API-side sampling (even at temperature 0)
- Token ordering variations
- Model updates by provider

## Files

- `summary.json` — Complete results with breakdown
- `run_id.txt` — Run identifier for comparison

## Verification Checksum

```bash
# Verify suite integrity
sha256sum suites/Meridian_core_50.jsonl
# Expected: <checksum>
```
