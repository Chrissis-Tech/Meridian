# Methodology

How Meridian test suites are designed and scored.

## Design Principles

1. **Real-world tasks** - All prompts are based on actual production use cases, not synthetic benchmarks
2. **Clear expected answers** - Every test has an objectively verifiable expected output
3. **Multiple scoring methods** - Exact match, contains, regex, JSON schema, semantic similarity
4. **Transparency** - All suites are public in `suites/` for external review

## Suite Categories

| Category | Source | Difficulty |
|----------|--------|------------|
| RAG Evaluation | Real document retrieval tasks | Medium |
| Code Analysis | Actual debugging scenarios | Medium-Hard |
| Business Analysis | Financial calculations | Hard |
| Document Processing | Format extraction tasks | Medium |
| Multi-step Reasoning | Chain-of-thought problems | Hard |

## Scoring Methods

### Exact Match
```json
{"type": "exact", "value": "APPROVED"}
```
Model output must exactly match the expected value.

### Contains
```json
{"type": "contains", "required_words": ["revenue", "$1.5M"]}
```
Model output must contain all required words/phrases.

### Regex
```json
{"type": "regex", "pattern": "\\d{4}-\\d{2}-\\d{2}"}
```
Model output must match the regex pattern.

### JSON Schema
```json
{"type": "json_schema", "schema": {"type": "object", "required": ["name", "value"]}}
```
Model output must be valid JSON matching the schema.

## Known Limitations

### Potential Biases

1. **Strict scoring** - Some tests require exact formats that capable models might express differently
2. **English-only** - All current suites are in English
3. **Task selection** - Our suites emphasize structured outputs over creative tasks
4. **Sample size** - 10-20 tests per suite may not capture full capability range

### What We Don't Test

- Creative writing quality
- Long-form essay coherence
- Multilingual capabilities
- Real-time conversation flow
- Image/multimodal understanding

## Reproducibility

Every evaluation can be reproduced:

```bash
# Run same suite with same model
python -m meridian.cli run --suite rag_evaluation --model deepseek_chat --attest

# Verify results haven't been tampered
python -m meridian.cli verify --id <run_id>
```

## Example: Failed Response Analysis

### RAG Evaluation - PASS
**Prompt:** Extract the revenue from Q3 2024 from this report...
**Expected:** Contains "1.5M" or "1,500,000"
**Model output:** "The Q3 2024 revenue was $1.5M"
**Result:** PASS

### Business Analysis - FAIL
**Prompt:** Calculate the compound annual growth rate...
**Expected:** Contains "12.4%"
**Model output:** "The CAGR is approximately 12.38%"
**Result:** FAIL (strict threshold, though answer is correct)

This illustrates a known limitation: our scoring can be too strict for numerically close answers.

## How to Validate Our Claims

1. **Review the suites** - All test cases are in `suites/*.jsonl`
2. **Run yourself** - Use your own API keys and verify results
3. **Check attestation** - Import our golden runs and verify hashes
4. **File issues** - If you find unfair tests, report them on GitHub

## Comparison with Standard Benchmarks

| Benchmark | DeepSeek | Notes |
|-----------|----------|-------|
| MMLU | ~85% | Multiple choice, easy |
| HumanEval | ~70% | Code synthesis |
| **Meridian** | ~56% | Production tasks |

The gap between standard benchmarks and Meridian reflects the difference between controlled academic tasks and messy real-world scenarios.

## Contributing

If you believe a test case is unfair:

1. Open an issue with the test ID and reasoning
2. Propose an alternative expected answer
3. We'll review and update if warranted

We actively want external review of our methodology.
