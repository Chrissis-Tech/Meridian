# Provider Certification

Meridian includes a comprehensive certification system to verify that model adapters work correctly before using them in production evaluations.

## Quick Start

```bash
# Certify a provider
python -m meridian.cli certify --model deepseek_chat --save

# Output includes:
# - Test results (14 tests)
# - Score (0-100)
# - Badge hash (for verification)
# - JSON report and SVG badge
```

## Certified Providers (December 2025)

| Provider | Score | Tests | Latency | Badge |
|----------|-------|-------|---------|-------|
| DeepSeek Chat | 100% | 14/14 | 1819ms | ![deepseek](https://img.shields.io/badge/meridian-certified%20100%25-brightgreen) |
| OpenAI GPT-4 | 100% | 14/14 | 864ms | ![gpt4](https://img.shields.io/badge/meridian-certified%20100%25-brightgreen) |
| Mistral Medium | 100% | 14/14 | 1715ms | ![mistral](https://img.shields.io/badge/meridian-certified%20100%25-brightgreen) |

## 14 Certification Tests

### Basic Tests (7)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| `connectivity` | API is reachable | Returns any response |
| `basic_generation` | Model can do basic math | "2+2" returns "4" |
| `temperature_zero` | Temperature 0 works | Config accepted, response generated |
| `determinism` | Same input = same output | Two identical calls match |
| `latency` | Reasonable response time | Average < 30 seconds |
| `max_tokens` | Token limit respected | Output stays within limit |
| `error_handling` | Graceful error handling | No crashes on edge cases |

### Advanced Tests (7)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| `system_prompt` | Follows instructions | Responds to "say only PINEAPPLE" correctly |
| `json_mode` | Generates valid JSON | Returns parseable JSON with expected keys |
| `context_window` | Handles long context | 2K+ tokens processed correctly |
| `unicode_handling` | Non-ASCII characters | Handles Japanese, Chinese, Arabic, etc. |
| `empty_prompt` | Minimal input | Handles single character prompt |
| `long_output` | Extended generation | Can generate 500+ tokens |
| `special_characters` | Symbol handling | Processes @#$%&*<>{} correctly |

## Scoring

| Score | Status | Meaning |
|-------|--------|---------|
| 80-100% | CERTIFIED | Adapter is production-ready |
| 50-79% | PARTIAL | Some features may not work |
| 0-49% | FAILED | Adapter has critical issues |

## CLI Reference

```bash
# Basic certification
python -m meridian.cli certify --model <model_id>

# Save report and badge
python -m meridian.cli certify --model <model_id> --save

# Custom output directory
python -m meridian.cli certify --model <model_id> --save --output ./my_certs/
```

## Output Files

When using `--save`, the following files are created:

```
certifications/
├── deepseek_chat_20251230.json      # Full test results
└── deepseek_chat_20251230_badge.svg # Embeddable badge
```

### JSON Report Structure

```json
{
  "provider_id": "deepseek",
  "model_id": "deepseek_chat",
  "timestamp": "2025-12-30T08:43:52Z",
  "tests": [
    {
      "test_name": "connectivity",
      "passed": true,
      "message": "API connection successful",
      "latency_ms": 1234.5
    }
  ],
  "overall_passed": true,
  "score": 100,
  "badge_hash": "78cf88f0be4c0fe0",
  "environment": {
    "python_version": "3.14.0",
    "os_name": "Windows",
    "meridian_version": "0.4.0"
  }
}
```

## Badge Hash Verification

Each certification generates a unique `badge_hash` based on:
- Provider ID
- Model ID
- Timestamp
- Score

This hash can be used to verify that a certification is authentic and hasn't been modified.

## Adding New Tests

To add custom certification tests, extend `CertificationSuite` in `meridian/certification/tests.py`:

```python
def _test_my_custom_test(self):
    """Test N: Description."""
    test_name = "my_custom_test"
    
    try:
        result = self.adapter.generate("test prompt", None)
        
        if some_condition(result):
            self.results.append(CertificationResult(
                test_name=test_name,
                passed=True,
                message="Test passed"
            ))
        else:
            self.results.append(CertificationResult(
                test_name=test_name,
                passed=False,
                message="Test failed: reason"
            ))
    except Exception as e:
        self.results.append(CertificationResult(
            test_name=test_name,
            passed=False,
            message=f"Error: {str(e)[:50]}"
        ))
```

Then add the test to `run_all()`:

```python
def run_all(self):
    # ... existing tests ...
    self._test_my_custom_test()
```

## Why Certification Matters

1. **Adapter Reliability**: Ensures adapters work before running expensive evaluations
2. **Reproducibility**: Verifies determinism at temperature 0
3. **Edge Case Handling**: Tests Unicode, special characters, context limits
4. **Production Readiness**: Confirms JSON mode, long outputs, instruction following

## Comparison with Other Tools

| Feature | Meridian | Promptfoo | LangSmith |
|---------|----------|-----------|-----------|
| Provider certification | Yes | No | No |
| 14 standardized tests | Yes | No | No |
| Badge generation | Yes | No | No |
| Determinism verification | Yes | Partial | No |
| JSON mode testing | Yes | No | No |
| Context window testing | Yes | No | No |
| Open source | Yes | Yes | No |

---

# Suite Certification

In addition to provider certification, Meridian allows users to certify their **evaluation runs** and generate verifiable badges proving their results.

## Quick Start

```bash
# 1. Run an evaluation with attestation
python -m meridian.cli run --suite rag_evaluation --model deepseek_chat --attest

# 2. Certify the run to get a badge
python -m meridian.cli certify-run --id <run_id> --save
```

## Example Output

```
==================================================
Suite: rag_evaluation
Model: deepseek_chat
Accuracy: 80.0%
Tests: 8/10
Verified: Yes
Attestation hash: e73a0153f658e9d5
Badge hash: 11d5dd5c730bd205
==================================================

Badge: ![rag_evaluation](https://img.shields.io/badge/rag%20evaluation-80%25%20verified-brightgreen)

Report saved: suite_certifications/rag_evaluation_deepseek_chat_82b5835d.json
Badge saved: suite_certifications/rag_evaluation_deepseek_chat_82b5835d_badge.svg
```

## CLI Reference

```bash
# Certify an attested run
python -m meridian.cli certify-run --id <run_id>

# Save report and badge
python -m meridian.cli certify-run --id <run_id> --save

# Custom output directory
python -m meridian.cli certify-run --id <run_id> --save --output ./my_badges/
```

## Output Files

```
suite_certifications/
├── rag_evaluation_deepseek_chat_82b5835d.json      # Full report
└── rag_evaluation_deepseek_chat_82b5835d_badge.svg # SVG badge
```

### JSON Report Structure

```json
{
  "run_id": "run_20251230_030301_82b5835d",
  "suite_name": "rag_evaluation",
  "model_id": "deepseek_chat",
  "accuracy": 80.0,
  "passed_tests": 8,
  "total_tests": 10,
  "timestamp": "2025-12-30T09:03:01",
  "attestation_hash": "e73a0153f658e9d5",
  "badge_hash": "11d5dd5c730bd205",
  "verified": true
}
```

## Badge Colors

| Accuracy | Color | Example |
|----------|-------|---------|
| 80-100% | Green | ![80%](https://img.shields.io/badge/suite-80%25%20verified-brightgreen) |
| 60-79% | Yellow-green | ![65%](https://img.shields.io/badge/suite-65%25%20verified-yellowgreen) |
| 40-59% | Orange | ![50%](https://img.shields.io/badge/suite-50%25%20verified-orange) |
| 0-39% | Red | ![30%](https://img.shields.io/badge/suite-30%25%20verified-red) |

## Verified vs Unverified

- **Verified**: The run has valid attestation (tamper-evident)
- **Unverified**: The run exists but attestation failed verification

Badges show `verified` or `unverified` status:
- ![verified](https://img.shields.io/badge/suite-80%25%20verified-brightgreen)
- ![unverified](https://img.shields.io/badge/suite-80%25%20unverified-orange)

## Use Cases

1. **Team Dashboards**: Embed badges in internal wikis showing model performance
2. **CI/CD**: Generate badges as artifacts after automated evaluation runs
3. **Client Reports**: Prove evaluation results with verifiable badges
4. **Model Cards**: Include certified accuracy in model documentation

## Complete Workflow

```bash
# Step 1: Run evaluation with attestation
python -m meridian.cli run --suite business_analysis --model openai_gpt4 --attest
# Output: Attestation bundle: results/run_20251230_093000_abc12345/

# Step 2: Verify the attestation
python -m meridian.cli verify --id run_20251230_093000_abc12345
# Output: ✓ Attestation valid

# Step 3: Certify and get badge
python -m meridian.cli certify-run --id run_20251230_093000_abc12345 --save
# Output: Badge: ![business_analysis](https://.../badge/...)

# Step 4: Export for sharing
python -m meridian.cli export --id run_20251230_093000_abc12345 --format zip
# Output: Exported to: run_20251230_093000_abc12345.zip
```

## Summary: Provider vs Suite Certification

| Aspect | Provider Certification | Suite Certification |
|--------|------------------------|---------------------|
| Command | `certify --model` | `certify-run --id` |
| Purpose | Verify adapter works | Prove evaluation results |
| Tests | 14 standardized tests | Actual suite test cases |
| Output | Provider reliability score | Accuracy + verification status |
| Who uses it | Meridian maintainers | End users |
| Badge shows | "certified 100%" | "80% verified" |
