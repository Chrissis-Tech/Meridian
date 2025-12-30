# Custom Test Suites

Create and run your own evaluation suites to test model performance on prompts that matter to your use case.

## Quick Start

### Option 1: UI Upload (Recommended)

1. Go to **Create Suite** page
2. Click **"Create Demo Suite"** for a sample, or upload your own JSONL/CSV
3. Validate and save
4. Go to **Run Suite** → select `[Custom] your_suite_name`

### Option 2: CLI

```bash
# Create suite file
cat > suites/my_suite.jsonl << EOF
{"id": "t1", "input": "What is 2+2?", "expected": "4", "scorer": "exact"}
{"id": "t2", "input": "Capital of France?", "expected": "Paris", "scorer": "contains"}
EOF

# Run
python -m meridian.cli run --suite my_suite --model deepseek_chat --attest
```

## File Format

### Simple Format (UI Upload)

```jsonl
{"id": "math_1", "input": "What is 2+2?", "expected": "4", "scorer": "exact"}
{"id": "translate_1", "input": "Translate 'hello' to Spanish", "expected": "hola", "scorer": "contains"}
{"id": "essay_1", "input": "Write about AI ethics", "scorer": "llm_judge", "rubric": "Score 1-5 on clarity"}
```

### CSV Format

```csv
id,input,expected,scorer
math_1,"What is 2+2?",4,exact
translate_1,"Translate 'hello' to Spanish",hola,contains
```

## Fields

| Field | Required | Description |
|-------|----------|-------------|
| `input` or `prompt` | Yes | The prompt to send to the model |
| `expected` or `answer` | No* | The expected response |
| `id` | No | Unique identifier (auto-generated if missing) |
| `scorer` | No | Scoring method (default: `exact`) |
| `rubric` | Only for llm_judge | Evaluation criteria |
| `tags` | No | Comma-separated tags for filtering |

*Required unless using `llm_judge` scorer.

## Scorers

| Scorer | Description | Example |
|--------|-------------|---------|
| `exact` | Output must exactly match expected | `"4"` matches `"4"` |
| `contains` | Output must contain expected | `"hola"` in `"hola, amigo"` |
| `regex` | Output must match regex pattern | `\d+` matches any number |
| `llm_judge` | LLM evaluates output against rubric | Subjective/open-ended |

## Dev/Holdout Split

Custom suites automatically use **80/20 holdout split** to prevent overfitting:

```
Running custom suite: 8 dev + 2 holdout tests

Completed: Dev 7/8 | Holdout 2/2

+------------------+--------------------+---------------+---------+
| Dev Accuracy     | Holdout Accuracy   | Total Passed  | Latency |
| 87.5%            | 100.0% (+12.5pp)   | 9             | 1230ms  |
+------------------+--------------------+---------------+---------+
```

### Why Holdout Matters

- **Dev set (80%)**: Use this to iterate and improve your prompts
- **Holdout set (20%)**: The "real" score - not seen during development
- **Overfitting warning**: If Holdout << Dev, you may have over-optimized
- **Certification uses Holdout**: The badge shows the holdout accuracy, not dev

The holdout split uses a fixed seed (42) for reproducibility.

## Validation Features

### Leak Detection

If your prompt contains the expected answer:

```
LEAK WARNING: Test 'math_1' input contains expected answer
```

This prevents accidentally "testing" the model by giving it the answer in the question.

### LLM Judge Safety

If using `llm_judge` scorer without a rubric:

```
Test 'essay_1' uses llm_judge but missing 'rubric'
```

### Smart Suggestions

- Empty expected → suggest `llm_judge` with rubric
- Expected looks like JSON → suggest `json_schema` scorer

## Versioning

Suites support automatic versioning:

- First save: `v1`
- Re-upload with same name: automatically becomes `v2`, `v3`, etc.
- Version displayed in UI and reports

## Storage

Custom suites are stored in SQLite database:

```
data/custom_suites.db
```

Export to JSONL via UI for:
- Sharing with teammates
- Version control (git)
- CLI usage

## Complete Workflow

```
1. Create Suite     →  Upload 30 test cases
2. Validate         →  Fix any leak warnings
3. Run Suite        →  Dev: 70% | Holdout: 75%
4. Improve prompts  →  Based on Dev failures
5. Run Again        →  Dev: 85% | Holdout: 78%
6. Certify          →  Get badge: "78% verified"
```

The certification badge uses the **holdout accuracy** - preventing gaming.

## Advanced: Full Schema (for CLI)

For complex test cases with the full runner schema:

```jsonl
{"suite_name": "my_suite", "description": "My tests", "version": "1.0.0"}
{"id": "TEST-001", "prompt": "Your prompt", "expected": {"type": "contains", "required_words": ["answer"]}}
{"id": "TEST-002", "prompt": "Format check", "expected": {"type": "regex", "pattern": "\\d{4}-\\d{2}-\\d{2}"}}
```

### Expected Types (Advanced)

```json
{"type": "contains", "required_words": ["word1", "word2"]}
{"type": "exact", "value": "APPROVED"}
{"type": "regex", "pattern": "\\d+"}
{"type": "length", "max_words": 50, "max_sentences": 3}
{"type": "json_schema", "schema": {"type": "object", "required": ["name"]}}
```

## Best Practices

### 1. Start Small
Begin with 10-20 tests covering your most important cases.

### 2. Diverse Test Types
Include:
- Easy baseline tests (sanity check)
- Hard edge cases (where you've seen failures)
- Different output formats (JSON, lists, paragraphs)

### 3. Clear Expectations
For `exact` scorer, be precise:
- `"Yes"` won't match `"yes"` or `"Yes."`
- Use `contains` for more flexibility

### 4. Version Control
Export JSONL and commit to git.

### 5. Watch the Holdout
If your holdout score drops significantly over iterations, you're overfitting. Consider adding more diverse test cases.

## FAQ

**Q: Where are my suites stored?**
A: UI uploads go to SQLite (`data/custom_suites.db`). CLI suites go to `suites/` folder.

**Q: How many tests should I have?**
A: At least 10. The holdout split works best with 20+.

**Q: Can I disable holdout split?**
A: Not in UI. For CLI, use built-in suites which run all tests together.

**Q: How do I share a suite?**
A: Export as JSONL from UI, then share the file or commit to git.
