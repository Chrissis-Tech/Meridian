# Custom Test Suites

This guide explains how to create your own test suites for Meridian.

## Quick Start

1. Create a `.jsonl` file in the `suites/` directory
2. Add a header line with suite metadata
3. Add test cases, one per line
4. Run with: `python -m core.cli run --suite your_suite_name --model deepseek_chat`

## File Format

Test suites are **JSONL** files (JSON Lines). Each line is a separate JSON object.

### Line 1: Suite Header

```json
{"suite_name": "my_custom_suite", "description": "What this suite tests", "version": "1.0.0"}
```

### Lines 2+: Test Cases

```json
{"id": "TEST-001", "prompt": "Your prompt here", "expected": {"type": "contains", "required_words": ["answer"]}}
```

## Expected Answer Types

### 1. Contains (most common)
Check if output contains specific words:

```json
{
  "id": "MATH-001",
  "prompt": "What is 7 + 8?",
  "expected": {
    "type": "contains",
    "required_words": ["15"]
  }
}
```

### 2. Exact Match
Output must exactly match:

```json
{
  "id": "FORMAT-001",
  "prompt": "Reply with only 'APPROVED' or 'REJECTED'",
  "expected": {
    "type": "exact",
    "value": "APPROVED"
  }
}
```

### 3. Regex Pattern
Output must match a regex:

```json
{
  "id": "DATE-001",
  "prompt": "What is today's date? Format: YYYY-MM-DD",
  "expected": {
    "type": "regex",
    "pattern": "\\d{4}-\\d{2}-\\d{2}"
  }
}
```

### 4. Length Constraint
Limit response length:

```json
{
  "id": "CONCISE-001",
  "prompt": "Summarize this in one sentence.",
  "expected": {
    "type": "length",
    "max_sentences": 1
  }
}
```

### 5. JSON Schema
Validate structured output:

```json
{
  "id": "JSON-001",
  "prompt": "Return a JSON object with 'name' and 'age' fields.",
  "expected": {
    "type": "json_schema",
    "schema": {
      "type": "object",
      "required": ["name", "age"],
      "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
      }
    }
  }
}
```

## Complete Example

Create `suites/my_company_tests.jsonl`:

```jsonl
{"suite_name": "my_company_tests", "description": "Tests for our specific use case", "version": "1.0.0"}
{"id": "SUPPORT-001", "prompt": "Customer says: 'My order #12345 hasn't arrived.' Extract the order number.", "expected": {"type": "contains", "required_words": ["12345"]}}
{"id": "SUPPORT-002", "prompt": "Customer says: 'I want a refund for my broken laptop.' What is the issue type? Reply with only: REFUND, EXCHANGE, or SUPPORT", "expected": {"type": "exact", "value": "REFUND"}}
{"id": "SUPPORT-003", "prompt": "Summarize this ticket in under 20 words: 'I purchased item SKU-789 on Monday and it arrived damaged. The screen has a crack. I need a replacement ASAP.'", "expected": {"type": "length", "max_words": 20}}
```

## Running Your Suite

### CLI
```bash
python -m core.cli run --suite my_company_tests --model deepseek_chat
```

### UI
1. Go to "Run Suite" page
2. Your suite appears in the dropdown
3. Select model and run

## Best Practices

### 1. Use Clear Test IDs
```
GOOD: SUPPORT-001, EXTRACTION-042, FORMAT-TEST-12
BAD:  test1, t, abc
```

### 2. Add Tags for Filtering
```json
{"id": "X", "prompt": "...", "expected": {...}, "tags": ["extraction", "customer_support"]}
```

### 3. Keep Prompts Realistic
Use actual examples from your production data (sanitized).

### 4. Mix Difficulty Levels
Include easy, medium, and hard cases to see where models struggle.

### 5. Test Edge Cases
- Empty inputs
- Very long inputs
- Ambiguous requests
- Malformed data

## Importing from CSV

If you have tests in CSV format:

```python
import csv
import json

with open('tests.csv') as f:
    reader = csv.DictReader(f)
    
    with open('suites/from_csv.jsonl', 'w') as out:
        # Header
        out.write(json.dumps({"suite_name": "from_csv", "description": "Imported from CSV"}) + '\n')
        
        for i, row in enumerate(reader):
            test = {
                "id": f"CSV-{i+1:03d}",
                "prompt": row['prompt'],
                "expected": {"type": "contains", "required_words": row['expected'].split(',')}
            }
            out.write(json.dumps(test) + '\n')
```

## Validating Your Suite

Check for errors before running:

```bash
python -c "
from core.suites import SuiteLoader
loader = SuiteLoader()
suite = loader.load_suite('suites/my_company_tests.jsonl')
print(f'Loaded {len(suite.test_cases)} tests')
for tc in suite.test_cases[:3]:
    print(f'  - {tc.id}: {tc.prompt[:50]}...')
"
```

## FAQ

**Q: Where do I put my files?**
A: In the `suites/` directory, with `.jsonl` extension.

**Q: How many tests should I have?**
A: At least 10 for statistical significance. 30-50 is ideal.

**Q: Can I use my own scoring logic?**
A: Yes, extend `core/scoring/methods.py` with custom scorers.

**Q: How do I test multi-turn conversations?**
A: Currently single-turn only. Multi-turn support coming in v0.4.

---

For more examples, see the existing suites in `suites/`:
- `rag_evaluation.jsonl` - Document retrieval
- `code_analysis.jsonl` - Code review
- `business_analysis.jsonl` - Financial calculations
