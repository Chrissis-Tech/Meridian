# Meridian Core 50: Dataset Card

## Dataset Summary

**Meridian_core_50** is a curated evaluation dataset for Large Language Models designed to assess core capabilities and detect common failure modes. The dataset focuses on reproducible, objective evaluation across five capability dimensions.

| Property | Value |
|----------|-------|
| **Name** | Meridian_core_50 |
| **Version** | 1.0.0 |
| **Size** | 50 test cases |
| **License** | MIT |
| **Format** | JSONL |
| **Languages** | English |

---

## Intended Use

### Primary Use
- Evaluating LLM quality on structured tasks
- Detecting regressions between model versions
- Comparing prompt engineering strategies
- CI/CD integration for automated testing

### Out of Scope
- Comprehensive capability assessment (use domain-specific suites)
- Multilingual evaluation
- Long-context evaluation (>2K tokens)
- Safety/toxicity evaluation (use dedicated frameworks)

---

## Dataset Structure

### Schema

```json
{
  "id": "CE-XXX",
  "capability": "string",
  "failure_mode": "string",
  "prompt": "string",
  "expected": {
    "type": "exact|contains|numeric|regex|json|refusal",
    "value": "string|number|object"
  },
  "difficulty": "easy|medium|hard",
  "tags": ["string"]
}
```

### Capability Distribution

| Capability | Count | Description |
|------------|-------|-------------|
| arithmetic | 10 | Basic mathematical operations |
| instruction_following | 12 | Format compliance, constraints |
| factual_recall | 10 | Knowledge retrieval |
| consistency | 8 | Self-agreement across runs |
| refusal | 10 | Appropriate abstention |

### Failure Mode Coverage

| Failure Mode | Count | Description |
|--------------|-------|-------------|
| calculation_error | 8 | Incorrect arithmetic |
| format_violation | 10 | Ignoring output constraints |
| hallucination | 8 | Fabricated information |
| inconsistency | 8 | Variable answers to same question |
| overconfidence | 8 | Answering unanswerable questions |
| prompt_sensitivity | 8 | Breaking under noise/typos |

---

## Dataset Creation

### Methodology

1. **Capability identification:** Selected five core capabilities based on common production use cases
2. **Failure mode mapping:** Identified typical failure patterns from literature and production experience
3. **Test case design:** Created objective, unambiguous tests with deterministic expected outputs
4. **Difficulty calibration:** Assigned difficulty based on DeepSeek Baseline performance

### Curation Principles

- **Objectivity:** All tests have unambiguous correct answers
- **Independence:** No test depends on another
- **Minimality:** Each test targets one capability/failure mode
- **Determinism:** Expected outputs are deterministic, not subjective

### Quality Control

- Manual review of all 50 cases
- Validation against GPT-2 to establish baseline
- Edge case verification for regex patterns

---

## Known Biases and Limitations

### Biases

| Bias | Description | Impact |
|------|-------------|--------|
| English-only | All prompts in English | Not suitable for multilingual evaluation |
| Western-centric | Factual questions based on Western knowledge | May disadvantage non-Western models |
| Format bias | Assumes certain output formats | May penalize valid alternative formats |
| Difficulty distribution | Calibrated on GPT-2 | May be trivial for larger models |

### Limitations

1. **Small sample size:** 50 tests provide limited statistical power (CI width ~10-15%)
2. **Coverage gaps:** Does not cover reasoning, creativity, or safety
3. **Static dataset:** Does not adapt to model capabilities
4. **Prompt sensitivity:** Results may vary with minor prompt changes

---

## Ethical Considerations

- No personally identifiable information (PII)
- No harmful or offensive content
- No copyrighted material beyond fair use
- No tests designed to elicit harmful outputs

---

## Maintenance

### Versioning
- Semantic versioning (MAJOR.MINOR.PATCH)
- MAJOR: Breaking changes to schema or test semantics
- MINOR: New tests or capability categories
- PATCH: Corrections to existing tests

### Contact
- Issues: GitHub repository
- Updates: CHANGELOG.md

---

## Citation

```bibtex
@dataset{Meridian_core_50,
  title={Meridian Core 50: A Curated LLM Evaluation Dataset},
  author={Meridian Contributors},
  year={2025},
  version={1.0.0},
  url={https://github.com/Chrissis-Tech/Meridian}
}
```

---

## License

MIT License - see LICENSE file.
