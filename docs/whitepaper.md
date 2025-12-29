# Meridian: A Framework for Systematic LLM Evaluation and Interpretability

**Version 0.3 | December 2025**

## Abstract

This paper presents Meridian, a framework for systematic evaluation of Large Language Models (LLMs) that integrates behavioral testing with mechanistic interpretability. We introduce operational definitions for key evaluation concepts, propose a risk-estimation model for LLM deployment, and establish connections between observable behaviors and underlying model components. The framework addresses the gap between black-box evaluation and white-box analysis, enabling both capability assessment and failure mode diagnosis.

---

## 1. Introduction

### 1.1 The Evaluation Problem

LLMs exhibit emergent behaviors that are difficult to characterize through traditional software testing. Unlike deterministic systems, LLMs:

- Produce variable outputs for identical inputs
- Exhibit capability degradation under distribution shift
- Generate plausible but incorrect information (hallucinations)
- Show sensitivity to prompt formatting and phrasing

These properties demand evaluation frameworks that go beyond pass/fail testing to capture uncertainty, consistency, and calibration.

### 1.2 Contribution

Meridian addresses these challenges through:

1. **Operational definitions** for consistency, hallucination detection, and calibration
2. **Statistical rigor** via bootstrap confidence intervals and significance testing
3. **Interpretability integration** connecting behavioral failures to model components
4. **Regression detection** for continuous deployment scenarios

---

## 2. Metrics and Failure Modes Taxonomy

We propose a structured taxonomy of LLM failure modes, each mapped to detection mechanisms:

| Category | Failure Mode | Detection Method | Example |
|----------|--------------|------------------|---------|
| **Format** | Invalid JSON | Schema validation | `{"name": Alice}` (unquoted) |
| **Format** | Length violation | Token/word count | Output exceeds max_tokens |
| **Format** | Constraint violation | Regex patterns | Missing required prefix |
| **Consistency** | High variance | Multi-run agreement | Different answers to same question |
| **Consistency** | Self-contradiction | Semantic similarity | Conflicting claims within output |
| **Factuality** | Fake citation | Pattern heuristics | `(Smith, 2029)` |
| **Factuality** | Unsourced statistic | Pattern heuristics | `73% of experts agree...` |
| **Factuality** | Confident fabrication | Refusal detection | Answering unanswerable questions |
| **Robustness** | Typo sensitivity | Perturbed inputs | `Whta is...` → wrong answer |
| **Robustness** | Padding sensitivity | Noise injection | Extra whitespace breaks output |
| **Robustness** | Format sensitivity | Template variation | Minor rephrasing causes failure |
| **Retrieval** | Needle miss | Position-based tests | Fails to find info in context |
| **Retrieval** | Distractor capture | Multi-document tests | Returns wrong source |
| **Security** | Injection success | Adversarial prompts | Executes injected instruction |
| **Security** | Jailbreak success | Policy violation | Bypasses safety guidelines |

This taxonomy enables systematic coverage analysis: a complete evaluation suite should include tests for each failure mode relevant to the deployment context.

---

## 3. Operational Definitions

### 3.1 Self-Consistency

**Definition:** A model exhibits self-consistency on task T if, across n independent generations, the normalized outputs converge to a stable distribution.

**Primary metric (Mode Consistency):**
```
ModeConsistency(T) = max_k |{outputs matching mode_k}| / n
```

**Secondary metric (Pairwise Agreement):**
```
PairwiseAgreement(T) = (2 / n(n-1)) × Σ_{i<j} I(o_i ≈ o_j)
```

We report Mode Consistency as the primary metric (interpretable, fast) and Pairwise Agreement as secondary (robust to multi-modal distributions).

**Interpretation:**
- Consistency ≥ 0.8: High agreement, suitable for deterministic tasks
- Consistency ∈ [0.5, 0.8): Moderate agreement, requires ensemble or verification
- Consistency < 0.5: Low agreement, indicates ambiguity or capability gap

**Definition of deterministic tasks:** Suites run with `temperature=0` and scoring via exact-match, numeric-match, or JSON-schema validation. These tasks have objectively correct answers and should yield identical outputs across runs.

### 3.2 Hallucination Heuristics

**Definition:** Hallucination indicators are textual patterns that correlate with unverifiable or fabricated claims.

**Operationalization:**

| Indicator | Pattern | Risk Level |
|-----------|---------|------------|
| Fake citations | `(Author, YYYY)` without source | High |
| Unsourced statistics | `X% of studies show...` | Medium |
| Confident false claims | `It is well-established that...` | Medium |
| Fabricated URLs | URLs matching common patterns but non-existent | High |

**Limitation:** These heuristics detect surface patterns, not semantic accuracy. Ground-truth verification requires external knowledge bases.

### 3.3 Calibration

**Definition:** A model is well-calibrated if its confidence scores correlate with actual accuracy.

**Expected Calibration Error (ECE):**
```
ECE = Σ (|B_i| / n) × |acc(B_i) - conf(B_i)|
```

**Maximum Calibration Error (MCE):**
```
MCE = max_i |acc(B_i) - conf(B_i)|
```

**Interpretation:**
- ECE < 0.05: Well-calibrated
- ECE ∈ [0.05, 0.15): Moderately calibrated
- ECE ≥ 0.15: Poorly calibrated

### 3.4 Risk-Coverage Trade-off

**Definition:** The risk-coverage curve characterizes model performance as a function of selective prediction threshold.

- Risk(τ) = error rate on samples where confidence > τ
- Coverage(τ) = fraction of samples where confidence > τ

**Application:** Determine optimal abstention threshold where risk must satisfy SLA constraints (e.g., risk < 5%) while maximizing coverage.

---

## 4. Conceptual Model: Evaluation as Risk Estimation

### 4.1 Task Distribution Framework

We model LLM evaluation as estimating risk under a distribution of tasks:

```
R(M, D) = E_{t~D}[L(M(t), y_t)]
```

Where:
- M: Model under evaluation
- D: Task distribution
- L: Loss function (0-1 for classification, custom for generation)
- y_t: Expected output for task t

### 4.2 Confidence Intervals

Point estimates of accuracy are insufficient for decision-making. Meridian provides:

**Bootstrap CI for accuracy:**
1. Resample n results with replacement (B=1000 iterations)
2. Compute accuracy for each resample
3. Report [2.5th, 97.5th] percentiles as 95% CI

**Interpretation:** If CI width > 0.1, sample size is insufficient for reliable conclusions.

### 4.3 Significance Testing

For A/B comparisons between models or prompts:

**McNemar's Test:** For paired binary outcomes
```
χ² = (|b - c| - 1)² / (b + c)
```
Where b = cases A correct, B wrong; c = cases A wrong, B correct.

**Effect Size (Cohen's d):**
```
d = (μ_B - μ_A) / σ_pooled
```

| Effect Size | Interpretation |
|-------------|----------------|
| \|d\| < 0.2 | Negligible |
| \|d\| ∈ [0.2, 0.5) | Small |
| \|d\| ∈ [0.5, 0.8) | Medium |
| \|d\| ≥ 0.8 | Large |

### 4.4 Implementation Status

| Statistical Method | Implemented | Requirements |
|--------------------|-------------|---------------|
| Bootstrap CI (accuracy) | Yes | ≥20 samples |
| McNemar's test (paired A/B) | Yes | Binary pass/fail outcomes |
| Cohen's d (effect size) | Yes | Continuous scores |
| Mode Consistency | Yes | Multi-run (n≥3) |
| Pairwise Agreement | Yes | Multi-run (n≥3) |
| ECE/MCE (calibration) | Yes | Model must expose log-probs |
| Risk-Coverage curve | Yes | Confidence scores or consistency proxy |

---

## 5. Interpretability Integration

### 5.1 From Behavior to Components

Meridian bridges the gap between:
- **Behavioral evaluation:** What does the model do wrong?
- **Mechanistic analysis:** Which components cause the failure?

### 5.2 Attention Analysis

**Purpose:** Identify which input tokens influence specific outputs.

**Method:**
1. Extract attention patterns A ∈ R^(layers × heads × seq × seq)
2. Compute entropy per head: H(A_l,h) = -Σ a log(a)
3. Identify focused heads (low entropy) as "circuit candidates"

**Limitation:** Attention patterns indicate information flow, not causal responsibility.

### 5.3 Logit Lens

**Purpose:** Track prediction evolution across layers.

**Method:**
1. Extract hidden states h_l at each layer
2. Project to vocabulary: logits_l = h_l @ W_E^T
3. Observe when correct prediction emerges

**Interpretation:** Earlier emergence suggests robust representation; late emergence suggests fragile computation.

### 5.4 Causal Tracing

**Purpose:** Identify components necessary for correct behavior.

**Method:**
1. Run model on clean input → correct output
2. Run model on corrupted input → incorrect output
3. Patch activations from clean run into corrupted run
4. Measure recovery of correct output

**Corruption strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Noise corruption | Replace token span with random tokens | General robustness |
| Entity swap | Replace key entity (e.g., "France" → "Germany") | Factual association |
| Key-span deletion | Remove the "needle" or critical information | Retrieval tasks |

**Note:** Results depend on corruption choice. We recommend reporting which strategy was used.

**Metric:** Impact(component) = P(correct | patched) - P(correct | corrupted)

---

## 6. Validity and Reliability

Rigorous evaluation requires acknowledging measurement limitations:

### 6.1 Construct Validity

**Issue:** Test suites sample from a constructed distribution that may not represent production traffic.

**Mitigation:**
- Use domain-specific suites reflecting actual use cases
- Validate suite coverage against production query logs
- Report results as "performance on suite X" rather than "model capability"

### 6.2 Measurement Error

**Issue:** Heuristic-based detection (e.g., hallucination patterns) introduces systematic measurement error. Pattern-based detectors have non-zero false positive and false negative rates.

**Mitigation:**
- Report heuristic precision/recall when ground truth is available
- Use multiple detection methods and report agreement
- Flag results as "heuristic-detected" vs. "verified"

### 6.3 Statistical Power

**Issue:** Confidence interval width reflects statistical uncertainty. A 95% CI of [0.35, 0.55] indicates insufficient sample size for reliable conclusions.

**Guidance:**
- For CI width ≤ 0.10: n ≥ 100 tests typically required
- For CI width ≤ 0.05: n ≥ 400 tests typically required
- Report power analysis or minimum detectable effect size

### 6.4 Reliability

**Issue:** Results may vary across runs due to:
- Model API non-determinism
- Sampling temperature > 0
- Random seed differences

**Mitigation:**
- Use temperature=0 for deterministic evaluation
- Report variance across multiple runs
- Define "reproducible" as within 5% of baseline

---

## 7. Relation to Transformer Circuits

### 7.1 Conceptual Bridge

Meridian operationalizes the Transformer Circuits methodology for practical evaluation:

| Circuits Concept | Meridian Implementation |
|------------------|---------------------------|
| Attention heads as "movers" | Attention visualization + entropy ranking |
| Residual stream | Hidden state extraction at each layer |
| Induction heads | Consistency analysis across repeated patterns |
| Feature superposition | Logit lens for vocabulary projection |

### 7.2 Evidence Flow

```
Behavioral failure → Attention analysis → Layer identification → Causal tracing → Component hypothesis
```

This flow enables:
1. Detecting capability regressions through behavioral tests
2. Diagnosing root causes through interpretability
3. Guiding interventions (fine-tuning, prompt engineering)

---

## 8. Limitations

### 8.1 Methodological Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Hallucination heuristics are pattern-based | May miss semantic hallucinations | Combine with external verification |
| Calibration requires confidence scores | Not all APIs expose probabilities | Use consistency as proxy |
| Causal tracing is approximate | True circuits may be distributed | Use multiple corruption strategies |

### 8.2 Coverage Limitations

Meridian does not address:
- Multi-modal evaluation (images, audio)
- Long-context evaluation (>4K tokens by default)
- Real-time latency requirements (<100ms)
- Multilingual evaluation (English-centric suites)

---

## 9. Conclusion

Meridian provides a principled approach to LLM evaluation that:

1. Defines measurable properties (consistency, calibration, hallucination risk)
2. Quantifies uncertainty through statistical methods
3. Connects observable failures to model components
4. Enables continuous monitoring through CI integration

The framework is designed for production use while maintaining methodological rigor. Future work includes expanding coverage to multi-modal and multilingual scenarios, and deeper integration with mechanistic interpretability tools.

---

## References

1. Elhage, N., Nanda, N., Olsson, C., et al. (2022). A Mathematical Framework for Transformer Circuits. *Transformer Circuits Thread* [Blog post]. Anthropic. https://transformer-circuits.pub/2021/framework/index.html (Accessed 2025-12-28)

2. Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. In *Advances in Neural Information Processing Systems 30* (NeurIPS). https://arxiv.org/abs/1705.08500

3. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. In *Proceedings of the 34th International Conference on Machine Learning* (ICML). https://arxiv.org/abs/1706.04599

4. Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics* (ACL). https://arxiv.org/abs/2109.07958

5. Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. In *Advances in Neural Information Processing Systems 35* (NeurIPS). https://arxiv.org/abs/2202.05262

6. Ribeiro, M. T., Wu, T., Guestrin, C., & Singh, S. (2020). Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (ACL). https://arxiv.org/abs/2005.04118

---

*Meridian v0.2 | MIT License | December 2025*
