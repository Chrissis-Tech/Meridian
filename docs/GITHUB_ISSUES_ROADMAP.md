# GitHub Issues Roadmap

Create these issues after pushing v0.3.0:

---

## Issue 1: v0.4 - Verified Factuality Module Enhancement

**Title:** [Feature] Enhanced VerifiedFactuality with LLM-judge option

**Labels:** enhancement, v0.4

**Body:**
```
## Summary
Extend the current heuristic-based factuality verification with optional LLM-judge support for semantic verification.

## Current State
- Claim extraction: regex-based
- Evidence retrieval: BM25/embedding
- Grounding: heuristic overlap scoring

## Proposed Enhancements
- [ ] LLM-judge option for supported/refuted classification
- [ ] Wikipedia/CommonCrawl integration for evidence base
- [ ] Confidence calibration for grounding scores
- [ ] Citation resolution (DOI lookup, URL checking)

## Dependencies
- Requires API key for judge model
- Optional: Wikipedia dump for offline use
```

---

## Issue 2: v0.4 - Consistency Formalization Enhancement

**Title:** [Feature] Semantic clustering for consistency analysis

**Labels:** enhancement, v0.4

**Body:**
```
## Summary
Improve consistency analysis with semantic clustering and normalized entropy reporting.

## Current State
- Mode consistency (dominant cluster / n)
- Pairwise agreement
- 4 similarity methods

## Proposed Enhancements
- [ ] Hierarchical clustering with dendrogram visualization
- [ ] Normalized entropy H(clusters) for consistency quantification
- [ ] Answer extraction normalization (remove preamble variations)
- [ ] Multi-modal consistency for outputs with different formats
```

---

## Issue 3: v0.4 - Reward Hacking Suite Expansion

**Title:** [Feature] Expanded reward hacking test coverage

**Labels:** enhancement, testing, v0.4

**Body:**
```
## Summary
Expand reward hacking detection with more sophisticated gaming patterns.

## Current Coverage (20 tests)
- Verifier spoofing (5)
- Length gaming (5)
- Citation laundering (5)
- Tool misuse (5)

## Proposed Additions
- [ ] Sycophancy detection (agreeing with false user statements)
- [ ] Confidence gaming (hedging to avoid penalties)
- [ ] Format exploitation (abusing structured output scoring)
- [ ] Multi-turn context manipulation
```

---

## Issue 4: v0.4 - Interpretability Dashboard Enhancement

**Title:** [Feature] Enhanced Explain UI with attribution visualization

**Labels:** enhancement, ui, v0.4

**Body:**
```
## Summary
Upgrade the Streamlit Explain page with residual stream and head attribution visualizations.

## Current State
- Attention heatmaps
- Logit lens line plots
- Causal tracing heatmaps

## Proposed Additions
- [ ] Residual decomposition waterfall chart
- [ ] Per-head attribution bar chart
- [ ] Interactive layer/head selection
- [ ] Export analysis as JSON/PDF
```

---

## Issue 5: v1.0 - Production Readiness

**Title:** [Milestone] v1.0 Production Release Checklist

**Labels:** milestone, v1.0

**Body:**
```
## Criteria for v1.0

### Testing
- [ ] 80%+ unit test coverage
- [ ] Integration tests for all adapters
- [ ] Fuzz testing for scoring functions

### Documentation
- [ ] API reference (Sphinx/MkDocs)
- [ ] Video tutorial
- [ ] Production deployment guide

### Performance
- [ ] Benchmarks for 1000+ test runs
- [ ] Memory profiling
- [ ] Async execution option

### Security
- [ ] External security audit
- [ ] Dependency vulnerability scan
- [ ] Rate limiting for UI
```

---

Copy these to GitHub after pushing v0.3.0.
