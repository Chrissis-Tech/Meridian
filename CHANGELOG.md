# Changelog

All notable changes to Meridian will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Multi-modal evaluation support
- Multilingual test suites
- OpenTelemetry integration

---

## [0.4.0] - 2025-12-29

### Added

#### REST API (`api/server.py`)
- Full REST API with FastAPI
- Endpoints: `/api/suites`, `/api/runs`, `/api/compare`, `/api/generate`
- Swagger documentation at `/docs`
- CLI command: `python -m core.cli serve`

#### Playground UI (`ui/pages/6_Playground.py`)
- Interactive prompt testing across models
- File upload for documents (txt, md, csv, json, py, etc.)
- Real-time cost and latency comparison
- LLM Judge for complex evaluation
- Export to JSON/Markdown
- Save as test case

#### LLM Judge (`core/judge.py`)
- Evaluate complex outputs with another LLM
- Scores: correctness, completeness, format adherence, no hallucination
- Configurable pass threshold

#### New Model Providers
- **Mistral AI**: mistral_small, mistral_medium, mistral_large
- **Groq**: groq_llama70b, groq_llama8b, groq_mixtral (ultra-fast)
- **Together AI**: together_llama70b, together_mixtral, together_codellama

#### HTML Reports (`core/reports/html_generator.py`)
- Beautiful single-file HTML reports
- KPIs, confidence intervals, results table
- CLI: `python -m core.cli report --run-id <id> --open`

#### Premium UI Design
- Monochrome "Vercel/Stripe" aesthetic
- Custom Vega-Lite theme for charts
- Reusable components: kpi_card, pills_bar, callout
- Dumbbell charts for comparisons

#### New Test Suite
- `imo_generation_quality.jsonl` - Mathematical rigor evaluation
- Anti-hedging, anti-contradiction, anti-truncation tests

#### Webhook Notifications (`core/notifications/webhooks.py`)
- Send alerts on run complete, regression, threshold breach
- Supports Slack, Discord, Microsoft Teams
- Configurable via environment variables

#### Streaming Support (`core/streaming.py`)
- Stream responses from LLM providers
- Time to First Token (TTFT) metrics
- Callback-based chunk handling

### Changed
- UI navigation updated with Playground
- Model adapter registry refactored
- Version bump to 0.4.0

---

## [0.3.0] - 2025-12-28

### Added

#### Production-Grade Test Suites
- `rag_evaluation` (10 tests) - Long-context retrieval from documents
- `code_analysis` (10 tests) - Debug, review, optimize code
- `document_processing` (10 tests) - Summarize, extract, transform
- `multi_step_reasoning` (10 tests) - Logic and dependency resolution
- `edge_cases` (12 tests) - Ambiguity, malformed data, contradictions
- `business_analysis` (10 tests) - SaaS metrics, forecasting, ROI

#### DeepSeek Integration
- DeepSeek Chat as default model
- DeepSeekAdapter with OpenAI-compatible API
- Cost-effective evaluation (~$0.001 per test)

#### Reliability Module (`core/reliability/`)
- `DeterminismProfile` enum: strict, quasi, stochastic
- `DeterminismChecker` with replication-based verification
- `VerbosityPenalty` for length gaming detection

#### Verified Factuality (`core/verify/`)
- Claim extraction from text
- BM25 and embedding-based evidence retrieval
- Grounding score: supported/refuted/not_enough_info

#### Consistency Formalization (`core/consistency/`)
- `normalize()` with canonical JSON, numeric tolerance
- 4 similarity methods: exact, edit, embedding, jaccard
- `cluster_outputs()` with mode consistency and pairwise agreement

#### Interpretability Upgrades (`core/explain/`)
- `decompose_residual_stream()` for logit attribution
- `compute_head_attribution()` for per-head analysis

#### Reward Hacking Suite
- `suites/reward_hacking.jsonl` (20 tests, 4 categories)
- Detection for spoofing, length gaming, citation laundering, tool misuse

#### Documentation
- Whitepaper with failure modes taxonomy
- THREAT_MODEL.md with reward hacking definition
- CITATION.cff, REPRODUCIBILITY.md
- CONTRIBUTING.md, CODE_OF_CONDUCT.md

### Changed
- Default model: deepseek_chat (replaces local_gpt2)
- Baselines updated with real DeepSeek results

---

## [0.1.0] - 2025-12-01

Initial implementation with core infrastructure.

---

[Unreleased]: https://github.com/Chrissis-Tech/Meridian/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Chrissis-Tech/Meridian/releases/tag/v0.3.0
[0.1.0]: https://github.com/Chrissis-Tech/Meridian/releases/tag/v0.1.0

