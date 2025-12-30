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

## [0.4.0] - 2025-12-30

### Added

#### Ed25519 Attestation Signing (`meridian/storage/signing.py`)
- Cryptographic signatures for attestation bundles
- CLI commands: `keygen`, `sign`, `verify --key`
- Explicit threat model documentation
- Key management (generate, save, load)

#### Custom Suites (`meridian/suites/custom.py`)
- Upload your own prompts (JSONL/CSV)
- Dev/Holdout split (80/20) to prevent overfitting
- Leak detection (warning if input contains expected)
- Scorer suggestions (LLM Judge, JSON schema)
- Suite versioning (v1, v2, etc.)
- SQLite storage with export

#### Suite Certification (`meridian/certification/suite_badge.py`)
- Certify evaluation runs with verifiable badges
- CLI: `meridian certify-run --id <run_id> --save`
- SVG and Markdown badge generation
- Accuracy + verification status

#### Provider Certification (`meridian/certification/tests.py`)
- 14 standardized tests for adapter validation
- CLI: `meridian certify --model <model_id> --save`
- 80%+ score = CERTIFIED status

#### REST API (`api/server.py`)
- Full REST API with FastAPI
- Endpoints: `/api/suites`, `/api/runs`, `/api/compare`, `/api/generate`
- Swagger documentation at `/docs`
- CLI command: `python -m meridian.cli serve`

#### UI Pages
- **Certification** (`ui/pages/8_Certification.py`) - Provider and Suite certification
- **Create Suite** (`ui/pages/9_Create_Suite.py`) - Upload custom prompts
- **Playground** (`ui/pages/6_Playground.py`) - Interactive testing

#### New Model Providers
- **Mistral AI**: mistral_small, mistral_medium, mistral_large
- **Groq**: groq_llama70b, groq_llama8b, groq_mixtral
- **Together AI**: together_llama70b, together_mixtral

#### HTML Reports (`meridian/reports/html_generator.py`)
- Beautiful single-file HTML reports
- CLI: `meridian report --run-id <id> --open`

#### Documentation
- `docs/CUSTOM_SUITES.md` - Complete guide for custom suites
- `docs/CERTIFICATION.md` - Provider and Suite certification
- `docs/THREAT_MODEL.md` - Explicit threat model for attestation

### Changed
- **Breaking**: `core` module deprecated, use `meridian` instead
- Verify command now supports signature verification with `--key`
- Custom suites visible in Run Suite dropdown as `[Custom] name`

### Removed
- `core/` directory (replaced by `meridian/`)

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

