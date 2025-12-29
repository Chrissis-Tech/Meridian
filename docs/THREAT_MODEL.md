# Threat Model

This document describes the security threats Meridian addresses, those it does not, and the assumptions underlying our security design.

## Scope

Meridian is designed to evaluate LLM behavior, including resistance to adversarial attacks. This document covers:

1. Attacks against LLMs that Meridian can detect
2. Attacks against Meridian itself
3. Operational security considerations

---

## 1. LLM Attack Detection

### 1.1 Attacks Covered

| Attack Type | Coverage | Test Suite | Detection Method |
|-------------|----------|------------|------------------|
| Prompt Injection | Partial | security_adversarial | Pattern matching, output validation |
| Jailbreaking | Partial | security_adversarial | Refusal detection, policy violation |
| Data Extraction | Partial | security_adversarial | Information leakage heuristics |
| Format Breaking | Full | security_adversarial | Structured output validation |
| Hallucination Induction | Partial | hallucination_control | Citation verification, refusal checks |
| Reward Hacking | Partial | reward_hacking | Spoofing, length, citation, tool detection |

### 1.2 Reward Hacking Definition

**Reward hacking** = optimizing evaluation metrics without improving task truth or success.

| Category | Description | Detection |
|----------|-------------|-----------|
| **Verifier spoofing** | Academic tone + wrong answer | Confidence phrases + error check |
| **Length gaming** | Padding output to increase score | Word count + filler patterns |
| **Citation laundering** | Fake citations with correct format | DOI/URL validation, pattern matching |
| **Tool misuse** | Claiming tool access without it | Tool claim patterns vs. available tools |

Meridian's `reward_hacking` suite (20 tests) evaluates resistance to these gaming behaviors.

### 1.2 Attack Detection Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Pattern-based detection | Novel attacks may bypass | Regular suite updates |
| No semantic understanding | Subtle policy violations missed | Combine with human review |
| English-only | Non-English attacks not covered | Extend suites for target languages |
| Static payloads | Adaptive attacks not tested | Implement fuzzing extensions |

### 1.3 Not Covered

Meridian does **not** detect:

- Model poisoning or backdoors
- Training data extraction attacks
- Timing-based side channels
- Multimodal attacks (images, audio)
- Coordinated multi-turn attacks

---

## 2. Threats to Meridian

### 2.1 Attack Surface

```
External Input → [Test Suites] → [Runner] → [Model API] → [Storage] → [UI/Reports]
```

### 2.2 Threat Matrix

| Threat | Vector | Impact | Likelihood | Mitigation |
|--------|--------|--------|------------|------------|
| Malicious test suite | JSONL injection | Code execution | Low | Input validation, sandboxing |
| API key exposure | Log files, artifacts | Credential theft | Medium | Redaction, gitignore |
| Data exfiltration | Model outputs | Privacy breach | Medium | Output redaction |
| DoS via expensive tests | Long prompts, many runs | Resource exhaustion | Medium | Rate limiting, timeouts |
| Storage injection | SQL injection in results DB | Data corruption | Low | Parameterized queries |

### 2.3 Trust Boundaries

```
┌─────────────────────────────────────────────────┐
│                  Trusted Zone                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Test     │    │ Core     │    │ Storage  │  │
│  │ Suites   │───▶│ Runner   │───▶│ Layer    │  │
│  └──────────┘    └──────────┘    └──────────┘  │
└─────────────────────────────────────────────────┘
         │                │
         ▼                ▼
┌─────────────────────────────────────────────────┐
│            Untrusted Zone                       │
│  ┌──────────┐    ┌──────────┐                   │
│  │ External │    │ Model    │                   │
│  │ Models   │    │ Outputs  │                   │
│  └──────────┘    └──────────┘                   │
└─────────────────────────────────────────────────┘
```

### 2.4 Security Controls

| Control | Implementation | Status |
|---------|----------------|--------|
| Input validation | JSON schema for test suites | Implemented |
| Output redaction | PII pattern masking | Implemented |
| API key protection | Environment variables, .gitignore | Implemented |
| SQL injection prevention | Parameterized queries | Implemented |
| Rate limiting | Configurable request limits | Planned |
| Audit logging | Structured logs with timestamps | Partial |

---

## 3. Operational Security

### 3.1 Deployment Recommendations

**Minimal Exposure:**
- Run behind authentication (e.g., Streamlit password)
- Do not expose to public internet without review
- Use environment variables for all secrets

**Data Handling:**
- Do not evaluate production data with PII
- Enable redaction for sensitive deployments
- Regularly purge old runs and artifacts

**Access Control:**
- Limit database access to application only
- Review file permissions on artifacts directory
- Use separate API keys for evaluation vs. production

### 3.2 Incident Response

If credentials are exposed:
1. Rotate affected API keys immediately
2. Review access logs for unauthorized usage
3. Purge artifacts containing credentials
4. Update .gitignore and redaction rules

---

## 4. Assumptions

1. **Test suites are trusted:** Malicious test suites could execute arbitrary patterns
2. **Local filesystem is secure:** Artifacts stored without encryption
3. **Model APIs are reliable:** No protection against API unavailability
4. **Single-user deployment:** No multi-tenant isolation

---

## 5. Future Security Work

- [ ] Test suite sandboxing
- [ ] Encrypted artifact storage
- [ ] Multi-tenant isolation
- [ ] Adversarial robustness testing for Meridian itself
- [ ] Formal security audit

---

## Contact

Report security issues to: security@Chrissis-Tech.com (or via GitHub Security Advisories)

See SECURITY.md for vulnerability disclosure policy.
