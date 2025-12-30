# Attestation Threat Model

This document explicitly states what Meridian's attestation system protects against and what it does NOT protect against.

## What Attestation DOES Protect

### 1. Bundle Tampering (Integrity)
**Threat:** Someone modifies responses, config, or metrics after the run.

**Protection:** SHA256 hashes of all files in manifest.json. Any modification causes hash mismatch and verify fails.

```bash
python -m meridian.cli verify --id run_xxx
# Output: ✗ Tampered file: responses/test_5.json
```

### 2. File Deletion/Addition (Completeness)
**Threat:** Someone removes failing tests or adds fake passing tests.

**Protection:** Manifest includes complete file list. Missing or extra files detected.

### 3. Signer Identity (Authenticity) [v0.4+]
**Threat:** Someone claims results came from a specific party.

**Protection:** Ed25519 digital signature. If bundle is signed:
- Signature proves holder of private key created/approved the bundle
- Public key can be verified against known identity

```bash
python -m meridian.cli verify --id run_xxx --key org_public.pub
# Output: ✓ Valid signature from key abc12345
```

### 4. Environment Drift Detection
**Threat:** Results differ due to different Python/OS/dependencies.

**Protection:** Environment captured at run time:
- Python version
- Platform
- Git commit
- Meridian version

## What Attestation DOES NOT Protect

### 1. Remote Model Behavior ❌
**Cannot prove:** That OpenAI/DeepSeek/Anthropic actually ran the model.

**Why:** We only see API responses. The remote provider could:
- Route to a different model
- Apply hidden filters
- Change model weights between runs

**Mitigation:** Replay with drift detection can catch major changes.

### 2. Model Version/Weights ❌
**Cannot prove:** The exact model weights used.

**Why:** API providers don't expose model checksums.

**Mitigation:** Capture model_id, timestamp, and compare with provider changelogs.

### 3. Local Faking ❌
**Cannot prove:** That results weren't locally generated without calling the API.

**Why:** Anyone with the private key can sign anything.

**Mitigation:** 
- Trust based on key holder reputation
- Cross-reference with API usage logs (if available)
- Third-party attestation services (future)

### 4. Key Compromise ❌
**Cannot protect:** Against stolen private keys.

**If private key is compromised:**
- Attacker can sign fake bundles
- All previous signatures remain valid

**Mitigation:**
- Key rotation
- Hardware security modules (HSM)
- Multi-signature schemes (future)

### 5. Replay Attacks ❌
**Cannot protect:** Against replaying old valid bundles as new.

**Mitigation:** Check timestamps, maintain audit log of expected runs.

## Trust Levels

| Level | Protections | Use Case |
|-------|-------------|----------|
| **Basic** (default) | Integrity only | Internal reproducibility |
| **Signed** | Integrity + Identity | Shared results, audits |
| **Verified Replay** | Integrity + Fresh execution | CI gates, compliance |

## Verification Commands

```bash
# Basic integrity check
python -m meridian.cli verify --id run_xxx

# With signature verification
python -m meridian.cli verify --id run_xxx --key signer.pub

# Replay with drift detection
python -m meridian.cli replay --id run_xxx --mode drift
```

## Summary

| Threat | Protected? | Method |
|--------|------------|--------|
| File tampering | ✅ Yes | SHA256 hashes |
| File deletion | ✅ Yes | Manifest completeness |
| Signer identity | ✅ Yes | Ed25519 signature |
| Environment drift | ⚠️ Detect | Captured metadata |
| Remote model behavior | ❌ No | Cannot verify externally |
| Model version | ❌ No | No checksum from providers |
| Local faking | ❌ No | Trust-based |
| Key compromise | ❌ No | Operational security |

## Honest Statement

> Meridian attestation proves **who ran what configuration** and **that results weren't modified**.
> It does NOT prove **what the model actually did internally**.
>
> For API-based models, this is an inherent limitation of the threat model.
> For local models, replay --strict provides stronger guarantees.
