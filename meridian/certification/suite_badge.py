"""
Meridian Suite Certification

Generates verification badges for user evaluation runs.
Allows users to prove their evaluation results are authentic.
"""

import json
import hashlib
from typing import Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class SuiteCertification:
    """Certification for a completed evaluation run."""
    run_id: str
    suite_name: str
    model_id: str
    accuracy: float
    passed_tests: int
    total_tests: int
    timestamp: str
    attestation_hash: str
    badge_hash: str
    verified: bool
    
    def to_dict(self):
        return asdict(self)


def certify_suite_run(run_id: str) -> SuiteCertification:
    """Generate a certification badge for an attested run."""
    from meridian.storage import ArtifactManager
    from meridian.storage.attestation import AttestationManager
    
    am = ArtifactManager()
    atm = AttestationManager()
    
    # Load run data
    run_dir = am.base_dir / run_id
    if not run_dir.exists():
        raise ValueError(f"Run not found: {run_id}")
    
    # Check attestation
    is_valid, _ = atm.verify(run_id)
    
    # Load config
    config_file = run_dir / "config.json"
    if config_file.exists():
        with open(config_file, encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Calculate metrics from responses
    responses_dir = run_dir / "responses"
    passed = 0
    total = 0
    
    if responses_dir.exists():
        for resp_file in responses_dir.glob("*.json"):
            try:
                with open(resp_file, encoding='utf-8') as f:
                    resp = json.load(f)
                total += 1
                # Check if test passed (correct field may vary)
                if resp.get("correct") or resp.get("passed") or resp.get("is_correct"):
                    passed += 1
            except:
                pass
    
    accuracy = (passed / total * 100) if total > 0 else 0.0
    
    # Load attestation manifest
    manifest_file = run_dir / "attestation.json"
    attestation_hash = ""
    if manifest_file.exists():
        with open(manifest_file, encoding='utf-8') as f:
            manifest = json.load(f)
            attestation_hash = manifest.get("manifest_hash", "")[:16]
    
    suite_name = config.get("suite", "unknown")
    model_id = config.get("model", "unknown")
    timestamp = config.get("timestamp", datetime.utcnow().isoformat())
    
    # Generate badge hash
    badge_data = f"{run_id}:{suite_name}:{model_id}:{accuracy}:{attestation_hash}"
    badge_hash = hashlib.sha256(badge_data.encode()).hexdigest()[:16]
    
    return SuiteCertification(
        run_id=run_id,
        suite_name=suite_name,
        model_id=model_id,
        accuracy=accuracy,
        passed_tests=passed,
        total_tests=total,
        timestamp=timestamp,
        attestation_hash=attestation_hash,
        badge_hash=badge_hash,
        verified=is_valid
    )


def generate_suite_badge_svg(cert: SuiteCertification) -> str:
    """Generate an SVG badge for the suite certification."""
    accuracy = cert.accuracy
    
    # Color based on accuracy
    if accuracy >= 80:
        color = "#4c1"  # Green
    elif accuracy >= 60:
        color = "#a4a61d"  # Yellow-green
    elif accuracy >= 40:
        color = "#fe7d37"  # Orange
    else:
        color = "#e05d44"  # Red
    
    verified_text = "✓" if cert.verified else "?"
    
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="200" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="a">
    <rect width="200" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#a)">
    <path fill="#555" d="M0 0h100v20H0z"/>
    <path fill="{color}" d="M100 0h100v20H100z"/>
    <path fill="url(#b)" d="M0 0h200v20H0z"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="50" y="15" fill="#010101" fill-opacity=".3">{cert.suite_name}</text>
    <text x="50" y="14">{cert.suite_name}</text>
    <text x="150" y="15" fill="#010101" fill-opacity=".3">{verified_text} {accuracy:.0f}%</text>
    <text x="150" y="14">{verified_text} {accuracy:.0f}%</text>
  </g>
</svg>'''
    return svg


def generate_suite_badge_markdown(cert: SuiteCertification) -> str:
    """Generate markdown badge for README inclusion."""
    accuracy = cert.accuracy
    
    if accuracy >= 80:
        color = "brightgreen"
    elif accuracy >= 60:
        color = "yellowgreen"
    elif accuracy >= 40:
        color = "orange"
    else:
        color = "red"
    
    verified = "verified" if cert.verified else "unverified"
    
    # shields.io style badge
    label = cert.suite_name.replace("_", "%20")
    badge = f"![{cert.suite_name}](https://img.shields.io/badge/{label}-{accuracy:.0f}%25%20{verified}-{color})"
    return badge


def save_suite_certification(cert: SuiteCertification, output_dir: Optional[Path] = None) -> Path:
    """Save suite certification report and badge."""
    if output_dir is None:
        output_dir = Path("suite_certifications")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filename based on run_id
    base_name = f"{cert.suite_name}_{cert.model_id}_{cert.run_id[-8:]}"
    
    # Save JSON report
    json_path = output_dir / f"{base_name}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(cert.to_dict(), f, indent=2)
    
    # Save SVG badge
    svg_path = output_dir / f"{base_name}_badge.svg"
    with open(svg_path, 'w', encoding='utf-8') as f:
        f.write(generate_suite_badge_svg(cert))
    
    return json_path
