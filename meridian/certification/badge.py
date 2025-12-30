"""
Meridian Provider Certification - Badge Generation

Generates verification badges for certified providers.
"""

import json
from typing import Optional
from pathlib import Path
from datetime import datetime

from .tests import ProviderCertification


def generate_badge_svg(cert: ProviderCertification) -> str:
    """Generate an SVG badge for the certification."""
    score = cert.score
    
    # Color based on score
    if score >= 80:
        color = "#4c1"  # Green
        status = "CERTIFIED"
    elif score >= 50:
        color = "#fe7d37"  # Orange
        status = "PARTIAL"
    else:
        color = "#e05d44"  # Red
        status = "FAILED"
    
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="150" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="a">
    <rect width="150" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#a)">
    <path fill="#555" d="M0 0h70v20H0z"/>
    <path fill="{color}" d="M70 0h80v20H70z"/>
    <path fill="url(#b)" d="M0 0h150v20H0z"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="35" y="15" fill="#010101" fill-opacity=".3">meridian</text>
    <text x="35" y="14">meridian</text>
    <text x="110" y="15" fill="#010101" fill-opacity=".3">{status} {score}%</text>
    <text x="110" y="14">{status} {score}%</text>
  </g>
</svg>'''
    return svg


def generate_badge_markdown(cert: ProviderCertification) -> str:
    """Generate markdown badge for README inclusion."""
    score = cert.score
    
    if score >= 80:
        color = "brightgreen"
        status = "certified"
    elif score >= 50:
        color = "orange"
        status = "partial"
    else:
        color = "red"
        status = "failed"
    
    # shields.io style badge
    badge = f"![Meridian {cert.model_id}](https://img.shields.io/badge/meridian-{status}%20{score}%25-{color})"
    return badge


def save_certification(cert: ProviderCertification, output_dir: Optional[Path] = None) -> Path:
    """Save certification report and badge."""
    if output_dir is None:
        output_dir = Path("certifications")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filename based on model and date
    date_str = datetime.now().strftime("%Y%m%d")
    base_name = f"{cert.model_id}_{date_str}"
    
    # Save JSON report
    json_path = output_dir / f"{base_name}.json"
    with open(json_path, 'w') as f:
        json.dump(cert.to_dict(), f, indent=2, default=str)
    
    # Save SVG badge
    svg_path = output_dir / f"{base_name}_badge.svg"
    with open(svg_path, 'w') as f:
        f.write(generate_badge_svg(cert))
    
    return json_path
