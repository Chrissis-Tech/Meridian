"""
Meridian Provider Certification

Standard tests for provider adapter validation.
"""

from .tests import (
    CertificationSuite,
    CertificationResult,
    ProviderCertification,
    certify_provider,
)
from .badge import (
    generate_badge_svg,
    generate_badge_markdown,
    save_certification,
)
from .suite_badge import (
    SuiteCertification,
    certify_suite_run,
    generate_suite_badge_svg,
    generate_suite_badge_markdown,
    save_suite_certification,
)

__all__ = [
    # Provider certification
    "CertificationSuite",
    "CertificationResult", 
    "ProviderCertification",
    "certify_provider",
    "generate_badge_svg",
    "generate_badge_markdown",
    "save_certification",
    # Suite certification
    "SuiteCertification",
    "certify_suite_run",
    "generate_suite_badge_svg",
    "generate_suite_badge_markdown",
    "save_suite_certification",
]
