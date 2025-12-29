"""
Meridian Scoring - Citation Validity

Validates citations for authenticity.
"""

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse


@dataclass 
class CitationResult:
    """Result of citation validation."""
    is_valid: bool
    score: float  # 0-1
    citation_type: str  # doi, url, author_year, none
    found_citations: list[str]
    issues: list[str]


def validate_citations(text: str) -> CitationResult:
    """
    Validate citations in text for structural correctness.
    
    Checks:
    - DOI format validity
    - URL reachability patterns
    - Author-year format
    - Known fake patterns
    """
    issues = []
    found = []
    
    # 1. Find DOIs
    doi_pattern = r'10\.\d{4,}/[^\s]+'
    dois = re.findall(doi_pattern, text)
    for doi in dois:
        found.append(f"DOI: {doi}")
        if not validate_doi_format(doi):
            issues.append(f"Malformed DOI: {doi}")
    
    # 2. Find URLs
    url_pattern = r'https?://[^\s<>"\']+'
    urls = re.findall(url_pattern, text)
    for url in urls:
        found.append(f"URL: {url}")
        url_issues = validate_url_format(url)
        issues.extend(url_issues)
    
    # 3. Find author-year citations
    author_year_pattern = r'([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?)\s*\((\d{4})\)'
    author_years = re.findall(author_year_pattern, text)
    for author, year in author_years:
        found.append(f"Citation: {author} ({year})")
        year_int = int(year)
        if year_int > 2025:
            issues.append(f"Future citation: {author} ({year})")
        if year_int < 1900:
            issues.append(f"Unlikely old citation: {author} ({year})")
    
    # 4. Check for fake patterns
    fake_patterns = [
        (r'\(Smith et al\., \d{4}\)', "Generic 'Smith et al.' citation"),
        (r'Journal of [A-Z][a-z]+ Studies', "Suspicious generic journal name"),
        (r'DOI:\s*10\.1234/', "Known fake DOI prefix"),
        (r'example\.com|test\.com|fake\.com', "Test/fake domain"),
    ]
    
    for pattern, description in fake_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            issues.append(description)
    
    # Calculate score
    if not found:
        citation_type = "none"
        score = 0.5  # Neutral if no citations expected
    else:
        citation_type = "doi" if dois else "url" if urls else "author_year"
        # Penalize for issues
        score = max(0, 1 - len(issues) * 0.2)
    
    return CitationResult(
        is_valid=len(issues) == 0 and len(found) > 0,
        score=score,
        citation_type=citation_type,
        found_citations=found,
        issues=issues,
    )


def validate_doi_format(doi: str) -> bool:
    """Check if DOI follows valid format."""
    # DOI should be 10.XXXX/something
    pattern = r'^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$'
    return bool(re.match(pattern, doi))


def validate_url_format(url: str) -> list[str]:
    """Check URL for suspicious patterns."""
    issues = []
    
    try:
        parsed = urlparse(url)
        
        # Check domain
        domain = parsed.netloc.lower()
        
        # Suspicious TLDs
        suspicious_tlds = ['.xyz', '.tk', '.ml', '.ga', '.cf']
        if any(domain.endswith(tld) for tld in suspicious_tlds):
            issues.append(f"Suspicious TLD in URL: {url}")
        
        # Known fake domains
        fake_domains = ['example.com', 'test.com', 'fake.com', 'placeholder.org']
        if any(fake in domain for fake in fake_domains):
            issues.append(f"Test/fake domain: {url}")
        
        # Check for overly long random strings (often fake)
        if len(parsed.path) > 100:
            issues.append(f"Suspiciously long URL path: {url[:50]}...")
        
    except Exception:
        issues.append(f"Malformed URL: {url}")
    
    return issues


def extract_claims_with_citations(text: str) -> list[dict]:
    """
    Extract claims that have citations and pair them.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    results = []
    for sentence in sentences:
        # Check if sentence has a citation
        has_citation = bool(
            re.search(r'\([A-Z][a-z]+.*?\d{4}\)', sentence) or
            re.search(r'10\.\d{4,}/', sentence) or
            re.search(r'https?://', sentence)
        )
        
        if has_citation:
            results.append({
                "claim": sentence.strip(),
                "has_citation": True,
                "validation": validate_citations(sentence),
            })
    
    return results
