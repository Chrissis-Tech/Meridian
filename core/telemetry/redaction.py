"""
Meridian Telemetry - Output Redaction

Provides redaction of sensitive information in model outputs.
"""

import re
from typing import Optional


# Redaction patterns
PATTERNS = {
    "email": {
        "pattern": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "replacement": "[EMAIL_REDACTED]",
    },
    "phone": {
        "pattern": r'\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b',
        "replacement": "[PHONE_REDACTED]",
    },
    "ssn": {
        "pattern": r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',
        "replacement": "[SSN_REDACTED]",
    },
    "credit_card": {
        "pattern": r'\b(?:\d{4}[-.\s]?){3}\d{4}\b',
        "replacement": "[CC_REDACTED]",
    },
    "api_key": {
        "pattern": r'\b(?:sk-|pk-)[\w-]{20,}\b',
        "replacement": "[API_KEY_REDACTED]",
    },
    "ip_address": {
        "pattern": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        "replacement": "[IP_REDACTED]",
    },
    "url_with_auth": {
        "pattern": r'https?://[^:]+:[^@]+@[^\s]+',
        "replacement": "[AUTH_URL_REDACTED]",
    },
}


class Redactor:
    """Redacts sensitive information from text."""
    
    def __init__(
        self,
        patterns: Optional[dict] = None,
        enabled: bool = True,
    ):
        self.patterns = patterns or PATTERNS
        self.enabled = enabled
        self._compiled = {
            name: re.compile(p["pattern"], re.IGNORECASE)
            for name, p in self.patterns.items()
        }
    
    def redact(self, text: str) -> str:
        """Redact all sensitive patterns from text."""
        if not self.enabled or not text:
            return text
        
        result = text
        for name, pattern in self._compiled.items():
            replacement = self.patterns[name]["replacement"]
            result = pattern.sub(replacement, result)
        
        return result
    
    def redact_dict(self, data: dict) -> dict:
        """Recursively redact values in a dictionary."""
        if not self.enabled:
            return data
        
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.redact(value)
            elif isinstance(value, dict):
                result[key] = self.redact_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.redact(v) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                result[key] = value
        return result
    
    def add_pattern(self, name: str, pattern: str, replacement: str):
        """Add a custom redaction pattern."""
        self.patterns[name] = {
            "pattern": pattern,
            "replacement": replacement,
        }
        self._compiled[name] = re.compile(pattern, re.IGNORECASE)
    
    def remove_pattern(self, name: str):
        """Remove a redaction pattern."""
        if name in self.patterns:
            del self.patterns[name]
            del self._compiled[name]
    
    def get_matches(self, text: str) -> dict:
        """Find all matches without redacting (for debugging)."""
        matches = {}
        for name, pattern in self._compiled.items():
            found = pattern.findall(text)
            if found:
                matches[name] = found
        return matches


# Global redactor instance
_redactor: Optional[Redactor] = None


def get_redactor(enabled: bool = True) -> Redactor:
    """Get or create the global redactor."""
    global _redactor
    if _redactor is None:
        _redactor = Redactor(enabled=enabled)
    return _redactor


def redact_output(text: str) -> str:
    """Convenience function to redact text."""
    return get_redactor().redact(text)


def redact_for_logging(data: dict) -> dict:
    """Convenience function to redact a dictionary."""
    return get_redactor().redact_dict(data)
