"""
Meridian Reliability - Determinism Levels

Defines operational determinism profiles for LLM evaluation.
"""

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from Levenshtein import distance as levenshtein_distance


class DeterminismProfile(Enum):
    """
    Three-level determinism classification for LLM outputs.
    
    - STRICT: Exact reproducibility (hash match after normalization)
    - QUASI: Reproducibility within tolerance (same semantics, minor variations)
    - STOCHASTIC: Intentional variation (temperature > 0, sampling)
    """
    STRICT = "strict"
    QUASI = "quasi"
    STOCHASTIC = "stochastic"


@dataclass
class DeterminismConfig:
    """Configuration for determinism checks."""
    num_replications: int = 5
    # Strict: hash comparison
    normalize_whitespace: bool = True
    normalize_case: bool = False
    # Quasi: tolerance thresholds
    levenshtein_threshold: float = 0.05  # max edit distance as fraction of length
    semantic_threshold: float = 0.95  # min cosine similarity
    json_canonical: bool = True


@dataclass
class DeterminismResult:
    """Result of a determinism check."""
    profile: DeterminismProfile
    num_replications: int
    unique_outputs: int
    strict_match: bool
    quasi_match: bool
    match_rate: float  # fraction of outputs matching mode
    details: dict


def normalize_output(text: str, config: DeterminismConfig) -> str:
    """Normalize output for comparison."""
    result = text.strip()
    
    if config.normalize_whitespace:
        # Collapse whitespace
        import re
        result = re.sub(r'\s+', ' ', result)
    
    if config.normalize_case:
        result = result.lower()
    
    if config.json_canonical:
        # Try to canonicalize JSON
        import json
        try:
            parsed = json.loads(result)
            result = json.dumps(parsed, sort_keys=True, separators=(',', ':'))
        except (json.JSONDecodeError, TypeError):
            pass
    
    return result


def hash_output(text: str) -> str:
    """Generate hash of normalized output."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class DeterminismChecker:
    """
    Check output determinism by running multiple replications.
    
    Determinism levels:
    - STRICT: All outputs hash-identical after normalization
    - QUASI: All outputs within tolerance (edit distance or semantic)
    - STOCHASTIC: Significant variation across runs
    """
    
    def __init__(self, config: Optional[DeterminismConfig] = None):
        self.config = config or DeterminismConfig()
        self._similarity_model = None
    
    def check(
        self,
        generate_fn: Callable[[], str],
        n: Optional[int] = None,
    ) -> DeterminismResult:
        """
        Run n replications and assess determinism.
        
        Args:
            generate_fn: Function that generates model output
            n: Number of replications (defaults to config)
        
        Returns:
            DeterminismResult with profile classification
        """
        n = n or self.config.num_replications
        outputs = [generate_fn() for _ in range(n)]
        
        return self.check_outputs(outputs)
    
    def check_outputs(self, outputs: list[str]) -> DeterminismResult:
        """Check determinism given a list of outputs."""
        n = len(outputs)
        
        # Normalize all outputs
        normalized = [normalize_output(o, self.config) for o in outputs]
        
        # Hash-based strict check
        hashes = [hash_output(o) for o in normalized]
        unique_hashes = set(hashes)
        strict_match = len(unique_hashes) == 1
        
        # Find mode (most common output)
        from collections import Counter
        hash_counts = Counter(hashes)
        mode_hash, mode_count = hash_counts.most_common(1)[0]
        match_rate = mode_count / n
        
        # Quasi check: all outputs within tolerance of mode
        mode_text = normalized[hashes.index(mode_hash)]
        quasi_match = self._check_quasi_match(normalized, mode_text)
        
        # Classify profile
        if strict_match:
            profile = DeterminismProfile.STRICT
        elif quasi_match:
            profile = DeterminismProfile.QUASI
        else:
            profile = DeterminismProfile.STOCHASTIC
        
        return DeterminismResult(
            profile=profile,
            num_replications=n,
            unique_outputs=len(unique_hashes),
            strict_match=strict_match,
            quasi_match=quasi_match,
            match_rate=match_rate,
            details={
                "hashes": hashes,
                "unique_count": len(unique_hashes),
                "mode_count": mode_count,
            }
        )
    
    def _check_quasi_match(self, normalized: list[str], mode_text: str) -> bool:
        """Check if all outputs are within quasi-deterministic tolerance."""
        for text in normalized:
            if text == mode_text:
                continue
            
            # Edit distance check
            max_len = max(len(text), len(mode_text), 1)
            edit_dist = levenshtein_distance(text, mode_text)
            edit_ratio = edit_dist / max_len
            
            if edit_ratio > self.config.levenshtein_threshold:
                # Try semantic similarity as fallback
                if not self._check_semantic_similarity(text, mode_text):
                    return False
        
        return True
    
    def _check_semantic_similarity(self, text1: str, text2: str) -> bool:
        """Check semantic similarity using embeddings."""
        try:
            if self._similarity_model is None:
                from sentence_transformers import SentenceTransformer
                self._similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            embeddings = self._similarity_model.encode([text1, text2])
            similarity = float(embeddings[0] @ embeddings[1])
            return similarity >= self.config.semantic_threshold
        except ImportError:
            # Fallback: if no sentence-transformers, be conservative
            return False


def infer_determinism_profile(
    temperature: float,
    top_p: float = 1.0,
    has_seed: bool = False,
) -> DeterminismProfile:
    """
    Infer expected determinism profile from generation config.
    
    Note: This is a prior expectation; actual determinism should be verified.
    """
    if temperature == 0 and top_p == 1.0:
        if has_seed:
            return DeterminismProfile.STRICT
        return DeterminismProfile.QUASI
    return DeterminismProfile.STOCHASTIC
