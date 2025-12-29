"""Meridian Verify Package - Factual Verification"""

from .claims import (
    Claim,
    extract_claims,
    extract_entities,
    split_into_sentences,
)
from .retrieval import (
    Evidence,
    RetrievalResult,
    BM25Retriever,
    EmbeddingRetriever,
    create_retriever,
)
from .grounding_score import (
    SupportLevel,
    GroundingResult,
    FactualityReport,
    compute_grounding_score,
    generate_factuality_report,
)

__all__ = [
    "Claim",
    "extract_claims",
    "extract_entities",
    "split_into_sentences",
    "Evidence",
    "RetrievalResult",
    "BM25Retriever",
    "EmbeddingRetriever",
    "create_retriever",
    "SupportLevel",
    "GroundingResult",
    "FactualityReport",
    "compute_grounding_score",
    "generate_factuality_report",
]
