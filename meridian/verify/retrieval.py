"""
Meridian Verify - Evidence Retrieval

Retrieves evidence for claim verification using BM25 or embeddings.
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class Evidence:
    """A piece of evidence retrieved for claim verification."""
    text: str
    source: str
    score: float
    rank: int


@dataclass
class RetrievalResult:
    """Result of evidence retrieval."""
    query: str
    evidences: list[Evidence]
    num_searched: int


class BM25Retriever:
    """
    BM25 retriever for evidence lookup.
    
    Simple implementation for offline use without external dependencies.
    """
    
    def __init__(
        self,
        documents: list[str],
        sources: Optional[list[str]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.documents = documents
        self.sources = sources or [f"doc_{i}" for i in range(len(documents))]
        self.k1 = k1
        self.b = b
        
        # Precompute
        self.doc_tokens = [self._tokenize(d) for d in documents]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        
        # Build inverted index
        self.df = {}  # document frequency
        self.index = {}  # term -> [(doc_id, freq), ...]
        
        for doc_id, tokens in enumerate(self.doc_tokens):
            seen = set()
            term_freq = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
                if token not in seen:
                    self.df[token] = self.df.get(token, 0) + 1
                    seen.add(token)
            
            for token, freq in term_freq.items():
                if token not in self.index:
                    self.index[token] = []
                self.index[token].append((doc_id, freq))
        
        self.N = len(documents)
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _idf(self, term: str) -> float:
        """Inverse document frequency."""
        df = self.df.get(term, 0)
        if df == 0:
            return 0
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)
    
    def _score(self, query_tokens: list[str], doc_id: int) -> float:
        """BM25 score for a document."""
        score = 0.0
        doc_len = self.doc_lens[doc_id]
        
        for token in query_tokens:
            if token not in self.index:
                continue
            
            # Find term frequency in this doc
            tf = 0
            for d_id, freq in self.index[token]:
                if d_id == doc_id:
                    tf = freq
                    break
            
            if tf == 0:
                continue
            
            idf = self._idf(token)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator
        
        return score
    
    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Retrieve top-k documents for query."""
        query_tokens = self._tokenize(query)
        
        # Score all documents
        scores = [(i, self._score(query_tokens, i)) for i in range(self.N)]
        scores.sort(key=lambda x: -x[1])
        
        evidences = []
        for rank, (doc_id, score) in enumerate(scores[:top_k]):
            if score > 0:
                evidences.append(Evidence(
                    text=self.documents[doc_id],
                    source=self.sources[doc_id],
                    score=score,
                    rank=rank + 1,
                ))
        
        return RetrievalResult(
            query=query,
            evidences=evidences,
            num_searched=self.N,
        )


class EmbeddingRetriever:
    """
    Embedding-based retriever using sentence-transformers.
    """
    
    def __init__(
        self,
        documents: list[str],
        sources: Optional[list[str]] = None,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.documents = documents
        self.sources = sources or [f"doc_{i}" for i in range(len(documents))]
        self.model_name = model_name
        self._model = None
        self._embeddings = None
    
    def _ensure_loaded(self):
        """Lazy load model and compute embeddings."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._embeddings = self._model.encode(self.documents, convert_to_tensor=True)
    
    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Retrieve top-k documents by semantic similarity."""
        self._ensure_loaded()
        
        query_embedding = self._model.encode([query], convert_to_tensor=True)
        
        # Compute cosine similarities
        import torch
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding, self._embeddings
        )
        
        # Get top-k
        top_indices = torch.argsort(similarities, descending=True)[:top_k]
        
        evidences = []
        for rank, idx in enumerate(top_indices):
            idx = int(idx)
            score = float(similarities[idx])
            if score > 0:
                evidences.append(Evidence(
                    text=self.documents[idx],
                    source=self.sources[idx],
                    score=score,
                    rank=rank + 1,
                ))
        
        return RetrievalResult(
            query=query,
            evidences=evidences,
            num_searched=len(self.documents),
        )


def create_retriever(
    documents: list[str],
    sources: Optional[list[str]] = None,
    method: str = "bm25",
) -> BM25Retriever | EmbeddingRetriever:
    """Factory function for retrievers."""
    if method == "bm25":
        return BM25Retriever(documents, sources)
    elif method == "embedding":
        return EmbeddingRetriever(documents, sources)
    else:
        raise ValueError(f"Unknown retrieval method: {method}")
