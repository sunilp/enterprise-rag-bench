"""Hybrid Search: BM25 + vector similarity with Reciprocal Rank Fusion.

Combines lexical search (BM25) with semantic search (embeddings) to
capture both exact keyword matches and meaning-based similarity.
RRF merges the two result sets without requiring score normalization.

Good for: most enterprise use cases — handles jargon, acronyms, and
          domain-specific terminology that pure semantic search misses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi


@dataclass
class SearchResult:
    text: str
    score: float
    source: str
    metadata: dict = field(default_factory=dict)


class HybridSearch:
    """Hybrid retrieval combining BM25 and vector similarity.

    Uses Reciprocal Rank Fusion (RRF) to merge results from both
    retrieval methods. RRF is robust to score distribution differences
    between retrievers.
    """

    def __init__(
        self,
        documents: List[str],
        embedder: Any,
        vector_store: Any,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        self.documents = documents
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

        # Build BM25 index
        tokenized = [doc.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized)

    def _bm25_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Lexical search using BM25."""
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        return [
            SearchResult(
                text=self.documents[i],
                score=float(scores[i]),
                source="bm25",
                metadata={"doc_index": i},
            )
            for i in top_indices
            if scores[i] > 0
        ]

    def _vector_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Semantic search using vector similarity."""
        results = self.vector_store.similarity_search(query, k=top_k)
        return [
            SearchResult(
                text=r.get("text", r.get("page_content", "")),
                score=r.get("score", 0.0),
                source="vector",
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]

    def _reciprocal_rank_fusion(
        self, *result_lists: List[SearchResult], top_k: int = 10
    ) -> List[SearchResult]:
        """Merge multiple ranked lists using RRF.

        score(d) = sum(1 / (k + rank_i(d))) for each ranker i
        """
        scores: Dict[str, float] = {}
        text_map: Dict[str, SearchResult] = {}

        for results in result_lists:
            for rank, result in enumerate(results):
                key = result.text[:200]  # Use truncated text as key
                if key not in scores:
                    scores[key] = 0.0
                    text_map[key] = result
                scores[key] += 1.0 / (self.rrf_k + rank + 1)

        sorted_keys = sorted(scores, key=scores.get, reverse=True)[:top_k]

        return [
            SearchResult(
                text=text_map[k].text,
                score=scores[k],
                source="hybrid_rrf",
                metadata=text_map[k].metadata,
            )
            for k in sorted_keys
        ]

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Run hybrid search combining BM25 and vector retrieval."""
        bm25_results = self._bm25_search(query, top_k=top_k * 2)
        vector_results = self._vector_search(query, top_k=top_k * 2)

        return self._reciprocal_rank_fusion(bm25_results, vector_results, top_k=top_k)
