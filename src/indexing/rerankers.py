"""Reranking strategies for improving retrieval precision.

Rerankers take initial retrieval results and re-score them using
a more expensive but more accurate model. Cross-encoders process
query-document pairs jointly, capturing interactions that bi-encoder
embeddings miss.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RankedResult:
    """A retrieval result with relevance score."""

    text: str
    score: float
    metadata: dict


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[RankedResult]:
        ...


class CrossEncoderReranker(BaseReranker):
    """Rerank using a cross-encoder model (sentence-transformers).

    Cross-encoders process (query, document) pairs jointly and produce
    a relevance score. Much more accurate than bi-encoder similarity
    but O(n) per query — only practical as a second-stage ranker.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("pip install sentence-transformers")
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[RankedResult]:
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            RankedResult(text=doc, score=float(score), metadata={})
            for doc, score in ranked[:top_k]
        ]


class CohereReranker(BaseReranker):
    """Rerank using Cohere's rerank API.

    High quality but requires external API call.
    Consider latency and data residency implications.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "rerank-v3.5"):
        self.api_key = api_key
        self.model = model

    def rerank(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[RankedResult]:
        import cohere

        client = cohere.Client(api_key=self.api_key)
        response = client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_n=top_k,
        )

        return [
            RankedResult(
                text=documents[r.index],
                score=r.relevance_score,
                metadata={"original_index": r.index},
            )
            for r in response.results
        ]


class ReciprocalRankFusion:
    """Fuse rankings from multiple retrievers using Reciprocal Rank Fusion.

    RRF combines results from different retrieval methods (e.g., BM25 + vector)
    without requiring score normalization. Each result's fused score is:
        score = sum(1 / (k + rank_i)) for each ranker i

    k=60 is the standard constant from the original RRF paper.
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, *ranked_lists: List[str], top_k: int = 10) -> List[RankedResult]:
        scores: dict[str, float] = {}

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                if doc not in scores:
                    scores[doc] = 0.0
                scores[doc] += 1.0 / (self.k + rank + 1)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [
            RankedResult(text=doc, score=score, metadata={"fusion": "rrf"})
            for doc, score in sorted_docs[:top_k]
        ]
