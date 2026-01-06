"""Embedding providers for document indexing and retrieval.

Wraps multiple embedding backends behind a common interface to enable
benchmarking across providers without changing retrieval code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseEmbedder(ABC):
    """Common interface for all embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed a single text string."""
        ...

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed a batch of texts. Override for provider-native batching."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend([self.embed(t) for t in batch])
        return results

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embeddings via sentence-transformers.

    Good for: cost control, data residency, offline usage.
    Models: all-MiniLM-L6-v2 (384d), bge-base-en-v1.5 (768d),
            e5-large-v2 (1024d).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("pip install sentence-transformers")
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> List[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        embeddings = self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._dim


class VertexAIEmbedder(BaseEmbedder):
    """Google Vertex AI text embeddings.

    Uses textembedding-gecko or text-embedding-004 models.
    Requires: google-cloud-aiplatform, GCP credentials.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-004",
        project: Optional[str] = None,
        location: str = "us-central1",
    ):
        self.model_name = model_name
        self.project = project
        self.location = location
        self._dim = 768  # Default for text-embedding-004

    def embed(self, text: str) -> List[float]:
        from vertexai.language_models import TextEmbeddingModel

        model = TextEmbeddingModel.from_pretrained(self.model_name)
        result = model.get_embeddings([text])
        return result[0].values

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        from vertexai.language_models import TextEmbeddingModel

        model = TextEmbeddingModel.from_pretrained(self.model_name)
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results = model.get_embeddings(batch)
            all_embeddings.extend([r.values for r in results])
        return all_embeddings

    @property
    def dimension(self) -> int:
        return self._dim


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings (text-embedding-3-small/large).

    Best quality per benchmark, but external API dependency.
    Consider data residency implications in regulated environments.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key
        self._dim = 1536 if "small" in model else 3072

    def embed(self, text: str) -> List[float]:
        import openai

        client = openai.OpenAI(api_key=self._api_key)
        response = client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        import openai

        client = openai.OpenAI(api_key=self._api_key)
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(input=batch, model=self.model)
            all_embeddings.extend([d.embedding for d in response.data])
        return all_embeddings

    @property
    def dimension(self) -> int:
        return self._dim
