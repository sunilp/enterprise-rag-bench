"""Chunking strategies for document processing in RAG pipelines.

Implements 5 strategies with different trade-offs between chunk quality,
processing speed, and context preservation. Each chunker returns a list
of Chunk objects with metadata for downstream evaluation.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class Chunk:
    """A document chunk with positional and strategy metadata."""

    text: str
    index: int
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Rough token count (whitespace split). Use tiktoken for precision."""
        return len(self.text.split())

    def __len__(self) -> int:
        return len(self.text)


class BaseChunker(ABC):
    """Base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__})"


class FixedSizeChunker(BaseChunker):
    """Split text into fixed-size character chunks with configurable overlap.

    Simple and fast. Works well when document structure is uniform.
    Poor at preserving semantic boundaries — sentences and paragraphs
    get split mid-thought.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        chunks = []
        start = 0
        idx = 0
        step = self.chunk_size - self.overlap

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=idx,
                        metadata={
                            "strategy": "fixed_size",
                            "char_start": start,
                            "char_end": end,
                        },
                    )
                )
                idx += 1
            start += step

        return chunks


class RecursiveCharacterChunker(BaseChunker):
    """Split by separators recursively until chunks fit within size limit.

    Tries the most meaningful separators first (paragraph breaks, then
    line breaks, then sentences, then words). Produces semantically
    coherent chunks in most cases.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        raw = self._split_recursive(text, self.separators)
        return [
            Chunk(text=t, index=i, metadata={"strategy": "recursive"})
            for i, t in enumerate(raw)
            if t.strip()
        ]

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            return [text[: self.chunk_size]]

        sep = separators[0]
        parts = text.split(sep)
        merged: List[str] = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    if len(current) <= self.chunk_size:
                        merged.append(current)
                    else:
                        merged.extend(self._split_recursive(current, separators[1:]))
                current = part

        if current.strip():
            if len(current) <= self.chunk_size:
                merged.append(current)
            else:
                merged.extend(self._split_recursive(current, separators[1:]))

        return merged


class SentenceWindowChunker(BaseChunker):
    """Chunk at sentence level with a sliding window of surrounding context.

    Each chunk is centered on a single sentence but includes neighboring
    sentences for context. Useful when retrieval needs sentence-level
    precision but the LLM needs paragraph-level context.
    """

    _SENT_PATTERN = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, window_size: int = 3):
        self.window_size = window_size

    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        sentences = [s.strip() for s in self._SENT_PATTERN.split(text) if s.strip()]

        if not sentences:
            return []

        chunks = []
        for i, sent in enumerate(sentences):
            start = max(0, i - self.window_size)
            end = min(len(sentences), i + self.window_size + 1)
            window_text = " ".join(sentences[start:end])
            chunks.append(
                Chunk(
                    text=window_text,
                    index=i,
                    metadata={
                        "strategy": "sentence_window",
                        "center_sentence": sent,
                        "window_range": [start, end],
                    },
                )
            )

        return chunks


class SemanticChunker(BaseChunker):
    """Split by semantic similarity between consecutive sentences.

    Computes embeddings for each sentence, then finds breakpoints where
    cosine similarity drops below a threshold — indicating a topic shift.

    Requires an embedding function: (str) -> List[float].
    """

    def __init__(self, threshold: float = 0.5, embed_fn: Optional[Callable] = None):
        self.threshold = threshold
        self._embed_fn = embed_fn

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x**2 for x in a) ** 0.5
        norm_b = sum(x**2 for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        embed_fn = kwargs.get("embed_fn", self._embed_fn)
        if embed_fn is None:
            raise ValueError("SemanticChunker requires an embed_fn")

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) <= 1:
            return [Chunk(text=text, index=0, metadata={"strategy": "semantic"})]

        embeddings = [embed_fn(s) for s in sentences]

        groups: List[List[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < self.threshold:
                groups.append([sentences[i]])
            else:
                groups[-1].append(sentences[i])

        return [
            Chunk(
                text=" ".join(group),
                index=i,
                metadata={"strategy": "semantic", "num_sentences": len(group)},
            )
            for i, group in enumerate(groups)
        ]


class DocumentStructureChunker(BaseChunker):
    """Split by document structure markers (markdown headers, sections).

    Preserves section boundaries and includes the header as context
    in each chunk. Falls back to recursive splitting for sections
    that exceed the max chunk size.
    """

    _HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(self, max_chunk_size: int = 1024):
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        headers = list(self._HEADER_RE.finditer(text))

        if not headers:
            fallback = RecursiveCharacterChunker(chunk_size=self.max_chunk_size)
            return fallback.chunk(text)

        sections: List[tuple] = []

        # Content before first header
        if headers[0].start() > 0:
            preamble = text[: headers[0].start()].strip()
            if preamble:
                sections.append((0, "preamble", preamble))

        for i, match in enumerate(headers):
            start = match.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            level = len(match.group(1))
            title = match.group(2).strip()
            content = text[start:end].strip()
            sections.append((level, title, content))

        chunks: List[Chunk] = []
        for level, title, content in sections:
            if len(content) <= self.max_chunk_size:
                chunks.append(
                    Chunk(
                        text=content,
                        index=len(chunks),
                        metadata={
                            "strategy": "document_structure",
                            "section": title,
                            "level": level,
                        },
                    )
                )
            else:
                fallback = RecursiveCharacterChunker(chunk_size=self.max_chunk_size)
                sub_chunks = fallback.chunk(content)
                for sc in sub_chunks:
                    sc.metadata.update(
                        {"parent_section": title, "level": level, "strategy": "document_structure"}
                    )
                    sc.index = len(chunks)
                    chunks.append(sc)

        return chunks
