"""Tests for chunking strategies."""

import pytest
from src.indexing.chunkers import (
    Chunk,
    FixedSizeChunker,
    RecursiveCharacterChunker,
    SentenceWindowChunker,
    SemanticChunker,
    DocumentStructureChunker,
)


SAMPLE_TEXT = """The Basel Committee on Banking Supervision published BCBS 239 in January 2013. The principles address risk data aggregation and risk reporting practices at banks.

Principle 1 requires strong governance arrangements. Banks should establish a framework for risk data aggregation and reporting that is approved by the board.

Principle 2 addresses data architecture and IT infrastructure. Banks should design and maintain data architecture that fully supports risk data aggregation capabilities.

Principle 3 covers accuracy and integrity. Risk data should be accurate and reliable. Controls should be in place to ensure data quality.

Principle 4 requires completeness. Risk data aggregation should capture all material risks across the banking group."""


class TestFixedSizeChunker:
    def test_basic_chunking(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(len(c.text) <= 100 for c in chunks)

    def test_overlap_creates_redundancy(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=30)
        chunks = chunker.chunk(SAMPLE_TEXT)
        # With overlap, consecutive chunks should share some text
        for i in range(len(chunks) - 1):
            end_of_current = chunks[i].text[-30:]
            assert end_of_current in chunks[i + 1].text or len(chunks[i].text) < 100

    def test_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, overlap=100)

    def test_short_text_single_chunk(self):
        chunker = FixedSizeChunker(chunk_size=1000)
        chunks = chunker.chunk("Short text.")
        assert len(chunks) == 1

    def test_metadata_includes_strategy(self):
        chunker = FixedSizeChunker(chunk_size=100)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert all(c.metadata["strategy"] == "fixed_size" for c in chunks)


class TestRecursiveCharacterChunker:
    def test_respects_chunk_size(self):
        chunker = RecursiveCharacterChunker(chunk_size=200)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert all(len(c.text) <= 200 for c in chunks)

    def test_preserves_paragraphs_when_possible(self):
        chunker = RecursiveCharacterChunker(chunk_size=500)
        chunks = chunker.chunk(SAMPLE_TEXT)
        # With 500 char limit, should split at paragraph boundaries
        assert len(chunks) >= 2

    def test_empty_text(self):
        chunker = RecursiveCharacterChunker(chunk_size=100)
        chunks = chunker.chunk("")
        assert len(chunks) == 0


class TestSentenceWindowChunker:
    def test_window_includes_neighbors(self):
        chunker = SentenceWindowChunker(window_size=1)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) > 0
        # Center sentence should be in the metadata
        for c in chunks:
            assert "center_sentence" in c.metadata
            assert c.metadata["center_sentence"] in c.text

    def test_window_size_zero(self):
        chunker = SentenceWindowChunker(window_size=0)
        chunks = chunker.chunk(SAMPLE_TEXT)
        # Each chunk should be approximately one sentence
        for c in chunks:
            assert c.metadata["window_range"][1] - c.metadata["window_range"][0] == 1


class TestDocumentStructureChunker:
    def test_splits_on_headers(self):
        doc = "# Introduction\nSome intro text.\n\n# Methods\nMethodology here.\n\n# Results\nResults here."
        chunker = DocumentStructureChunker(max_chunk_size=500)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 3
        sections = [c.metadata.get("section") for c in chunks]
        assert "Introduction" in sections
        assert "Methods" in sections

    def test_no_headers_fallback(self):
        chunker = DocumentStructureChunker(max_chunk_size=100)
        chunks = chunker.chunk(SAMPLE_TEXT)
        # Should fall back to recursive chunking
        assert len(chunks) > 0

    def test_large_section_gets_split(self):
        doc = "# Big Section\n" + "Word " * 500
        chunker = DocumentStructureChunker(max_chunk_size=200)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1


class TestSemanticChunker:
    def test_requires_embed_fn(self):
        chunker = SemanticChunker()
        with pytest.raises(ValueError, match="embed_fn"):
            chunker.chunk(SAMPLE_TEXT)

    def test_with_mock_embeddings(self):
        # Mock: return different embeddings for different sentences
        call_count = [0]
        def mock_embed(text):
            call_count[0] += 1
            # Alternate between two "topics"
            if call_count[0] % 3 == 0:
                return [1.0, 0.0, 0.0]
            return [0.0, 1.0, 0.0]

        chunker = SemanticChunker(threshold=0.5, embed_fn=mock_embed)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) > 1
