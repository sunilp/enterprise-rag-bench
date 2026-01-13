"""Tests for retrieval patterns."""

import pytest
from src.retrieval.naive_rag import NaiveRAG, RAGResponse


class MockVectorStore:
    def similarity_search(self, query, k=5):
        return [
            {"text": "BCBS 239 requires banks to aggregate risk data accurately.", "score": 0.9},
            {"text": "Data quality controls must be in place for risk reporting.", "score": 0.8},
        ]


class MockLLM:
    def generate(self, prompt):
        return "BCBS 239 requires accurate risk data aggregation with quality controls."


class TestNaiveRAG:
    def test_basic_query(self):
        rag = NaiveRAG(
            vector_store=MockVectorStore(),
            llm=MockLLM(),
            top_k=2,
        )
        response = rag.query("What does BCBS 239 require?")

        assert isinstance(response, RAGResponse)
        assert len(response.answer) > 0
        assert len(response.source_documents) == 2
        assert response.metadata["pattern"] == "naive_rag"

    def test_custom_prompt_template(self):
        custom_template = "Context: {context}\n\nQ: {question}\n\nA:"
        rag = NaiveRAG(
            vector_store=MockVectorStore(),
            llm=MockLLM(),
            prompt_template=custom_template,
        )
        response = rag.query("test question")
        assert response.answer is not None
