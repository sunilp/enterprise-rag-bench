"""Naive RAG: basic retrieve-and-generate pattern.

The simplest RAG approach — embed the query, find similar chunks
via vector search, pass them as context to the LLM. No reranking,
no verification, no self-correction.

Good for: prototyping, low-stakes internal tools, simple Q&A.
Poor for: complex reasoning, multi-hop questions, regulated outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol


class VectorStore(Protocol):
    """Minimal vector store interface."""

    def similarity_search(self, query: str, k: int = 5) -> List[dict]:
        ...


class LLM(Protocol):
    """Minimal LLM interface."""

    def generate(self, prompt: str) -> str:
        ...


@dataclass
class RAGResponse:
    """Response from a RAG pipeline."""

    answer: str
    source_documents: List[dict]
    metadata: dict


class NaiveRAG:
    """Basic retrieve-and-generate RAG implementation.

    Steps:
    1. Embed query using the vector store's embedding model
    2. Retrieve top-k similar chunks
    3. Format chunks as context in the prompt
    4. Generate response via LLM
    """

    PROMPT_TEMPLATE = """Answer the question based only on the following context.
If the context doesn't contain enough information, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""

    def __init__(
        self,
        vector_store: VectorStore,
        llm: LLM,
        top_k: int = 5,
        prompt_template: Optional[str] = None,
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k
        self.prompt_template = prompt_template or self.PROMPT_TEMPLATE

    def query(self, question: str) -> RAGResponse:
        # Retrieve
        docs = self.vector_store.similarity_search(question, k=self.top_k)

        # Format context
        context = "\n\n---\n\n".join(
            doc.get("text", doc.get("page_content", "")) for doc in docs
        )

        # Generate
        prompt = self.prompt_template.format(context=context, question=question)
        answer = self.llm.generate(prompt)

        return RAGResponse(
            answer=answer,
            source_documents=docs,
            metadata={
                "pattern": "naive_rag",
                "num_retrieved": len(docs),
                "top_k": self.top_k,
            },
        )
