"""Parent Document Retriever: retrieve child chunks, return parent context.

Indexes small chunks for precise retrieval but returns the larger parent
document (or section) for richer LLM context. Solves the chunk-size
trade-off: small chunks retrieve better, but large chunks generate better.

Good for: long documents where context matters (contracts, regulations).
Poor for: highly heterogeneous document collections with no clear hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from uuid import uuid4

from .naive_rag import RAGResponse


@dataclass
class ParentChild:
    """A parent document with its child chunks."""

    parent_id: str
    parent_text: str
    children: List[str]
    metadata: dict


class ParentDocumentRetriever:
    """Retrieve child chunks, return parent documents as context.

    Architecture:
    1. Split documents into large parent chunks
    2. Split each parent into smaller child chunks
    3. Index child chunks in vector store
    4. On query: retrieve child chunks, look up their parents
    5. Deduplicate parents, pass to LLM
    """

    def __init__(
        self,
        parent_chunker: Any,
        child_chunker: Any,
        vector_store: Any,
        llm: Any,
        top_k_children: int = 10,
        max_parents: int = 3,
    ):
        self.parent_chunker = parent_chunker
        self.child_chunker = child_chunker
        self.vector_store = vector_store
        self.llm = llm
        self.top_k_children = top_k_children
        self.max_parents = max_parents
        self._parent_store: Dict[str, ParentChild] = {}
        self._child_to_parent: Dict[str, str] = {}

    def index_documents(self, documents: List[str]) -> None:
        """Index documents by splitting into parent/child chunks."""
        for doc in documents:
            parent_chunks = self.parent_chunker.chunk(doc)

            for parent_chunk in parent_chunks:
                parent_id = str(uuid4())
                child_chunks = self.child_chunker.chunk(parent_chunk.text)

                child_texts = [c.text for c in child_chunks]
                self._parent_store[parent_id] = ParentChild(
                    parent_id=parent_id,
                    parent_text=parent_chunk.text,
                    children=child_texts,
                    metadata=parent_chunk.metadata,
                )

                # Index each child with reference to parent
                for child in child_chunks:
                    child_id = str(uuid4())
                    self._child_to_parent[child_id] = parent_id
                    self.vector_store.add_texts(
                        [child.text],
                        metadatas=[{"child_id": child_id, "parent_id": parent_id}],
                    )

    def query(self, question: str) -> RAGResponse:
        """Retrieve child chunks, return parent context."""
        child_results = self.vector_store.similarity_search(
            question, k=self.top_k_children
        )

        # Deduplicate parents while preserving order
        seen_parents = set()
        parent_texts = []

        for result in child_results:
            parent_id = result.get("metadata", {}).get("parent_id")
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent = self._parent_store.get(parent_id)
                if parent:
                    parent_texts.append(parent.parent_text)

            if len(parent_texts) >= self.max_parents:
                break

        context = "\n\n---\n\n".join(parent_texts)
        prompt = f"""Answer based on the following context.

Context:
{context}

Question: {question}

Answer:"""

        answer = self.llm.generate(prompt)

        return RAGResponse(
            answer=answer,
            source_documents=[{"text": t} for t in parent_texts],
            metadata={
                "pattern": "parent_document",
                "num_children_retrieved": len(child_results),
                "num_parents_used": len(parent_texts),
            },
        )
