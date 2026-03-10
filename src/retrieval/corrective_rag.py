"""Corrective RAG (CRAG): self-verification loop for retrieval quality.

After initial retrieval, an LLM evaluates whether the retrieved documents
actually answer the question. If confidence is low, triggers corrective
actions: refine the query, retrieve again, or fall back to direct generation.

Good for: high-stakes Q&A where wrong answers have consequences.
Trade-off: 2-3x latency due to verification step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, List

from .naive_rag import RAGResponse


class RetrievalQuality(Enum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@dataclass
class VerificationResult:
    quality: RetrievalQuality
    confidence: float
    reasoning: str


class CorrectiveRAG:
    """Self-correcting RAG with retrieval verification.

    Pipeline:
    1. Retrieve documents via vector search
    2. LLM evaluates retrieval quality (correct/ambiguous/incorrect)
    3. If correct: generate answer from retrieved docs
    4. If ambiguous: refine query, re-retrieve, then generate
    5. If incorrect: fall back to LLM knowledge (with disclosure)
    """

    VERIFICATION_PROMPT = """Evaluate whether the following retrieved documents
are relevant to answering the question.

Question: {question}

Retrieved Documents:
{documents}

Respond with exactly one of: CORRECT, AMBIGUOUS, or INCORRECT.
Then briefly explain why.

Format:
VERDICT: [CORRECT|AMBIGUOUS|INCORRECT]
CONFIDENCE: [0.0-1.0]
REASONING: [one sentence]"""

    REFINEMENT_PROMPT = """The following question did not get good retrieval results.
Rewrite it as a more specific search query that might find better documents.

Original question: {question}
Problem: Retrieved documents were not relevant.

Rewritten query:"""

    def __init__(
        self,
        vector_store: Any,
        llm: Any,
        top_k: int = 5,
        confidence_threshold: float = 0.6,
        max_retries: int = 1,
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries

    def _verify_retrieval(
        self, question: str, documents: List[str]
    ) -> VerificationResult:
        """Use LLM to assess whether retrieved docs answer the question."""
        docs_text = "\n\n".join(f"[Doc {i+1}]: {d}" for i, d in enumerate(documents))
        prompt = self.VERIFICATION_PROMPT.format(
            question=question, documents=docs_text
        )
        response = self.llm.generate(prompt)

        # Parse structured response
        quality = RetrievalQuality.AMBIGUOUS
        confidence = 0.5
        reasoning = response

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("VERDICT:"):
                verdict = line.split(":", 1)[1].strip().upper()
                quality = {
                    "CORRECT": RetrievalQuality.CORRECT,
                    "AMBIGUOUS": RetrievalQuality.AMBIGUOUS,
                    "INCORRECT": RetrievalQuality.INCORRECT,
                }.get(verdict, RetrievalQuality.AMBIGUOUS)
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return VerificationResult(
            quality=quality, confidence=confidence, reasoning=reasoning
        )

    def _refine_query(self, question: str) -> str:
        """Use LLM to generate a refined search query."""
        prompt = self.REFINEMENT_PROMPT.format(question=question)
        return self.llm.generate(prompt).strip()

    def query(self, question: str) -> RAGResponse:
        """Run corrective RAG pipeline."""
        docs = self.vector_store.similarity_search(question, k=self.top_k)
        doc_texts = [d.get("text", d.get("page_content", "")) for d in docs]

        verification = self._verify_retrieval(question, doc_texts)

        if verification.quality == RetrievalQuality.CORRECT:
            return self._generate(question, doc_texts, "corrective_rag_verified")

        if verification.quality == RetrievalQuality.AMBIGUOUS:
            # Try query refinement
            for attempt in range(self.max_retries):
                refined = self._refine_query(question)
                new_docs = self.vector_store.similarity_search(refined, k=self.top_k)
                new_texts = [d.get("text", d.get("page_content", "")) for d in new_docs]

                re_verify = self._verify_retrieval(question, new_texts)
                if re_verify.quality == RetrievalQuality.CORRECT:
                    return self._generate(
                        question, new_texts, "corrective_rag_refined"
                    )
                doc_texts = new_texts

            return self._generate(question, doc_texts, "corrective_rag_best_effort")

        # INCORRECT — fall back to LLM knowledge with disclosure
        return self._fallback_generate(question, verification.reasoning)

    def _generate(
        self, question: str, doc_texts: List[str], pattern: str
    ) -> RAGResponse:
        context = "\n\n---\n\n".join(doc_texts)
        prompt = f"Answer based on the context.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        answer = self.llm.generate(prompt)
        return RAGResponse(
            answer=answer,
            source_documents=[{"text": t} for t in doc_texts],
            metadata={"pattern": pattern},
        )

    def _fallback_generate(self, question: str, reason: str) -> RAGResponse:
        prompt = f"Answer this question using your knowledge: {question}"
        answer = self.llm.generate(prompt)
        disclaimer = (
            "\n\n[Note: This answer was generated without supporting documents. "
            f"Retrieved documents were not relevant: {reason}]"
        )
        return RAGResponse(
            answer=answer + disclaimer,
            source_documents=[],
            metadata={"pattern": "corrective_rag_fallback", "fallback_reason": reason},
        )
