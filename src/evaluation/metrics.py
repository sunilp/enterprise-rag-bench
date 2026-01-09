"""RAG evaluation metrics.

Implements core metrics for evaluating RAG pipeline quality:
- Faithfulness: is the answer grounded in the retrieved context?
- Relevance: does the answer address the question?
- Groundedness: can each claim be traced to a source?
- Context Precision: are the retrieved documents relevant?

Each metric returns a float in [0, 1].
"""

from __future__ import annotations

import re
from typing import Any, List, Optional


def faithfulness(
    answer: str,
    context: List[str],
    llm: Optional[Any] = None,
) -> float:
    """Measure whether the answer is faithful to the provided context.

    Uses claim extraction + verification approach:
    1. Extract atomic claims from the answer
    2. Check if each claim is supported by the context
    3. Score = supported_claims / total_claims
    """
    if not answer.strip() or not context:
        return 0.0

    if llm is None:
        # Fallback: simple word overlap heuristic
        return _word_overlap_score(answer, " ".join(context))

    # LLM-based evaluation
    claims_prompt = f"""Extract atomic factual claims from this answer.
List each claim on a new line prefixed with "- ".

Answer: {answer}

Claims:"""
    claims_text = llm.generate(claims_prompt)
    claims = [c.strip("- ").strip() for c in claims_text.strip().split("\n") if c.strip()]

    if not claims:
        return 0.0

    context_text = "\n\n".join(context)
    supported = 0

    for claim in claims:
        verify_prompt = f"""Is the following claim supported by the context?
Respond with only YES or NO.

Context: {context_text}

Claim: {claim}

Supported:"""
        result = llm.generate(verify_prompt).strip().upper()
        if result.startswith("YES"):
            supported += 1

    return supported / len(claims)


def relevance(
    answer: str,
    question: str,
    llm: Optional[Any] = None,
) -> float:
    """Measure whether the answer is relevant to the question."""
    if not answer.strip() or not question.strip():
        return 0.0

    if llm is None:
        return _word_overlap_score(answer, question)

    prompt = f"""Rate how well this answer addresses the question.
Score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).
Respond with just the number.

Question: {question}
Answer: {answer}

Score:"""
    response = llm.generate(prompt).strip()
    try:
        score = float(re.search(r"[\d.]+", response).group())
        return max(0.0, min(1.0, score))
    except (AttributeError, ValueError):
        return 0.5


def groundedness(
    answer: str,
    context: List[str],
    llm: Optional[Any] = None,
) -> float:
    """Measure whether each sentence in the answer can be traced to context.

    More granular than faithfulness — checks sentence-level grounding
    rather than claim-level support.
    """
    if not answer.strip() or not context:
        return 0.0

    sentences = re.split(r"(?<=[.!?])\s+", answer)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    if llm is None:
        context_text = " ".join(context)
        scores = [_word_overlap_score(s, context_text) for s in sentences]
        return sum(scores) / len(scores)

    context_text = "\n\n".join(context)
    grounded = 0

    for sentence in sentences:
        prompt = f"""Can this sentence be directly traced to the context below?
Respond YES or NO.

Context: {context_text}

Sentence: {sentence}

Traceable:"""
        result = llm.generate(prompt).strip().upper()
        if result.startswith("YES"):
            grounded += 1

    return grounded / len(sentences)


def context_precision(
    question: str,
    context: List[str],
    llm: Optional[Any] = None,
) -> float:
    """Measure whether the retrieved documents are relevant to the question.

    Precision@k: what fraction of retrieved documents are actually useful
    for answering the question?
    """
    if not context or not question.strip():
        return 0.0

    if llm is None:
        scores = [_word_overlap_score(question, doc) for doc in context]
        return sum(1 for s in scores if s > 0.1) / len(context)

    relevant = 0
    for doc in context:
        prompt = f"""Is this document relevant to answering the question?
Respond YES or NO.

Question: {question}
Document: {doc[:500]}

Relevant:"""
        result = llm.generate(prompt).strip().upper()
        if result.startswith("YES"):
            relevant += 1

    return relevant / len(context)


def _word_overlap_score(text_a: str, text_b: str) -> float:
    """Simple word overlap heuristic for when no LLM is available."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    # Remove stopwords
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                 "have", "has", "had", "do", "does", "did", "will", "would", "could",
                 "should", "may", "might", "shall", "can", "to", "of", "in", "for",
                 "on", "with", "at", "by", "from", "it", "this", "that", "and", "or",
                 "but", "not", "no", "if", "as", "so"}
    words_a -= stopwords
    words_b -= stopwords
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / max(len(words_a), len(words_b))
