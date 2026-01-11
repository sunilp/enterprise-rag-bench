"""Hallucination detection for RAG outputs.

Cross-references the generated answer against source documents to
identify claims that aren't supported by the retrieved context.
Uses both heuristic and LLM-based approaches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class HallucinationReport:
    """Report of hallucination analysis."""
    has_hallucinations: bool
    hallucination_rate: float
    unsupported_claims: List[str]
    supported_claims: List[str]
    total_claims: int
    metadata: dict = field(default_factory=dict)


class HallucinationDetector:
    """Detect unsupported claims in RAG outputs.

    Two detection modes:
    1. Heuristic: entity and number cross-referencing (fast, moderate accuracy)
    2. LLM-based: claim extraction and verification (slow, high accuracy)
    """

    def __init__(self, llm: Optional[Any] = None, threshold: float = 0.3):
        self.llm = llm
        self.threshold = threshold

    def check(self, answer: str, context: List[str]) -> HallucinationReport:
        """Check answer for hallucinations against context."""
        if self.llm:
            return self._llm_check(answer, context)
        return self._heuristic_check(answer, context)

    def _heuristic_check(self, answer: str, context: List[str]) -> HallucinationReport:
        """Fast heuristic check using entity and number matching."""
        context_text = " ".join(context).lower()

        # Extract entities: numbers, dates, proper nouns (capitalized words)
        answer_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", answer))
        context_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", context_text))

        # Check which answer numbers appear in context
        unsupported = []
        supported = []

        for num in answer_numbers:
            if num in context_numbers:
                supported.append(f"Number '{num}' found in context")
            else:
                unsupported.append(f"Number '{num}' not found in context")

        # Check capitalized terms (potential named entities)
        answer_entities = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", answer))
        for entity in answer_entities:
            if entity.lower() in context_text:
                supported.append(f"Entity '{entity}' found in context")
            elif len(entity) > 3:  # Skip short words
                unsupported.append(f"Entity '{entity}' not found in context")

        total = len(supported) + len(unsupported)
        rate = len(unsupported) / total if total > 0 else 0.0

        return HallucinationReport(
            has_hallucinations=rate > self.threshold,
            hallucination_rate=rate,
            unsupported_claims=unsupported,
            supported_claims=supported,
            total_claims=total,
        )

    def _llm_check(self, answer: str, context: List[str]) -> HallucinationReport:
        """LLM-based check with claim extraction and verification."""
        # Step 1: Extract claims
        claims_prompt = f"""Extract all factual claims from this answer.
List each claim on a separate line prefixed with "- ".
Only include factual assertions, not opinions or hedged statements.

Answer: {answer}

Claims:"""
        claims_text = self.llm.generate(claims_prompt)
        claims = [
            c.strip().lstrip("- ").strip()
            for c in claims_text.strip().split("\n")
            if c.strip() and c.strip() != "-"
        ]

        if not claims:
            return HallucinationReport(
                has_hallucinations=False,
                hallucination_rate=0.0,
                unsupported_claims=[],
                supported_claims=[],
                total_claims=0,
            )

        # Step 2: Verify each claim
        context_text = "\n\n".join(context)
        supported = []
        unsupported = []

        for claim in claims:
            verify_prompt = f"""Is this claim supported by the context below?
Answer SUPPORTED or NOT_SUPPORTED.

Context:
{context_text[:3000]}

Claim: {claim}

Verdict:"""
            verdict = self.llm.generate(verify_prompt).strip().upper()
            if "SUPPORTED" in verdict and "NOT" not in verdict:
                supported.append(claim)
            else:
                unsupported.append(claim)

        total = len(claims)
        rate = len(unsupported) / total if total > 0 else 0.0

        return HallucinationReport(
            has_hallucinations=rate > self.threshold,
            hallucination_rate=rate,
            unsupported_claims=unsupported,
            supported_claims=supported,
            total_claims=total,
            metadata={"detection_method": "llm"},
        )
