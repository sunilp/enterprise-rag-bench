"""LLM-as-judge for RAG evaluation.

Uses a configurable LLM to evaluate answer quality with
custom rubrics. Supports both reference-based and reference-free
evaluation modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import re


class JudgeMode(Enum):
    REFERENCE_BASED = "reference_based"
    REFERENCE_FREE = "reference_free"


@dataclass
class JudgeRubric:
    """Evaluation rubric for the LLM judge."""
    name: str
    criteria: str
    score_range: tuple = (1, 5)
    weight: float = 1.0


DEFAULT_RUBRICS = [
    JudgeRubric(
        name="correctness",
        criteria="Is the answer factually correct and accurate?",
        weight=0.3,
    ),
    JudgeRubric(
        name="completeness",
        criteria="Does the answer fully address all parts of the question?",
        weight=0.25,
    ),
    JudgeRubric(
        name="relevance",
        criteria="Is the answer focused on the question without irrelevant information?",
        weight=0.25,
    ),
    JudgeRubric(
        name="coherence",
        criteria="Is the answer well-structured and easy to understand?",
        weight=0.2,
    ),
]


@dataclass
class JudgeVerdict:
    """Structured verdict from the LLM judge."""
    scores: Dict[str, float]
    weighted_score: float
    explanation: str
    metadata: dict


class LLMJudge:
    """LLM-as-judge evaluator with configurable rubrics.

    Usage:
        judge = LLMJudge(llm=my_llm, rubrics=DEFAULT_RUBRICS)
        verdict = judge.evaluate(question="...", answer="...", context=["..."])
    """

    EVAL_PROMPT = """You are an expert evaluator for question-answering systems.

Evaluate the following answer based on these criteria:
{rubric_text}

Question: {question}
{context_section}
Answer: {answer}

Score each criterion from {min_score} to {max_score}.
Respond in this exact format:
{score_format}
EXPLANATION: [one paragraph explaining your scores]"""

    def __init__(
        self,
        llm: Any,
        rubrics: Optional[List[JudgeRubric]] = None,
        mode: JudgeMode = JudgeMode.REFERENCE_FREE,
    ):
        self.llm = llm
        self.rubrics = rubrics or DEFAULT_RUBRICS
        self.mode = mode

    def evaluate(
        self,
        question: str,
        answer: str,
        context: Optional[List[str]] = None,
        reference: Optional[str] = None,
    ) -> JudgeVerdict:
        """Evaluate an answer using LLM-as-judge."""
        rubric_text = "\n".join(
            f"- {r.name}: {r.criteria}" for r in self.rubrics
        )

        min_score = min(r.score_range[0] for r in self.rubrics)
        max_score = max(r.score_range[1] for r in self.rubrics)

        score_format = "\n".join(f"{r.name.upper()}: [score]" for r in self.rubrics)

        context_section = ""
        if context:
            context_section = f"Context:\n{chr(10).join(context[:3])}\n"
        if reference and self.mode == JudgeMode.REFERENCE_BASED:
            context_section += f"\nReference Answer: {reference}\n"

        prompt = self.EVAL_PROMPT.format(
            rubric_text=rubric_text,
            question=question,
            context_section=context_section,
            answer=answer,
            min_score=min_score,
            max_score=max_score,
            score_format=score_format,
        )

        response = self.llm.generate(prompt)
        return self._parse_verdict(response, max_score)

    def _parse_verdict(self, response: str, max_score: int) -> JudgeVerdict:
        """Parse the LLM's structured response into a verdict."""
        scores: Dict[str, float] = {}
        explanation = ""

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("EXPLANATION:"):
                explanation = line.split(":", 1)[1].strip()
                continue

            for rubric in self.rubrics:
                prefix = f"{rubric.name.upper()}:"
                if line.startswith(prefix):
                    try:
                        score_str = line.split(":", 1)[1].strip()
                        score = float(re.search(r"[\d.]+", score_str).group())
                        scores[rubric.name] = score
                    except (AttributeError, ValueError):
                        scores[rubric.name] = max_score / 2

        # Compute weighted score (normalized to 0-1)
        total_weight = sum(r.weight for r in self.rubrics)
        weighted = sum(
            scores.get(r.name, max_score / 2) / max_score * r.weight
            for r in self.rubrics
        )
        weighted_score = weighted / total_weight if total_weight > 0 else 0.5

        return JudgeVerdict(
            scores=scores,
            weighted_score=weighted_score,
            explanation=explanation,
            metadata={"mode": self.mode.value},
        )
