"""Automated evaluation harness for RAG pipelines.

Runs a set of question-answer pairs through one or more RAG patterns
and computes evaluation metrics. Outputs structured results for
comparison and regression detection.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .metrics import faithfulness, relevance, groundedness, context_precision


@dataclass
class EvalCase:
    """A single evaluation test case."""
    question: str
    expected_answer: Optional[str] = None
    expected_contexts: Optional[List[str]] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of evaluating a single case."""
    question: str
    answer: str
    faithfulness_score: float
    relevance_score: float
    groundedness_score: float
    context_precision_score: float
    latency_ms: float
    num_retrieved: int
    pattern: str
    metadata: dict = field(default_factory=dict)

    @property
    def composite_score(self) -> float:
        """Weighted composite of all metrics."""
        return (
            0.3 * self.faithfulness_score
            + 0.3 * self.relevance_score
            + 0.2 * self.groundedness_score
            + 0.2 * self.context_precision_score
        )


class EvalRunner:
    """Run evaluation across RAG patterns and test cases.

    Usage:
        runner = EvalRunner(patterns={"naive": naive_rag, "hybrid": hybrid_rag})
        results = runner.run(eval_cases, judge_llm=judge)
        runner.save_results(results, "benchmark_results.json")
    """

    def __init__(self, patterns: Dict[str, Any]):
        self.patterns = patterns

    def run(
        self,
        cases: List[EvalCase],
        judge_llm: Optional[Any] = None,
    ) -> Dict[str, List[EvalResult]]:
        """Run all cases through all patterns."""
        results: Dict[str, List[EvalResult]] = {}

        for pattern_name, pattern in self.patterns.items():
            pattern_results = []

            for case in cases:
                start = time.time()
                response = pattern.query(case.question)
                elapsed_ms = (time.time() - start) * 1000

                source_texts = [
                    d.get("text", d.get("page_content", ""))
                    for d in response.source_documents
                ]

                faith = faithfulness(response.answer, source_texts, llm=judge_llm)
                rel = relevance(response.answer, case.question, llm=judge_llm)
                ground = groundedness(response.answer, source_texts, llm=judge_llm)
                ctx_prec = context_precision(case.question, source_texts, llm=judge_llm)

                result = EvalResult(
                    question=case.question,
                    answer=response.answer,
                    faithfulness_score=faith,
                    relevance_score=rel,
                    groundedness_score=ground,
                    context_precision_score=ctx_prec,
                    latency_ms=elapsed_ms,
                    num_retrieved=len(response.source_documents),
                    pattern=pattern_name,
                )
                pattern_results.append(result)

            results[pattern_name] = pattern_results

        return results

    @staticmethod
    def summarize(results: Dict[str, List[EvalResult]]) -> Dict[str, Dict[str, float]]:
        """Compute aggregate metrics per pattern."""
        summary = {}
        for pattern_name, pattern_results in results.items():
            n = len(pattern_results)
            if n == 0:
                continue
            summary[pattern_name] = {
                "avg_faithfulness": sum(r.faithfulness_score for r in pattern_results) / n,
                "avg_relevance": sum(r.relevance_score for r in pattern_results) / n,
                "avg_groundedness": sum(r.groundedness_score for r in pattern_results) / n,
                "avg_context_precision": sum(r.context_precision_score for r in pattern_results) / n,
                "avg_composite": sum(r.composite_score for r in pattern_results) / n,
                "avg_latency_ms": sum(r.latency_ms for r in pattern_results) / n,
                "p95_latency_ms": sorted(r.latency_ms for r in pattern_results)[int(n * 0.95)] if n > 1 else pattern_results[0].latency_ms,
                "num_cases": n,
            }
        return summary

    @staticmethod
    def save_results(results: Dict[str, List[EvalResult]], path: str) -> None:
        """Save results to JSON."""
        data = {
            pattern: [asdict(r) for r in rs]
            for pattern, rs in results.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
