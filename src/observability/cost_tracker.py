"""Token usage and cost attribution for RAG pipelines.

Tracks token consumption across embedding, retrieval, and generation
steps. Essential for enterprise cost management when operating
multiple RAG applications at scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Cost per 1K tokens (USD) — approximate, update as pricing changes
MODEL_PRICING = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-20250414": {"input": 0.0008, "output": 0.004},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
    "text-embedding-004": {"input": 0.000025, "output": 0.0},
}


@dataclass
class TokenUsage:
    """Token usage for a single operation."""
    operation: str
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        pricing = MODEL_PRICING.get(self.model, {"input": 0.001, "output": 0.002})
        return (
            self.input_tokens / 1000 * pricing["input"]
            + self.output_tokens / 1000 * pricing["output"]
        )


@dataclass
class QueryCostReport:
    """Cost report for a single RAG query."""
    query_id: str
    usages: List[TokenUsage] = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.usages)

    @property
    def total_output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.usages)

    @property
    def total_cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.usages)

    @property
    def cost_by_operation(self) -> Dict[str, float]:
        costs: Dict[str, float] = {}
        for u in self.usages:
            costs[u.operation] = costs.get(u.operation, 0.0) + u.cost_usd
        return costs


class CostTracker:
    """Track and attribute costs across RAG operations.

    Usage:
        tracker = CostTracker()
        tracker.record("embedding", "text-embedding-3-small", input_tokens=150)
        tracker.record("generation", "gpt-4o", input_tokens=2000, output_tokens=500)
        report = tracker.report()
        print(f"Total cost: ${report.total_cost_usd:.4f}")
    """

    def __init__(self, query_id: Optional[str] = None):
        self._report = QueryCostReport(query_id=query_id or "default")

    def record(
        self,
        operation: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> TokenUsage:
        usage = TokenUsage(
            operation=operation,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self._report.usages.append(usage)
        return usage

    def report(self) -> QueryCostReport:
        return self._report

    def estimate_monthly_cost(self, queries_per_day: int) -> float:
        """Estimate monthly cost based on average query cost."""
        if not self._report.usages:
            return 0.0
        avg_cost = self._report.total_cost_usd
        return avg_cost * queries_per_day * 30

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate. Use tiktoken for precise counts."""
        return int(len(text.split()) * 1.3)
