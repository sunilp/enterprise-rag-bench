"""Request lifecycle tracing for RAG pipelines.

Captures full request traces: query → retrieval → generation → response.
Traces are structured for export to OpenTelemetry, Datadog, or plain JSON.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Generator, List, Optional


@dataclass
class Span:
    """A single span in a trace."""
    name: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    def add_event(self, name: str, attributes: Optional[Dict] = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })


@dataclass
class Trace:
    """A complete request trace."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    spans: List[Span] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_duration_ms(self) -> float:
        if not self.spans:
            return 0.0
        start = min(s.start_time for s in self.spans)
        end = max(s.end_time for s in self.spans)
        return (end - start) * 1000

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class RequestTracer:
    """Trace RAG request lifecycles.

    Usage:
        tracer = RequestTracer()

        with tracer.trace("rag_query") as trace:
            with tracer.span(trace, "retrieval") as span:
                span.attributes["num_results"] = 5
                results = retriever.search(query)

            with tracer.span(trace, "generation") as span:
                span.attributes["model"] = "gpt-4"
                answer = llm.generate(prompt)

        # Export trace
        tracer.export(trace)
    """

    def __init__(self, export_fn: Optional[Any] = None):
        self.export_fn = export_fn
        self._traces: List[Trace] = []

    @contextmanager
    def trace(self, name: str, **metadata) -> Generator[Trace, None, None]:
        """Create a new trace context."""
        t = Trace(metadata={"name": name, **metadata})
        try:
            yield t
        finally:
            self._traces.append(t)
            if self.export_fn:
                self.export_fn(t)

    @contextmanager
    def span(
        self, trace: Trace, name: str, parent: Optional[Span] = None
    ) -> Generator[Span, None, None]:
        """Create a span within a trace."""
        s = Span(
            name=name,
            parent_id=parent.span_id if parent else None,
        )
        s.start_time = time.time()
        try:
            yield s
        finally:
            s.end_time = time.time()
            trace.spans.append(s)

    @property
    def traces(self) -> List[Trace]:
        return self._traces

    def export_all(self, path: str) -> None:
        """Export all traces to a JSON file."""
        data = [t.to_dict() for t in self._traces]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
