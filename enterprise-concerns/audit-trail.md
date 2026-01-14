# Audit Trails for RAG Systems

## What Regulators Expect

In regulated industries, you must be able to answer: "What information did the system use to produce this output?" For RAG, this means logging:

1. The original user query
2. Which documents were retrieved (with scores)
3. What context was passed to the LLM
4. The full LLM prompt (including system prompt)
5. The LLM response
6. Any post-processing or filtering applied
7. The final response delivered to the user

## Implementation

Use the `RequestTracer` from `src/observability/tracer.py`:

```python
tracer = RequestTracer()

with tracer.trace("rag_query", user_id=user.id) as trace:
    with tracer.span(trace, "retrieval") as span:
        span.attributes["query"] = question
        span.attributes["top_k"] = 5
        results = retriever.search(question)
        span.attributes["num_results"] = len(results)
        span.attributes["doc_ids"] = [r.id for r in results]

    with tracer.span(trace, "generation") as span:
        span.attributes["model"] = "gpt-4o"
        span.attributes["prompt_tokens"] = count_tokens(prompt)
        answer = llm.generate(prompt)
        span.attributes["completion_tokens"] = count_tokens(answer)
```

## Retention

- Minimum 7 years for financial services (varies by jurisdiction)
- Store traces in append-only, tamper-evident storage
- Include trace IDs in user-facing responses for cross-referencing
