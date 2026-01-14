# Cost Optimization for Enterprise RAG

## Cost Components

1. **Embedding** — encoding queries and documents into vectors
2. **Vector search** — infrastructure cost of the vector store
3. **LLM generation** — the most expensive per-query component
4. **Reranking** — cross-encoder or API-based reranking
5. **Evaluation** — LLM-as-judge for quality monitoring

## Optimization Strategies

### Reduce LLM Token Usage
- **Compress retrieved context** — summarize or truncate chunks before passing to the LLM
- **Limit top-k** — retrieve fewer documents (measure quality impact first)
- **Use cheaper models for simple queries** — route simple questions to smaller models

### Cache Aggressively
- **Embedding cache** — cache embeddings for frequent queries
- **Response cache** — cache full responses for repeated questions (with TTL)
- **Semantic cache** — cache responses for semantically similar queries

### Right-size Your Models
- **Embeddings:** `text-embedding-3-small` is ~10x cheaper than `text-embedding-3-large` with modest quality loss
- **Generation:** `gpt-4o-mini` handles most enterprise Q&A at 1/20th the cost of `gpt-4o`
- **Reranking:** skip reranking for low-stakes queries; apply only when top-k > 10

### Monitor with CostTracker
Use `src/observability/cost_tracker.py` to attribute costs per query and identify optimization opportunities.

## Budgeting Rule of Thumb

For a typical enterprise RAG application:
- Embedding: ~5% of total cost
- Vector store: ~10% of total cost
- LLM generation: ~70% of total cost
- Reranking: ~10% of total cost
- Monitoring/eval: ~5% of total cost

Focus optimization efforts on generation (model selection, context compression, caching).
