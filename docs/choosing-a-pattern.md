# Choosing a RAG Retrieval Pattern

## Decision Framework

Start with the simplest pattern that meets your requirements. Upgrade when you have evidence (from evaluation) that a more complex pattern improves quality enough to justify the added latency and cost.

### Start Here: What's Your Primary Constraint?

**"I need low latency and simple ops"** → Naive RAG or Hybrid Search

**"My documents have domain-specific terminology"** → Hybrid Search (BM25 catches exact terms that embeddings miss)

**"I need high retrieval precision on long documents"** → Parent Document Retriever

**"Wrong answers have real consequences"** → Corrective RAG

**"Questions require information from multiple sources"** → Agentic RAG

### Pattern Selection Matrix

| Criterion | Naive | Hybrid | Parent Doc | Corrective | Agentic |
|-----------|-------|--------|------------|------------|---------|
| Implementation complexity | Low | Medium | Medium | High | Highest |
| Latency | ~1s | ~1.5s | ~1.5s | ~3-5s | ~5-15s |
| Cost per query | $ | $ | $ | $$$ | $$$$ |
| Handles jargon/acronyms | Poor | Good | Fair | Fair | Good |
| Long document quality | Fair | Fair | Good | Good | Good |
| Multi-hop reasoning | Poor | Poor | Fair | Fair | Good |
| Verifiability | Low | Low | Medium | High | Medium |

### When NOT to Use RAG

- The LLM already knows the answer (public knowledge, well-documented)
- The answer requires real-time data (use API calls instead)
- The answer requires computation or aggregation (use code execution)
- Your document corpus is very small (fine-tuning may be simpler)
