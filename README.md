# Enterprise RAG Bench

**RAG evaluation and benchmarking for enterprise environments.**

Most RAG tutorials stop at "it works in a notebook." This project provides the infrastructure to answer harder questions: which retrieval pattern works best for *your* documents? How do you measure faithfulness when the LLM hallucinates? What happens to retrieval quality when you change your chunking strategy?

---

## What This Is

A benchmarking framework for comparing RAG retrieval patterns on enterprise documents. Includes:

- **5 retrieval patterns** — naive RAG, hybrid search (BM25+vector), parent document, corrective RAG, agentic RAG
- **5 chunking strategies** — fixed-size, recursive character, sentence window, semantic, document structure
- **Evaluation harness** — automated metrics (faithfulness, relevance, groundedness, context precision)
- **LLM-as-judge** — configurable rubrics for quality assessment
- **Guardrails** — PII detection, prompt injection defense, hallucination detection
- **Observability** — request tracing, token usage, cost attribution

## Architecture

```
Query → [Injection Defense] → [Retrieval Pattern] → [Reranking] → [PII Filter]
                                                                        ↓
                                                                   [Generation]
                                                                        ↓
                                                              [Hallucination Check]
                                                                        ↓
                                                                    Response
                                                                        ↓
                                                               [Trace + Cost Log]
```

## Quick Start

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

To run benchmarks with your own documents:

```bash
# 1. Place your documents in benchmarks/financial_docs/
# 2. Create eval_cases.json with question-answer pairs
# 3. Configure providers in benchmarks/run_benchmarks.py
python benchmarks/run_benchmarks.py
```

## Retrieval Patterns

| Pattern | How It Works | Best For | Latency |
|---------|-------------|----------|---------|
| **Naive RAG** | Embed → retrieve → generate | Prototyping, simple Q&A | Low |
| **Hybrid Search** | BM25 + vector + RRF fusion | Domain jargon, acronyms | Low |
| **Parent Document** | Retrieve child chunks, return parent | Long docs, contracts | Medium |
| **Corrective RAG** | Verify retrieval quality, retry if poor | High-stakes Q&A | High |
| **Agentic RAG** | LLM routes retrieval strategy | Multi-hop, multi-source | Highest |

## Chunking Strategies

| Strategy | Approach | Trade-off |
|----------|----------|-----------|
| **Fixed Size** | Split by character count + overlap | Fast but breaks semantic boundaries |
| **Recursive Character** | Split by separators (paragraph → sentence → word) | Good balance of speed and quality |
| **Sentence Window** | Center sentence + N neighbors | Precise retrieval, rich context |
| **Semantic** | Split where embedding similarity drops | Best semantic coherence, requires embedding |
| **Document Structure** | Split by headers/sections | Preserves document hierarchy |

## Enterprise Concerns

The `enterprise-concerns/` directory covers patterns specific to regulated enterprise environments:

- **PII in retrieval** — handling customer data in document chunks
- **Access control** — document-level permissions in RAG
- **Audit trails** — tracing which documents informed which answer
- **Multi-tenancy** — data isolation across business units
- **Cost optimization** — token budgets, caching, reranker trade-offs

## Evaluation

```python
from src.evaluation import EvalRunner, EvalCase
from src.evaluation.metrics import faithfulness, relevance

# Create eval cases
cases = [EvalCase(question="What does BCBS 239 require?")]

# Run across patterns
runner = EvalRunner(patterns={"naive": naive_rag, "hybrid": hybrid_rag})
results = runner.run(cases)

# Summarize
summary = runner.summarize(results)
```

Metrics:
- **Faithfulness** — is the answer grounded in the context?
- **Relevance** — does the answer address the question?
- **Groundedness** — can each claim be traced to a source?
- **Context Precision** — are the retrieved documents relevant?

## Project Structure

```
enterprise-rag-bench/
├── src/
│   ├── indexing/          # Chunking and embedding strategies
│   ├── retrieval/         # 5 RAG retrieval patterns
│   ├── evaluation/        # Metrics, eval runner, LLM judge
│   ├── guardrails/        # PII, injection defense, hallucination detection
│   └── observability/     # Tracing and cost tracking
├── benchmarks/            # Benchmark runner and results
├── tests/                 # Unit tests
├── docs/                  # Pattern guides and comparisons
└── enterprise-concerns/   # Enterprise-specific patterns
```

## License

Apache 2.0
