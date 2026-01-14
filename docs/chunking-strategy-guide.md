# Chunking Strategy Guide

## Why Chunking Matters

Chunking is the most underrated component of a RAG pipeline. The wrong chunking strategy degrades retrieval quality more than the wrong embedding model. Yet most implementations use a fixed 512-character split with 50-character overlap and never revisit the decision.

## Strategy Comparison

### Fixed-Size Chunking
Split every N characters with M overlap.

**Pros:** Simple, predictable chunk count, fast.
**Cons:** Breaks sentences and paragraphs mid-thought. Creates chunks with mixed topics.
**Use when:** Document structure is uniform, you need predictable performance, or you're prototyping.

### Recursive Character Chunking
Try the most meaningful separator first (paragraph break), fall back to smaller separators (line break, sentence, word) when chunks exceed the size limit.

**Pros:** Preserves semantic boundaries in most cases. Good default.
**Cons:** Can produce very small chunks for short paragraphs. Separator hierarchy may not match all document types.
**Use when:** You want a reasonable default without embedding-based chunking.

### Sentence Window Chunking
Each chunk centers on a single sentence but includes N surrounding sentences as context.

**Pros:** Sentence-level retrieval precision with paragraph-level context for generation.
**Cons:** Creates N chunks where N = number of sentences (high chunk count). Significant overlap.
**Use when:** You need precise retrieval but rich generation context. Good for Q&A on regulatory documents.

### Semantic Chunking
Compute embeddings for each sentence, split where cosine similarity between consecutive sentences drops below a threshold.

**Pros:** Chunks align with actual topic boundaries. Best semantic coherence.
**Cons:** Requires an embedding model at indexing time. Threshold tuning is document-dependent.
**Use when:** Documents cover multiple topics per section and you need clean topic separation.

### Document Structure Chunking
Split at document structure markers (headers, sections). Each chunk corresponds to a logical section.

**Pros:** Preserves document hierarchy. Section titles provide useful metadata for filtering.
**Cons:** Section sizes vary wildly — some too small, some too large. Requires structured documents.
**Use when:** Documents have consistent heading structure (markdown, HTML, well-formatted PDFs).

## Chunk Size Guidelines

| Document Type | Recommended Strategy | Chunk Size |
|--------------|---------------------|------------|
| Regulatory text | Sentence window (w=2) | ~200-400 chars |
| Contracts | Document structure + recursive fallback | ~500-1000 chars |
| Technical manuals | Document structure | ~800-1200 chars |
| News articles | Recursive character | ~300-500 chars |
| Financial reports | Semantic | ~400-800 chars |

## Measuring Chunk Quality

Run your evaluation suite across different strategies and measure:
1. **Context precision** — are retrieved chunks relevant?
2. **Answer faithfulness** — is the answer grounded in chunks?
3. **Retrieval latency** — how does chunk count affect search speed?
4. **Token cost** — larger chunks = more tokens per query
