# PII in RAG Retrieval

## The Problem

Enterprise documents often contain personally identifiable information: customer names, account numbers, addresses, SSNs. When these documents are chunked and indexed for RAG, PII gets embedded in vector stores and potentially surfaced in LLM responses.

## Mitigation Patterns

### Pattern 1: Pre-indexing Redaction
Detect and redact PII before chunking and embedding. The redacted text is indexed; original documents are stored separately with access controls.

**Trade-off:** Redaction can remove context that's important for retrieval quality. "John Smith's account was flagged" becomes "[NAME]'s account was flagged" — losing the ability to search by customer name.

### Pattern 2: Post-retrieval Filtering
Index documents with PII intact, but filter the retrieved chunks before passing them to the LLM. Use the `PIIFilter` class in `src/guardrails/pii_filter.py`.

**Trade-off:** PII exists in the vector store. Acceptable if the store has appropriate access controls, but may not satisfy certain regulatory requirements.

### Pattern 3: Dual-index Architecture
Maintain two indexes: one with PII (restricted access) and one without (general access). Route queries to the appropriate index based on the user's authorization level.

**Trade-off:** Doubles indexing cost and operational complexity. But provides clean separation between authorized and unauthorized access.

## Recommendation

For banking: Pattern 3 (dual-index) for customer-facing applications. Pattern 2 for internal analyst tools with appropriate access controls. Pattern 1 is too lossy for most enterprise use cases.
