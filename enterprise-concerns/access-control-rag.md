# Access Control in RAG Systems

## Document-Level Permissions

In enterprise RAG, different users should see different documents. A compliance officer may access regulatory filings; a customer service agent should only see their assigned accounts.

## Implementation Approaches

### Metadata Filtering
Tag each document chunk with access metadata (department, classification level, owner). At query time, apply metadata filters before or after vector search.

```python
results = vector_store.similarity_search(
    query,
    filter={"department": user.department, "classification": {"$lte": user.clearance}},
    k=10,
)
```

### Namespace Isolation
Use vector store namespaces or collections per access group. Each group has its own index.

### Row-Level Security (pgvector)
If using pgvector, leverage PostgreSQL's row-level security policies to enforce access at the database level.

## Common Mistakes

1. **Filtering after retrieval** — retrieving all documents then filtering client-side leaks data through timing attacks and potentially through LLM context
2. **Relying on LLM instructions** — telling the LLM "don't reveal confidential information" is not a security control
3. **Shared embedding models** — if the embedding model is fine-tuned on restricted data, it may encode restricted information in embeddings of unrestricted queries
