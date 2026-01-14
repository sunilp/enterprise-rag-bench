# Multi-Tenant RAG Architecture

## The Challenge

Enterprise RAG systems often serve multiple business units, each with their own document corpus and access policies. Data must be isolated between tenants while sharing infrastructure for cost efficiency.

## Isolation Strategies

### Strategy 1: Separate Collections
Each tenant gets its own vector store collection/namespace. Complete data isolation. Simple to reason about. Higher infrastructure cost.

### Strategy 2: Shared Collection with Metadata Filtering
Single collection with a `tenant_id` metadata field. Filter by tenant on every query. Lower cost but requires discipline — a missing filter leaks data.

### Strategy 3: Shared Embeddings, Separate Stores
Share the embedding model across tenants but store vectors in tenant-specific databases. Balances cost and isolation.

## Recommendation

For regulated environments: Strategy 1 (separate collections). The cost premium is small compared to the risk of a cross-tenant data leak. Use Strategy 2 only for internal tools where all tenants have similar access levels.
