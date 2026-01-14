# Vector Database Comparison for Enterprise RAG

## Selection Criteria for Regulated Environments

Choosing a vector database for enterprise RAG in regulated industries involves constraints that most comparison guides ignore:

1. **Data residency** — where is the data stored and processed?
2. **Encryption** — at rest, in transit, and ideally client-side
3. **Access control** — row/document-level permissions
4. **Audit logging** — who queried what, when
5. **Availability SLAs** — guaranteed uptime for production workloads
6. **Backup and recovery** — point-in-time restoration

## Options

### pgvector (PostgreSQL Extension)
**Best for:** Teams already on PostgreSQL who need vector search without a new system.

- Data residency: wherever you host Postgres (full control)
- Encryption: PostgreSQL TDE + standard TLS
- Access control: PostgreSQL row-level security
- Scale: good to ~10M vectors, degrades beyond
- Managed options: Cloud SQL, AlloyDB, RDS

### ChromaDB
**Best for:** Prototyping and small-scale applications.

- Local-first, runs embedded in Python
- No built-in auth or encryption (not suitable for production regulated environments)
- Good developer experience
- Use for benchmarking and development, not production

### Vertex AI Vector Search (Google Cloud)
**Best for:** GCP-native teams needing managed, scalable vector search.

- Data residency: GCP regions
- Encryption: CMEK supported
- Access control: IAM integration
- Scale: billions of vectors
- Fully managed, no operational overhead

### Weaviate
**Best for:** Teams needing hybrid (vector + keyword) search in a managed service.

- Self-hosted or managed cloud
- Built-in hybrid search (BM25 + vector)
- Multi-tenancy support
- RBAC and API key management
- Good for multi-tenant RAG applications

### Pinecone
**Best for:** Teams wanting fully managed vector search with minimal ops.

- Managed cloud only (no self-hosted option)
- SOC 2 Type II certified
- Namespace-based isolation
- Limited data residency options
- Simplest operational model

## Recommendation for Banking

For regulated financial services:
1. **pgvector on Cloud SQL/AlloyDB** if you need maximum control and are already on GCP
2. **Vertex AI Vector Search** if you need scale and are committed to GCP
3. **Weaviate self-hosted** if you need hybrid search with full data control

Avoid solutions without clear data residency guarantees, encryption at rest, and audit logging.
