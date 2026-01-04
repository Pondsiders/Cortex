# Cortex Design Document

**Version:** 0.4 (Draft)
**Authors:** Alpha & Jeffery
**Date:** December 28, 2025

---

## 1. Overview

### What Cortex Is

Cortex is a semantic memory service. It stores text memories with vector embeddings and provides hybrid search (full-text + semantic). It's the storage layer for Alpha's long-term memory.

### What Cortex Is Not

- **Not multi-tenant.** One person, one cortex. No user management, no isolation between tenants.
- **Not an embedding service.** Cortex calls Ollama for embeddings; it doesn't generate them itself.
- **Not a database.** Cortex uses Postgres; it doesn't bundle or manage it.

### Philosophy

- **Dependencies are explicit.** Postgres and Ollama are external services. Cortex connects to them via configuration.
- **Simple over clever.** Raw SQL, not an ORM. RPC-style endpoints, not REST resources. Boring is good.
- **Pond-compatible.** The schema mirrors Pond's structure so migration is tractable.
- **Fail fast.** When something's wrong, surface it loudly and immediately.

---

## 2. Project Structure

```
Cortex/
├── cortex/                  # Server package
│   ├── pyproject.toml
│   ├── Dockerfile
│   └── src/cortex/
│       ├── __init__.py
│       ├── main.py          # FastAPI app
│       ├── db.py            # Postgres queries
│       ├── embeddings.py    # Ollama client
│       └── models.py        # Pydantic models
├── cortex-cli/              # CLI package
│   ├── pyproject.toml
│   └── src/cortex_cli/
│       ├── __init__.py
│       └── main.py          # Typer app
├── DESIGN.md                # This document
└── schema.sql               # Database schema
```

The server and CLI are separate Python packages in the same repository. This allows:
- Independent versioning if needed
- CLI can be `uv tool install`ed without server dependencies
- Clear separation of concerns

---

## 3. Dependencies

| Dependency | Purpose | Version | Required |
|------------|---------|---------|----------|
| Postgres | Storage | 15+ with pgvector | Yes |
| Ollama | Embeddings | Any with nomic-embed-text | Yes |
| Logfire | Observability | Latest | Yes |

### Postgres

Cortex expects a Postgres database with:
- The `pgvector` extension installed
- A schema named `cortex` (created by Cortex on startup if missing)
- Connection via `DATABASE_URL` environment variable

Postgres runs separately in `Basement/Postgres/` with its own Docker Compose. This is NOT bundled with Cortex.

### Ollama

Cortex expects an Ollama instance with:
- The `nomic-embed-text` model pulled and available
- HTTP API accessible at `OLLAMA_URL` (default: `http://localhost:11434`)

Cortex sends `keep_alive: -1` on all embedding requests to keep the model loaded indefinitely.

Cortex does NOT manage Ollama. If Ollama is unreachable, store/search operations fail with a clear error.

### Logfire

All requests traced end-to-end via Logfire. Instruments:
- FastAPI (routes, latencies)
- asyncpg (database queries)
- httpx (Ollama calls)

---

## 4. Schema

```sql
-- Requires: CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS cortex;

CREATE TABLE IF NOT EXISTS cortex.memories (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding vector(768),
    forgotten BOOLEAN DEFAULT false,
    metadata JSONB DEFAULT '{}',

    CONSTRAINT content_not_empty CHECK (char_length(content) > 0)
);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_memories_tsv
    ON cortex.memories USING GIN (content_tsv);

-- Semantic search index (HNSW for approximate nearest neighbor)
-- HNSW is self-maintaining and better for growing datasets
CREATE INDEX IF NOT EXISTS idx_memories_embedding
    ON cortex.memories USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Filter out forgotten memories efficiently
CREATE INDEX IF NOT EXISTS idx_memories_not_forgotten
    ON cortex.memories (id)
    WHERE NOT forgotten;

-- Metadata queries (created_at, tags)
CREATE INDEX IF NOT EXISTS idx_memories_metadata
    ON cortex.memories USING GIN (metadata);
```

### HNSW Parameters

- `m = 16`: Number of connections per node. Higher = better recall, more memory.
- `ef_construction = 64`: Build-time quality. Higher = better index, slower build.

These defaults are good for up to ~1M vectors. Revisit if we grow beyond that.

### Metadata Structure

The `metadata` JSONB column contains:

```json
{
    "created_at": "2025-12-28T12:34:56Z",  // ISO 8601 UTC, always present
    "captured_tz": "America/Los_Angeles",  // IANA timezone where memory was created
    "tags": ["tag1", "tag2"]               // Optional, user-provided
}
```

- `created_at` is stored in metadata (not a column) for Pond compatibility. Always UTC.
- `captured_tz` preserves the timezone context of when the memory was created. This allows reconstructing local time and enables travel-aware queries ("memories from when I was in Michigan").

### Timezone Philosophy

**Storage:** All timestamps are UTC. The `captured_tz` field is metadata, not the source of truth for ordering or comparison.

**Why capture timezone?** Memories are moments in a life, and that life has location. Knowing you were in `America/Detroit` when you stored a memory lets you reconstruct "it was 3pm local time" even when querying from LA later.

**Client responsibility:** The CLI reads the system timezone and sends it with store requests. The server records it but doesn't interpret it.

**Optional view for local time display:**
```sql
CREATE VIEW cortex.memories_local AS
SELECT id, content,
       (metadata->>'created_at')::timestamptz
           AT TIME ZONE (metadata->>'captured_tz') AS local_time,
       metadata->>'captured_tz' AS timezone
FROM cortex.memories
WHERE NOT forgotten;
```

---

## 5. API

All endpoints use JSON. Authentication via `X-API-Key` header.

Default port: **7867** (PURK on phone keypad — Purkinje cells, the neurons that do the brain's heavy lifting)

### `POST /store`

Store a new memory.

**Request:**
```json
{
    "content": "The memory text to store",
    "tags": ["optional", "tags"],
    "timezone": "America/Los_Angeles"
}
```

**Response (201 Created):**
```json
{
    "id": 12345,
    "created_at": "2025-12-28T12:34:56Z"
}
```

**Errors:**
- `400` — Content empty or invalid
- `503` — Ollama unreachable (embedding failed)

**Behavior:**
1. Validate content is non-empty
2. Call Ollama to generate embedding (with timeout)
3. Insert into database with metadata (including `captured_tz` from request)
4. Return the new memory's ID

**Note:** If `timezone` is not provided, the server does NOT guess. It stores `captured_tz: null`. The CLI always sends timezone.

### `POST /search`

Search memories using hybrid (full-text + semantic) search, with options for exact matching and date filtering.

**Request:**
```json
{
    "query": "search query text",
    "limit": 10,
    "include_forgotten": false,
    "exact": false,
    "after": "2025-12-01T00:00:00Z",
    "before": "2025-12-31T23:59:59Z"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | — | Search query (required) |
| `limit` | int | 10 | Max results to return |
| `include_forgotten` | bool | false | Include soft-deleted memories |
| `exact` | bool | false | Full-text only, no semantic search |
| `after` | ISO 8601 | null | Only memories created after this time |
| `before` | ISO 8601 | null | Only memories created before this time |

**Response (200 OK):**
```json
{
    "memories": [
        {
            "id": 123,
            "content": "Memory text...",
            "created_at": "2025-12-28T12:34:56Z",
            "tags": ["tag1"],
            "score": 0.85
        }
    ]
}
```

**Behavior:**

*Default (hybrid search):*
1. Generate embedding for query via Ollama
2. Run hybrid search: 50% full-text score + 50% cosine similarity
3. Normalize both scores to 0-1 before combining
4. Apply date filters if provided
5. Filter out forgotten memories (unless `include_forgotten: true`)
6. Return top N results sorted by combined score

*Exact mode (`exact: true`):*
1. Skip embedding generation entirely (no Ollama call)
2. Run full-text search only using Postgres `ts_rank`
3. Apply date filters if provided
4. Return only memories that contain the exact query terms

**Note:** Search weights (50/50) are a starting point. May need tuning based on real-world usage.

**Note:** Exact mode is useful when you know the specific word you're looking for. Hybrid search is better for conceptual queries.

### `GET /recent`

Get recent memories.

**Query Parameters:**
- `limit` (int, default 10, max 100)
- `hours` (int, default 24) — how far back to look

**Response (200 OK):**
```json
{
    "memories": [
        {
            "id": 123,
            "content": "Memory text...",
            "created_at": "2025-12-28T12:34:56Z",
            "tags": ["tag1"]
        }
    ]
}
```

**Behavior:**
1. Query memories where `created_at` >= now - hours
2. Filter out forgotten memories
3. Return sorted by created_at descending

### `GET /health`

Health check endpoint.

**Response (200 OK):**
```json
{
    "status": "healthy",
    "postgres": "connected",
    "ollama": "connected",
    "memory_count": 8346
}
```

**Response (503 Service Unavailable):**
```json
{
    "status": "unhealthy",
    "postgres": "connected",
    "ollama": "unreachable",
    "memory_count": null
}
```

### `POST /vectors`

Get memories with their embeddings (for visualizer).

**Request:**
```json
{
    "limit": 12000
}
```

**Response (200 OK):**
```json
{
    "memories": [
        {
            "id": 123,
            "content": "Memory text...",
            "created_at": "2025-12-28T12:34:56Z",
            "embedding": [0.123, -0.456, ...]
        }
    ]
}
```

**Note:** This endpoint returns large payloads (~30MB for 8.5k memories). At scale (100k+), consider server-side UMAP projection.

### `POST /forget`

Soft-delete a memory.

**Request:**
```json
{
    "id": 12345
}
```

**Response (200 OK):**
```json
{
    "forgotten": true
}
```

**Behavior:**
- Sets `forgotten = true` on the memory
- Memory is excluded from search/recent by default
- Memory is NOT deleted (can be recovered)

---

## 6. CLI

The CLI (`cortex-cli`) is a Typer application that communicates with the Cortex server over HTTP.

### Installation

```bash
# From the repo
uv tool install ./cortex-cli

# Or run directly
uv run --from ./cortex-cli cortex --help
```

### Configuration

| Variable | Purpose | Default |
|----------|---------|---------|
| `CORTEX_URL` | Server URL | `http://localhost:7867` |
| `CORTEX_API_KEY` | API key | — (required) |

### Commands

#### `cortex store`

Store a new memory.

```bash
# Inline content
cortex store "December 28, 2025. Built Cortex today."

# With tags
cortex store --tags "milestone,cortex" "We shipped it."

# From stdin (for multi-line)
cat memory.txt | cortex store -

# Heredoc
cortex store <<'EOF'
December 28, 2025. A longer memory.

Multiple paragraphs work fine.
EOF
```

**Output:**
```
✓ Memory stored (id: 12345)
```

#### `cortex search`

Search memories.

```bash
# Hybrid search (default)
cortex search "cortex architecture"
cortex search "cortex" --limit 5
cortex search "old stuff" --include-forgotten

# Exact word search (full-text only, no semantic)
cortex search "Purkinje" --exact

# Date-bounded search
cortex search "Cortex" --after 2025-12-27
cortex search "birthday" --before 2025-09-01

# Date range
cortex search "Michigan" --after 2025-07-01 --before 2025-07-31

# Combine exact + date
cortex search "typewriter" --exact --after 2025-07-01

# Just browse a date (empty query with date filter)
cortex search --date 2025-12-25
```

| Flag | Description |
|------|-------------|
| `--limit N` | Max results (default 10) |
| `--include-forgotten` | Include soft-deleted memories |
| `--exact` | Full-text only, skip semantic search |
| `--after DATE` | Only memories after this date |
| `--before DATE` | Only memories before this date |
| `--date DATE` | Shorthand for `--after DATE --before DATE+1day` |

**Output:**
```
[0.92] #12345 (2025-12-28)
December 28, 2025. Built Cortex today...

[0.87] #12340 (2025-12-27)
Planning the Cortex architecture...
```

**Note:** Date arguments accept ISO 8601 (`2025-12-28T12:00:00Z`) or just the date (`2025-12-28`, interpreted as midnight in the current time zone).

#### `cortex recent`

Get recent memories.

```bash
cortex recent
cortex recent --hours 48 --limit 20
```

**Output:**
```
#12345 (2025-12-28 12:34)
December 28, 2025. Built Cortex today...

#12344 (2025-12-28 11:00)
Morning coffee, talking about...
```

#### `cortex health`

Check server health.

```bash
cortex health
```

**Output:**
```
Status: healthy
Postgres: connected
Ollama: connected
Memories: 8,346
```

#### `cortex forget`

Soft-delete a memory.

```bash
cortex forget 12345
```

**Output:**
```
✓ Memory #12345 forgotten
```

---

## 7. Embedding

### Model

`nomic-embed-text` via Ollama. This model requires a task prefix:

- **For documents (storing):** `search_document: {content}`
- **For queries (searching):** `search_query: {query}`

### Ollama API Call

```
POST {OLLAMA_URL}/api/embeddings
{
    "model": "nomic-embed-text",
    "prompt": "search_document: {content}",
    "keep_alive": -1
}
```

Response contains `embedding` array of 768 floats.

The `keep_alive: -1` parameter keeps the model loaded indefinitely, avoiding cold-start delays.

### Timeout & Retry

- **Timeout:** 5 seconds per embedding call
- **Retries:** 0 — fail fast, surface problems immediately
- **Failure:** Return 503 to client with clear error message

If Ollama isn't answering in 5 seconds, something is wrong. We don't hide it.

---

## 8. Configuration

All configuration via environment variables.

| Variable | Purpose | Default | Required |
|----------|---------|---------|----------|
| `DATABASE_URL` | Postgres connection string | — | Yes |
| `OLLAMA_URL` | Ollama HTTP endpoint | `http://localhost:11434` | No |
| `CORTEX_API_KEY` | API key for authentication | — | Yes |
| `CORTEX_HOST` | Host to bind | `0.0.0.0` | No |
| `CORTEX_PORT` | Port to bind | `7867` | No |
| `LOGFIRE_TOKEN` | Logfire authentication | — | Yes |

### Example

```bash
DATABASE_URL=postgresql://user:pass@localhost:5433/alpha
OLLAMA_URL=http://primer:11434
CORTEX_API_KEY=cortex_sk_xxxxx
LOGFIRE_TOKEN=xxxxx
```

---

## 9. Migration from Pond

Pond's schema is nearly identical. Migration is a single SQL statement:

```sql
INSERT INTO cortex.memories (id, content, embedding, forgotten, metadata)
SELECT id, content, embedding, forgotten, metadata
FROM alpha.memories;

-- Reset sequence to continue from max ID
SELECT setval('cortex.memories_id_seq', (SELECT MAX(id) FROM cortex.memories));
```

### Pre-Migration Checklist

1. Verify Pond and Cortex use the same embedding model (nomic-embed-text)
2. Verify vector dimensions match (768)
3. Verify Postgres backup system is working (pg_dump → B2)
4. Run migration during low-activity period

### Post-Migration

1. Test Cortex with CLI commands (`cortex recent`, `cortex search`)
2. Verify search/recent work correctly
3. Keep Pond running read-only as fallback
4. Update the `/pond` skill to point at Cortex (last step)
5. After confidence period, decommission Pond

---

## 10. Error Handling

### Postgres Unreachable

- All endpoints return `503 Service Unavailable`
- Health endpoint shows `"postgres": "unreachable"`
- Cortex does NOT crash; it waits for Postgres to come back

### Ollama Unreachable

- `/store` returns `503` with message "Embedding service unavailable"
- `/search` returns `503` (can't embed query)
- `/recent`, `/vectors`, `/health` still work (no embedding needed)
- `/health` shows `"ollama": "unreachable"`

### Invalid Requests

- Missing/empty content: `400 Bad Request`
- Invalid JSON: `400 Bad Request`
- Missing API key: `401 Unauthorized`
- Invalid API key: `401 Unauthorized`

### Observability

All errors logged to Logfire with full context:
- Request details
- Error type and message
- Timing information
- Trace ID for correlation

---

## 11. Development

### Environment Setup

Development happens on Primer to protect production systems.

1. **Postgres**: Runs via `Basement/Postgres/docker-compose.yml` on a non-default port (e.g., 5433)
2. **Ollama**: Already running on Primer with GPU
3. **Cortex**: Run locally via `uv run`

### Running Locally

```bash
cd cortex
uv run uvicorn cortex.main:app --reload --port 7867
```

### Testing

```bash
# Health check
curl http://localhost:7867/health

# Store a memory
curl -X POST http://localhost:7867/store \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $CORTEX_API_KEY" \
  -d '{"content": "Test memory"}'
```

---

## Open Questions

1. **Search weights:** Is 50/50 full-text/semantic the right balance? May need tuning.
2. **Rate limiting:** Not implemented. Add `slowapi` if exposing to internet.
3. **Batch operations:** Not implemented. Add `/store-batch` if needed for bulk import.

---

## Changelog

- **0.4** (2025-12-28): Added `captured_tz` timezone metadata, timezone philosophy section, optional local-time view
- **0.3** (2025-12-28): Added exact-match search (`--exact`), date filtering (`--after`, `--before`, `--date`)
- **0.2** (2025-12-28): Added CLI, HNSW index, Logfire, port 7867, fail-fast timeout, project structure
- **0.1** (2025-12-28): Initial draft
