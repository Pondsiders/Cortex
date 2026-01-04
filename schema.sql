-- Cortex schema
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
