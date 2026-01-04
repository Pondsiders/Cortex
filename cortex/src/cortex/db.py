"""Postgres database operations."""

import json
from datetime import datetime, timezone
from typing import Any

import asyncpg
import logfire


class Database:
    """Async Postgres connection pool and queries."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: asyncpg.Pool | None = None

    async def connect(self):
        """Create connection pool."""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=2,
            max_size=10,
        )
        logfire.info("Database connected")

    async def disconnect(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logfire.info("Database disconnected")

    async def health_check(self) -> tuple[bool, int | None]:
        """Check database health, return (healthy, memory_count)."""
        if not self.pool:
            return False, None
        try:
            async with self.pool.acquire() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM cortex.memories WHERE NOT forgotten"
                )
                return True, count
        except Exception as e:
            logfire.error("Database health check failed: {error}", error=str(e))
            return False, None

    async def store_memory(
        self,
        content: str,
        embedding: list[float],
        tags: list[str] | None = None,
        timezone_str: str | None = None,
    ) -> tuple[int, datetime]:
        """Store a new memory. Returns (id, created_at)."""
        created_at = datetime.now(timezone.utc)
        metadata = {
            "created_at": created_at.isoformat(),
            "captured_tz": timezone_str,
        }
        if tags:
            metadata["tags"] = tags

        with logfire.span("db_store_memory"):
            async with self.pool.acquire() as conn:
                memory_id = await conn.fetchval(
                    """
                    INSERT INTO cortex.memories (content, embedding, metadata)
                    VALUES ($1, $2, $3)
                    RETURNING id
                    """,
                    content,
                    json.dumps(embedding),  # pgvector accepts JSON array
                    json.dumps(metadata),
                )
                return memory_id, created_at

    async def search_memories(
        self,
        query_embedding: list[float] | None,
        query_text: str,
        limit: int = 10,
        include_forgotten: bool = False,
        exact: bool = False,
        after: datetime | None = None,
        before: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search memories with hybrid (full-text + semantic) scoring.

        If exact=True, only uses full-text search (no embedding needed).
        """
        with logfire.span("db_search_memories", limit=limit, exact=exact):
            async with self.pool.acquire() as conn:
                # Build the WHERE clause
                conditions = []
                params = []
                param_idx = 1

                if not include_forgotten:
                    conditions.append("NOT forgotten")

                if after:
                    conditions.append(f"(metadata->>'created_at')::timestamptz >= ${param_idx}")
                    params.append(after)
                    param_idx += 1

                if before:
                    conditions.append(f"(metadata->>'created_at')::timestamptz < ${param_idx}")
                    params.append(before)
                    param_idx += 1

                where_clause = " AND ".join(conditions) if conditions else "TRUE"

                if exact:
                    # Full-text search only
                    query = f"""
                        SELECT
                            id,
                            content,
                            metadata,
                            ts_rank(content_tsv, plainto_tsquery('english', ${param_idx})) as score
                        FROM cortex.memories
                        WHERE {where_clause}
                          AND content_tsv @@ plainto_tsquery('english', ${param_idx})
                        ORDER BY score DESC
                        LIMIT ${param_idx + 1}
                    """
                    params.extend([query_text, limit])
                else:
                    # Hybrid search: 50% full-text + 50% semantic
                    embedding_json = json.dumps(query_embedding)
                    query = f"""
                        WITH scored AS (
                            SELECT
                                id,
                                content,
                                metadata,
                                -- Normalize full-text score to 0-1 range
                                COALESCE(
                                    ts_rank(content_tsv, plainto_tsquery('english', ${param_idx})),
                                    0
                                ) as fts_score,
                                -- Cosine similarity is already 0-1 for normalized vectors
                                1 - (embedding <=> ${param_idx + 1}::vector) as sem_score
                            FROM cortex.memories
                            WHERE {where_clause}
                              AND embedding IS NOT NULL
                        )
                        SELECT
                            id,
                            content,
                            metadata,
                            (0.5 * LEAST(fts_score, 1.0) + 0.5 * sem_score) as score
                        FROM scored
                        ORDER BY score DESC
                        LIMIT ${param_idx + 2}
                    """
                    params.extend([query_text, embedding_json, limit])

                rows = await conn.fetch(query, *params)

                return [
                    {
                        "id": row["id"],
                        "content": row["content"],
                        "metadata": json.loads(row["metadata"]),
                        "score": float(row["score"]),
                    }
                    for row in rows
                ]

    async def get_recent_memories(
        self,
        limit: int = 10,
        hours: int = 24,
    ) -> list[dict[str, Any]]:
        """Get recent memories within the specified time window."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        with logfire.span("db_recent_memories", limit=limit, hours=hours):
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, content, metadata
                    FROM cortex.memories
                    WHERE NOT forgotten
                      AND (metadata->>'created_at')::timestamptz >= $1
                    ORDER BY (metadata->>'created_at')::timestamptz DESC
                    LIMIT $2
                    """,
                    cutoff,
                    limit,
                )

                return [
                    {
                        "id": row["id"],
                        "content": row["content"],
                        "metadata": json.loads(row["metadata"]),
                    }
                    for row in rows
                ]

    async def get_vectors(self, limit: int = 12000) -> list[dict[str, Any]]:
        """Get memories with their embeddings (for visualizer)."""
        with logfire.span("db_get_vectors", limit=limit):
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, content, metadata, embedding::text
                    FROM cortex.memories
                    WHERE NOT forgotten
                      AND embedding IS NOT NULL
                    ORDER BY (metadata->>'created_at')::timestamptz DESC
                    LIMIT $1
                    """,
                    limit,
                )

                return [
                    {
                        "id": row["id"],
                        "content": row["content"],
                        "metadata": json.loads(row["metadata"]),
                        "embedding": json.loads(row["embedding"]),
                    }
                    for row in rows
                ]

    async def forget_memory(self, memory_id: int) -> bool:
        """Soft-delete a memory. Returns True if found and updated."""
        with logfire.span("db_forget_memory", memory_id=memory_id):
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE cortex.memories
                    SET forgotten = TRUE
                    WHERE id = $1 AND NOT forgotten
                    """,
                    memory_id,
                )
                return result == "UPDATE 1"
