"""Cortex - Semantic memory service."""

from contextlib import asynccontextmanager
from datetime import datetime

import logfire
import redis
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, status

from .db import Database
from .embeddings import EmbeddingClient, EmbeddingError
from .models import (
    ForgetRequest,
    ForgetResponse,
    HealthResponse,
    MemoryResult,
    MemoryWithVector,
    RecentResponse,
    SearchRequest,
    SearchResponse,
    Settings,
    StoreRequest,
    StoreResponse,
    VectorsRequest,
    VectorsResponse,
)

# Subvox STM keys
STM_MESSAGES_KEY = "stm:messages"
STM_MEMORABLES_KEY = "stm:memorables"

# Global instances
settings: Settings | None = None
db: Database | None = None
embeddings: EmbeddingClient | None = None
redis_client: redis.Redis | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    global settings, db, embeddings, redis_client

    # Load settings
    settings = Settings()

    # Configure Logfire
    logfire.configure(token=settings.logfire_token, service_name="cortex")
    logfire.instrument_asyncpg()
    logfire.instrument_httpx()

    # Initialize database
    db = Database(settings.database_url)
    await db.connect()

    # Initialize embeddings client
    embeddings = EmbeddingClient(settings.ollama_url)

    # Initialize Redis client (optional - for Subvox STM clearing)
    if settings.redis_url:
        try:
            redis_client = redis.from_url(settings.redis_url)
            redis_client.ping()  # Test connection
            logfire.info("Redis connected for Subvox STM")
        except redis.RedisError as e:
            logfire.warning("Redis connection failed, STM clearing disabled", error=str(e))
            redis_client = None

    logfire.info("Cortex started", port=settings.port)

    yield

    # Shutdown
    await db.disconnect()
    if redis_client:
        redis_client.close()
    logfire.info("Cortex stopped")


app = FastAPI(
    title="Cortex",
    description="Semantic memory service for Alpha",
    version="0.1.0",
    lifespan=lifespan,
)

# Instrument FastAPI with Logfire
logfire.instrument_fastapi(app)


async def verify_api_key(x_api_key: str = Header()):
    """Dependency to verify API key."""
    if not settings or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


@app.post("/store", response_model=StoreResponse, status_code=status.HTTP_201_CREATED)
async def store_memory(request: StoreRequest, _: None = Depends(verify_api_key)):
    """Store a new memory."""
    with logfire.span("store_memory"):
        try:
            embedding = await embeddings.embed_document(request.content)
        except EmbeddingError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            )

        memory_id, created_at = await db.store_memory(
            content=request.content,
            embedding=embedding,
            tags=request.tags,
            timezone_str=request.timezone,
        )

        # Clear Subvox STM on successful store
        if redis_client:
            try:
                deleted = redis_client.delete(STM_MESSAGES_KEY, STM_MEMORABLES_KEY)
                logfire.info("Subvox STM cleared", keys_deleted=deleted)
            except redis.RedisError as e:
                logfire.warning("Failed to clear Subvox STM", error=str(e))

        return StoreResponse(id=memory_id, created_at=created_at)


@app.post("/search", response_model=SearchResponse)
async def search_memories(request: SearchRequest, _: None = Depends(verify_api_key)):
    """Search memories using hybrid (full-text + semantic) search."""
    with logfire.span("search_memories", exact=request.exact):
        query_embedding = None
        if not request.exact:
            try:
                query_embedding = await embeddings.embed_query(request.query)
            except EmbeddingError as e:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=str(e),
                )

        results = await db.search_memories(
            query_embedding=query_embedding,
            query_text=request.query,
            limit=request.limit,
            include_forgotten=request.include_forgotten,
            exact=request.exact,
            after=request.after,
            before=request.before,
        )

        memories = [
            MemoryResult(
                id=r["id"],
                content=r["content"],
                created_at=datetime.fromisoformat(r["metadata"]["created_at"]),
                tags=r["metadata"].get("tags"),
                score=r.get("score"),
            )
            for r in results
        ]

        return SearchResponse(memories=memories)


@app.get("/recent", response_model=RecentResponse)
async def get_recent(
    limit: int = 10,
    hours: int = 24,
    _: None = Depends(verify_api_key),
):
    """Get recent memories."""
    with logfire.span("get_recent", limit=limit, hours=hours):
        if limit > 100:
            limit = 100

        results = await db.get_recent_memories(limit=limit, hours=hours)

        memories = [
            MemoryResult(
                id=r["id"],
                content=r["content"],
                created_at=datetime.fromisoformat(r["metadata"]["created_at"]),
                tags=r["metadata"].get("tags"),
            )
            for r in results
        ]

        return RecentResponse(memories=memories)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health."""
    pg_healthy, memory_count = await db.health_check()
    ollama_healthy = await embeddings.health_check()

    status_str = "healthy" if (pg_healthy and ollama_healthy) else "unhealthy"
    status_code = status.HTTP_200_OK if status_str == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE

    response = HealthResponse(
        status=status_str,
        postgres="connected" if pg_healthy else "unreachable",
        ollama="connected" if ollama_healthy else "unreachable",
        memory_count=memory_count,
    )

    if status_code != status.HTTP_200_OK:
        raise HTTPException(status_code=status_code, detail=response.model_dump())

    return response


@app.post("/vectors", response_model=VectorsResponse)
async def get_vectors(request: VectorsRequest, _: None = Depends(verify_api_key)):
    """Get memories with their embeddings (for visualizer)."""
    with logfire.span("get_vectors", limit=request.limit):
        results = await db.get_vectors(limit=request.limit)

        memories = [
            MemoryWithVector(
                id=r["id"],
                content=r["content"],
                created_at=datetime.fromisoformat(r["metadata"]["created_at"]),
                embedding=r["embedding"],
            )
            for r in results
        ]

        return VectorsResponse(memories=memories)


@app.post("/forget", response_model=ForgetResponse)
async def forget_memory(request: ForgetRequest, _: None = Depends(verify_api_key)):
    """Soft-delete a memory."""
    with logfire.span("forget_memory", memory_id=request.id):
        forgotten = await db.forget_memory(request.id)
        return ForgetResponse(forgotten=forgotten)


def run():
    """Entry point for running the server."""
    import os

    # Allow settings to be loaded for host/port
    s = Settings()
    uvicorn.run(
        "cortex.main:app",
        host=s.host,
        port=s.port,
        reload=os.getenv("CORTEX_DEV", "0") == "1",
    )


if __name__ == "__main__":
    run()
