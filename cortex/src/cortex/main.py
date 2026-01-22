"""Cortex - Semantic memory service.

OTel instrumentation is handled by opentelemetry-instrument wrapper at runtime.
Set OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_RESOURCE_ATTRIBUTES env vars to enable.
"""

from contextlib import asynccontextmanager
from datetime import datetime

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

# Redis pubsub channel for Intro to know when memories are stored
CORTEX_STORED_CHANNEL = "cortex:stored:{session_id}"

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

    # Initialize database
    db = Database(settings.database_url)
    await db.connect()

    # Initialize embeddings client
    embeddings = EmbeddingClient(settings.ollama_url)

    # Initialize Redis client (for publishing store events to Intro)
    if settings.redis_url:
        try:
            redis_client = redis.from_url(settings.redis_url)
            redis_client.ping()  # Test connection
            print("[Cortex] Redis connected for Intro notifications")
        except redis.RedisError as e:
            print(f"[Cortex] Redis connection failed, Intro notifications disabled: {e}")
            redis_client = None

    print(f"[Cortex] Started on port {settings.port}")

    yield

    # Shutdown
    await db.disconnect()
    if redis_client:
        redis_client.close()
    print("[Cortex] Stopped")


app = FastAPI(
    title="Cortex",
    description="Semantic memory service for Alpha",
    version="0.1.0",
    lifespan=lifespan,
)


async def verify_api_key(x_api_key: str = Header()):
    """Dependency to verify API key."""
    if not settings or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


@app.post("/store", response_model=StoreResponse, status_code=status.HTTP_201_CREATED)
async def store_memory(
    request: StoreRequest,
    _: None = Depends(verify_api_key),
    x_session_id: str | None = Header(default=None),
):
    """Store a new memory."""
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

    # Publish to Intro so they can clear their buffers for this session
    if redis_client and x_session_id:
        try:
            channel = CORTEX_STORED_CHANNEL.format(session_id=x_session_id)
            redis_client.publish(channel, str(memory_id))
            print(f"[Cortex] Published to {channel}: memory_id={memory_id}")
        except redis.RedisError as e:
            print(f"[Cortex] Failed to publish store event: {e}")

    return StoreResponse(id=memory_id, created_at=created_at)


@app.post("/search", response_model=SearchResponse)
async def search_memories(request: SearchRequest, _: None = Depends(verify_api_key)):
    """Search memories using hybrid (full-text + semantic) search."""
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
    forgotten = await db.forget_memory(request.id)
    return ForgetResponse(forgotten=forgotten)


@app.get("/get/{memory_id}", response_model=MemoryResult)
async def get_memory_by_id(memory_id: int, _: None = Depends(verify_api_key)):
    """Get a single memory by ID."""
    result = await db.get_memory(memory_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {memory_id} not found",
        )
    return MemoryResult(
        id=result["id"],
        content=result["content"],
        created_at=datetime.fromisoformat(result["metadata"]["created_at"]),
        tags=result["metadata"].get("tags"),
    )


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
