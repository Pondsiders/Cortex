"""Pydantic models for API request/response."""

from datetime import datetime

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables."""

    database_url: str = Field(alias="DATABASE_URL")
    ollama_url: str = Field(default="http://localhost:11434", alias="OLLAMA_URL")
    api_key: str = Field(alias="CORTEX_API_KEY")
    host: str = Field(default="0.0.0.0", alias="CORTEX_HOST")
    port: int = Field(default=7867, alias="CORTEX_PORT")
    lmnr_project_api_key: str | None = Field(default=None, alias="LMNR_PROJECT_API_KEY")
    redis_url: str | None = Field(default=None, alias="REDIS_URL")

    class Config:
        env_file = ".env"
        extra = "ignore"


# Request models


class StoreRequest(BaseModel):
    """Request to store a new memory."""

    content: str = Field(min_length=1)
    tags: list[str] | None = None
    timezone: str | None = None


class SearchRequest(BaseModel):
    """Request to search memories."""

    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    include_forgotten: bool = False
    exact: bool = False
    after: datetime | None = None
    before: datetime | None = None


class VectorsRequest(BaseModel):
    """Request to get memory vectors."""

    limit: int = Field(default=12000, ge=1, le=100000)


class ForgetRequest(BaseModel):
    """Request to forget a memory."""

    id: int


# Response models


class StoreResponse(BaseModel):
    """Response after storing a memory."""

    id: int
    created_at: datetime


class MemoryResult(BaseModel):
    """A memory in search results."""

    id: int
    content: str
    created_at: datetime
    tags: list[str] | None = None
    score: float | None = None


class SearchResponse(BaseModel):
    """Response from memory search."""

    memories: list[MemoryResult]


class RecentResponse(BaseModel):
    """Response from recent memories query."""

    memories: list[MemoryResult]


class MemoryWithVector(BaseModel):
    """A memory with its embedding vector."""

    id: int
    content: str
    created_at: datetime
    embedding: list[float]


class VectorsResponse(BaseModel):
    """Response from vectors endpoint."""

    memories: list[MemoryWithVector]


class HealthResponse(BaseModel):
    """Response from health check."""

    status: str
    postgres: str
    ollama: str
    memory_count: int | None


class ForgetResponse(BaseModel):
    """Response from forget endpoint."""

    forgotten: bool
