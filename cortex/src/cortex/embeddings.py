"""Ollama embeddings client."""

import httpx
import logfire


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""

    pass


class EmbeddingClient:
    """Client for generating embeddings via Ollama."""

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = "nomic-embed-text"
        self.timeout = 5.0  # Fail fast

    async def embed_document(self, content: str) -> list[float]:
        """Generate embedding for a document (for storage)."""
        return await self._embed(f"search_document: {content}")

    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a query (for search)."""
        return await self._embed(f"search_query: {query}")

    async def _embed(self, prompt: str) -> list[float]:
        """Call Ollama API to generate embedding."""
        with logfire.span("ollama_embed", model=self.model):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.ollama_url}/api/embeddings",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "keep_alive": -1,  # Keep model loaded indefinitely
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["embedding"]
            except httpx.TimeoutException:
                logfire.error("Ollama timeout after {timeout}s", timeout=self.timeout)
                raise EmbeddingError("Embedding service timed out")
            except httpx.HTTPStatusError as e:
                logfire.error("Ollama HTTP error: {status}", status=e.response.status_code)
                raise EmbeddingError(f"Embedding service error: {e.response.status_code}")
            except httpx.ConnectError:
                logfire.error("Ollama unreachable at {url}", url=self.ollama_url)
                raise EmbeddingError("Embedding service unreachable")
            except Exception as e:
                logfire.error("Ollama unexpected error: {error}", error=str(e))
                raise EmbeddingError(f"Embedding failed: {e}")

    async def health_check(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
