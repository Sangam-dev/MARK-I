"""Ollama embedding provider — fully local, no API key, no data leaving
the machine.

Select with ``KANCHA_RAG_EMBEDDING_PROVIDER=ollama``. Requires a running
Ollama daemon with an embedding model pulled::

    ollama pull nomic-embed-text

Uses ``httpx`` (already a project dependency) rather than the ``ollama``
package, so enabling this provider needs no new install.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import EmbeddingError, EmbeddingProvider, l2_normalise

logger = logging.getLogger("kancha.memory.rag.embeddings.ollama")


class OllamaEmbedder(EmbeddingProvider):
    """Embeddings via a local Ollama daemon's ``/api/embed`` endpoint."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        dimensions: int = 768,
        base_url: str = "http://localhost:11434",
        batch_size: int = 16,
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._dimensions = max(8, dimensions)
        self._base_url = base_url.rstrip("/")
        self._batch_size = max(1, batch_size)
        self._timeout = timeout
        self._client: Any = None

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import httpx  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover
                raise EmbeddingError("httpx is required for the ollama provider") from exc
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self._client

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        out: list[list[float]] = []

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                response = await client.post(
                    "/api/embed", json={"model": self._model, "input": batch}
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(
                    f"Ollama embedding failed against {self._base_url}: {exc}"
                ) from exc

            vectors = payload.get("embeddings")
            if vectors is None:
                # Older daemons expose /api/embeddings returning a single
                # "embedding" key; support both wire shapes.
                single = payload.get("embedding")
                vectors = [single] if single else []

            if len(vectors) != len(batch):
                raise EmbeddingError(
                    f"Ollama returned {len(vectors)} embeddings for {len(batch)} inputs"
                )
            out.extend(l2_normalise(list(v)[: self._dimensions]) for v in vectors)

        return out

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts)

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed([text])
        if not vectors:
            raise EmbeddingError("Ollama returned no embedding for the query")
        return vectors[0]

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
