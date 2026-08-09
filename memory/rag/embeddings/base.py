"""Embedding provider interface.

Any class implementing :class:`EmbeddingProvider` can back the RAG
subsystem. The manager only ever calls :meth:`embed_documents` and
:meth:`embed_query`, so swapping providers (Gemini -> Ollama -> a local
sentence-transformer) never touches the manager, the store, or the
Conversation Manager.

All providers return **L2-normalised** vectors. That is a contract, not
an implementation detail: the SQLite store computes cosine similarity as
a plain dot product and would silently return wrong scores if a provider
handed back unnormalised vectors.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

logger = logging.getLogger("kancha.memory.rag.embeddings")


def l2_normalise(vector: list[float]) -> list[float]:
    """Scale *vector* to unit length. A zero vector is returned unchanged."""
    norm = math.sqrt(sum(component * component for component in vector))
    if norm <= 1e-12:
        return list(vector)
    return [component / norm for component in vector]


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails unrecoverably."""


class EmbeddingProvider(ABC):
    """Abstract embedding backend."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs and health output."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector width this provider emits.

        The store persists this alongside each vector so a provider swap
        is detected instead of producing garbage similarity scores.
        """

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks (retrieval-corpus side).

        Must return one vector per input, in order. Raises
        :class:`EmbeddingError` if the batch cannot be embedded.
        """

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single search query (retrieval-query side).

        Kept separate from :meth:`embed_documents` because several
        providers (Gemini included) accept an asymmetric task-type hint
        that measurably improves retrieval quality.
        """

    async def health_check(self) -> bool:
        """Return True if the provider can currently produce embeddings."""
        try:
            vector = await self.embed_query("health check")
            return len(vector) == self.dimensions
        except Exception as exc:  # noqa: BLE001
            logger.debug("Embedding health check failed for %s: %s", self.name, exc)
            return False

    async def close(self) -> None:
        """Release any held resources. Default is a no-op."""
        return None
