"""Disabled vector memory backend.

RAG/vector memory is intentionally turned off for now. This module remains as
a compatibility shim so old imports do not pull in ChromaDB or Ollama.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import AbstractMemoryBackend


class VectorMemory(AbstractMemoryBackend):
    """No-op vector backend used while RAG is disabled."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        ollama_url: str,
        embedding_model: str,
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._ollama_url = ollama_url
        self._embedding_model = embedding_model

    @property
    def name(self) -> str:
        return "vector_disabled"

    async def initialize(self) -> None:
        """Do nothing while vector memory is disabled."""
        return None

    async def store(self, content: str, metadata: dict[str, Any]) -> str:
        """Ignore vector writes while RAG is disabled."""
        return ""

    async def retrieve(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return no vector results while RAG is disabled."""
        return []

    async def delete(self, record_id: str) -> bool:
        """No-op delete."""
        return False

    async def clear_session(self, session_id: str) -> int:
        """No-op clear."""
        return 0

    async def health_check(self) -> bool:
        """Report disabled/unhealthy so callers know vector memory is off."""
        return False

    async def close(self) -> None:
        """No resources to release."""
        return None
