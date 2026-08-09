"""Vector store interface.

The Conversation Manager never sees this type — only the RAG Manager
does. That indirection is the whole point: replacing SQLite with Chroma,
Qdrant or pgvector means writing one new subclass and changing one env
var, with no edits anywhere outside :mod:`memory.rag.stores`.

Implementations own persistence only. Embedding generation, chunking,
deduplication policy and prompt formatting all live above this layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import Document, VectorMatch, VectorRecord


class VectorStore(ABC):
    """Abstract persistence layer for embedded chunks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs and health output."""

    @abstractmethod
    async def initialize(self) -> None:
        """Open connections and create schema. Must be idempotent."""

    @abstractmethod
    async def add_documents(
        self, document: Document, records: list[VectorRecord]
    ) -> int:
        """Persist *records* belonging to *document*.

        Returns the number of chunks written. Implementations must upsert
        the document row so re-indexing an existing id refreshes its
        metadata rather than creating a duplicate entry.
        """

    @abstractmethod
    async def search(
        self,
        embedding: list[float],
        top_k: int,
        threshold: float = 0.0,
        doc_types: tuple[str, ...] | None = None,
    ) -> list[VectorMatch]:
        """Return the ``top_k`` most similar chunks above *threshold*.

        Similarity is cosine, in ``[-1, 1]`` but effectively ``[0, 1]``
        for text embeddings. An empty index returns ``[]`` — never an
        error; callers rely on that for graceful degradation.
        """

    @abstractmethod
    async def delete_document(self, document_id: str) -> int:
        """Delete a document and all of its chunks. Returns chunks removed."""

    @abstractmethod
    async def update_document(
        self, document: Document, records: list[VectorRecord]
    ) -> int:
        """Replace a document's chunks wholesale. Returns chunks written."""

    @abstractmethod
    async def list_documents(self) -> list[dict[str, Any]]:
        """Return document-level metadata for every indexed document."""

    @abstractmethod
    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        """Return one document's metadata, or ``None`` if unknown."""

    @abstractmethod
    async def existing_hashes(self, hashes: list[str]) -> set[str]:
        """Return the subset of *hashes* already present in the index.

        Used by the RAG Manager for cheap exact-duplicate rejection
        before spending an embedding call.
        """

    @abstractmethod
    async def count_chunks(self) -> int:
        """Total number of indexed chunks. ``0`` means an empty index."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""

    async def warm(self) -> None:
        """Preload whatever the first search would otherwise load lazily.

        Optional. Backends that keep an in-process similarity matrix use
        this to move that cost off the first user turn; backends that
        query an external service can leave it as the default no-op.
        """
        return None

    async def health_check(self) -> bool:
        """Return True if the store is queryable right now."""
        try:
            await self.count_chunks()
            return True
        except Exception:  # noqa: BLE001
            return False
