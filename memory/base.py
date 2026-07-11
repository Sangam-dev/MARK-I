"""Abstract base class for memory backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractMemoryBackend(ABC):
    """Abstract base class defining the interface all memory backends must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier."""
        ...

    @abstractmethod
    async def store(self, content: str, metadata: dict[str, Any]) -> str:
        """Store content with metadata. Returns record ID."""
        ...

    @abstractmethod
    async def retrieve(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant content for a query."""
        ...

    @abstractmethod
    async def delete(self, record_id: str) -> bool:
        """Delete a record by ID. Returns True if deleted."""
        ...

    @abstractmethod
    async def clear_session(self, session_id: str) -> int:
        """Clear all records for a session. Returns count deleted."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is healthy."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...