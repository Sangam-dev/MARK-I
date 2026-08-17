"""MemoryManager orchestrates short-term context and durable memories.

Durable memory (SQL facts + RAG entries) is fed exclusively by the
primary Gemini response: the model returns a JSON envelope
``{message, sql?, rag?}`` and the coordinator calls
:meth:`save_sql` / :meth:`append_rag` directly. There is no separate
extraction step after the conversation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from core.bus import EventBus

from .structured import StructuredMemory

_RAG_SEPARATOR = "-" * 40


@dataclass(frozen=True)
class ConversationContext:
    """Short-term in-memory conversation buffer."""

    max_history: int = 12
    _buffer: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def add(
        self, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add an interaction to the buffer."""
        self._buffer.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {},
            }
        )
        # Keep only recent history (in-place slice to avoid frozen dataclass reassignment)
        if len(self._buffer) > self.max_history:
            self._buffer[:] = self._buffer[-self.max_history :]

    def get_recent(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get recent interactions."""
        limit = limit or self.max_history
        return self._buffer[-limit:]

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()


class MemoryManager:
    """Orchestrates short-term context and durable structured facts."""

    def __init__(
        self,
        bus: EventBus,
        data_dir: Path,
        session_id: str,
        max_short_term: int = 12,
    ) -> None:
        self._bus = bus
        self._session_id = session_id
        self._data_dir = data_dir

        # Short-term memory (in-memory)
        self._short_term = ConversationContext(max_history=max_short_term)

        # Structured memory (SQLite)
        self._structured = StructuredMemory(data_dir / "structured.db")

        self._initialized = False

    @property
    def short_term(self) -> ConversationContext:
        return self._short_term

    @property
    def structured(self) -> StructuredMemory:
        return self._structured

    async def initialize(self) -> None:
        """Initialize structured memory only."""
        if self._initialized:
            return

        await self._structured.initialize()
        self._initialized = True

    async def close(self) -> None:
        """Close memory backends."""
        await self._structured.close()
        self._initialized = False

    # --- Convenience methods for direct use ---

    async def save_sql(self, items: list[dict[str, Any]]) -> int:
        """Upsert ``{key, value}`` items into structured (SQLite) memory.

        Reuses :meth:`StructuredMemory.store_fact`, which updates
        existing keys in place, inserts missing keys, and never creates
        duplicate records. Returns the number of items persisted.
        """
        saved = 0
        for item in items or []:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            value = item.get("value")
            if key is None or value is None:
                continue
            await self._structured.store_fact(str(key), str(value), self._session_id)
            saved += 1
        return saved

    async def append_rag(self, items: list[dict[str, Any]]) -> int:
        """Append ``{type, title, content}`` entries to ``memory/rag.txt``.

        The file is created automatically on first use and is append-only
        — existing content is never overwritten. Returns the number of
        entries appended.
        """
        path = self._rag_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        entries = [
            self._format_rag_entry(
                type_=str(item.get("type", "")),
                title=str(item.get("title", "")),
                content=str(item.get("content", "")),
            )
            for item in items or []
            if isinstance(item, dict) and str(item.get("content", "")).strip()
        ]
        if not entries:
            return 0

        with path.open("a", encoding="utf-8") as handle:
            handle.write("".join(entries))
        return len(entries)

    @staticmethod
    def _rag_file_path() -> Path:
        """Location of the RAG memory file, inside the ``memory/`` package."""
        return Path(__file__).resolve().parent / "rag.txt"

    @staticmethod
    def _format_rag_entry(type_: str, title: str, content: str) -> str:
        """Render one RAG entry block for ``rag.txt``."""
        timestamp = datetime.utcnow().isoformat()
        return (
            f"{_RAG_SEPARATOR}\n"
            f"Timestamp: {timestamp}\n"
            f"Type: {type_}\n"
            f"Title: {title}\n"
            f"\n"
            f"{content}\n"
            f"\n"
            f"{_RAG_SEPARATOR}\n"
        )

    async def store_fact(self, key: str, value: str) -> str:
        """Store a fact in structured memory."""
        return await self._structured.store_fact(key, value, self._session_id)

    async def get_fact(self, key: str) -> str | None:
        """Get a fact from structured memory."""
        return await self._structured.get_fact(key, self._session_id)

    async def get_all_facts(self) -> list[dict[str, Any]]:
        """Get all facts for the session."""
        return await self._structured.get_all_facts(self._session_id)

    async def store_task(
        self,
        description: str,
        due_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a task in structured memory."""
        return await self._structured.store_task(
            description=description,
            due_at=due_at,
            session_id=self._session_id,
            metadata=metadata,
        )

    async def get_pending_tasks(self) -> list[dict[str, Any]]:
        """Get pending tasks for the session."""
        return await self._structured.get_pending_tasks(self._session_id)

    async def update_task_status(self, task_id: str, status: str) -> bool:
        """Update task status."""
        return await self._structured.update_task_status(task_id, status)

    async def clear_session(self) -> int:
        """Clear all memory for this session."""
        self._short_term.clear()
        return await self._structured.clear_session(self._session_id)
