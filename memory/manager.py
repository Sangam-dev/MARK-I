"""MemoryManager orchestrates short-term context and structured facts."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from core.bus import EventBus, subscribe
from core.events import MemoryRetrieved, MemoryUpdateNeeded

from .structured import StructuredMemory

# RAG/vector memory is intentionally disabled for now.
# from .vector import VectorMemory


@dataclass(frozen=True)
class ConversationContext:
    """Short-term in-memory conversation buffer."""

    session_id: str
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

    def estimate_tokens(self) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        total_chars = sum(len(item["content"]) for item in self._buffer)
        return total_chars // 4


class MemoryManager:
    """Orchestrates short-term context and durable structured facts."""

    def __init__(
        self,
        bus: EventBus,
        data_dir: Path,
        session_id: str,
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        max_short_term: int = 12,
    ) -> None:
        self._bus = bus
        self._session_id = session_id
        self._data_dir = data_dir

        # Short-term memory (in-memory)
        self._short_term = ConversationContext(
            session_id=session_id, max_history=max_short_term
        )

        # Structured memory (SQLite)
        self._structured = StructuredMemory(data_dir / "structured.db")

        # RAG/vector memory is intentionally disabled for now.
        # self._vector = VectorMemory(
        #     persist_dir=data_dir / "vector",
        #     collection_name=f"kancha_{session_id}",
        #     ollama_url=ollama_url,
        #     embedding_model=embedding_model,
        # )
        self._vector = None

        self._initialized = False

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def short_term(self) -> ConversationContext:
        return self._short_term

    @property
    def structured(self) -> StructuredMemory:
        return self._structured

    @property
    def vector(self) -> None:
        return self._vector

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

    # --- Event handlers ---

    _FACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
        (
            re.compile(r"\bmy name is\s+(.+?)[.!?]*$", re.IGNORECASE),
            "name",
        ),
        (
            re.compile(r"\bi am called\s+(.+?)[.!?]*$", re.IGNORECASE),
            "name",
        ),
        (
            re.compile(r"\bcall me\s+(.+?)[.!?]*$", re.IGNORECASE),
            "preferred_name",
        ),
        (
            re.compile(r"\bi live in\s+(.+?)[.!?]*$", re.IGNORECASE),
            "location",
        ),
        (
            re.compile(r"\bi am from\s+(.+?)[.!?]*$", re.IGNORECASE),
            "origin",
        ),
        (
            re.compile(r"\bmy birthday is\s+(.+?)[.!?]*$", re.IGNORECASE),
            "birthday",
        ),
        (
            re.compile(r"\bmy email is\s+(.+?)[.!?]*$", re.IGNORECASE),
            "email",
        ),
        (
            re.compile(r"\bmy phone(?: number)? is\s+(.+?)[.!?]*$", re.IGNORECASE),
            "phone",
        ),
        (
            re.compile(r"\bi (?:like|love|prefer)\s+(.+?)[.!?]*$", re.IGNORECASE),
            "preference",
        ),
        (
            re.compile(r"\bi do not like\s+(.+?)[.!?]*$", re.IGNORECASE),
            "dislike",
        ),
        (
            re.compile(r"\bi hate\s+(.+?)[.!?]*$", re.IGNORECASE),
            "dislike",
        ),
        (
            re.compile(r"\bremember that\s+(.+?)[.!?]*$", re.IGNORECASE),
            "remembered_fact",
        ),
    )

    @classmethod
    def _extract_fact(cls, content: str) -> tuple[str, str] | None:
        """Extract one explicit durable fact from a user message."""
        text = " ".join(content.strip().split())
        if not text:
            return None

        for pattern, key in cls._FACT_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            value = match.group(1).strip(" .!?\"'")
            if len(value) < 2:
                return None
            if key in {"preference", "dislike", "remembered_fact"}:
                key = f"{key}:{value[:48].lower()}"
            return key, value

        return None

    @subscribe(MemoryUpdateNeeded)
    async def _on_memory_update(self, event: MemoryUpdateNeeded) -> None:
        """
        Persist only durable facts to SQLite.

        Short-term memory (in-memory buffer) is managed directly by the
        ReasoningCoordinator with synchronous adds — NOT via this event
        handler — to avoid race conditions. This handler is DB-only and
        intentionally does not store whole conversation turns.
        """
        if event.session_id != self._session_id:
            return
        if event.role != "user":
            return

        fact = self._extract_fact(event.content)
        if fact is None:
            return

        key, value = fact
        try:
            await self._structured.store_fact(key, value, self._session_id)
        except Exception as exc:
            import logging

            logging.getLogger("kancha.memory.manager").warning(
                "Fact store failed: %s", exc
            )

    async def retrieve_context(
        self,
        query: str,
        short_term_limit: int = 6,
        structured_limit: int = 5,
        vector_limit: int = 3,
    ) -> MemoryRetrieved:
        """Retrieve short-term context and stored facts."""
        # Short-term: recent conversation
        short_term = self._short_term.get_recent(short_term_limit)

        # Structured: durable facts only.
        facts = await self._structured.get_all_facts(self._session_id)

        return MemoryRetrieved(
            session_id=self._session_id,
            query=query,
            structured_context=facts,
            episodic_context=[],
        )

    # --- Convenience methods for direct use ---

    async def add_interaction(
        self, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add an interaction to short-term memory and store user facts only."""
        # Short-term
        self._short_term.add(role, content, metadata)

        if role == "user":
            fact = self._extract_fact(content)
            if fact is not None:
                key, value = fact
                await self._structured.store_fact(key, value, self._session_id)

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

    async def health_check(self) -> dict[str, bool]:
        """Check health of all memory layers."""
        return {
            "structured": await self._structured.health_check(),
            "vector": False,
        }
