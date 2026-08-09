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
from core.events import MemoryRetrieved

from .structured import StructuredMemory

# RAG/vector memory is intentionally disabled for now.
# from .vector import VectorMemory

_RAG_SEPARATOR = "-" * 40


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

        # Plan output cache: plan_id -> {task_id: result_str}. Used by
        # the planner's ReferenceResolver to bind conversational
        # references ("it", "that folder") against the most recent
        # completed task results. Keyed by plan_id so two plans never
        # collide; kept in-memory only because plans themselves are
        # in-memory (see planning/scheduler.py).
        self._recent_plan_outputs: dict[str, dict[str, str]] = {}

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
        """Add an interaction to short-term memory only.

        Durable memory is no longer extracted from raw turns — it comes
        exclusively from the primary Gemini response via
        :meth:`save_sql` and :meth:`append_rag`.
        """
        self._short_term.add(role, content, metadata)

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

    # --- Plan-output cache (used by the planner) ---

    def record_plan_output(
        self,
        plan_id: str,
        task_id: str,
        result: str,
    ) -> None:
        """Cache a completed task's result for reference resolution.

        Called by :class:`planning.scheduler.PlanScheduler` after each
        task completion. Used by the planner on the next user request
        to bind references like "it" against the most recent artifact.
        """
        bucket = self._recent_plan_outputs.setdefault(plan_id, {})
        bucket[task_id] = result

    def get_recent_plan_outputs(
        self,
        plan_id: str | None = None,
    ) -> dict[str, str]:
        """Return plan outputs as a flat ``{task_id: result}`` map.

        With no ``plan_id`` argument, returns the union of all known
        plan outputs (newer plans override older ones on overlap, since
        Python dicts preserve insertion order). Used by the
        ReferenceResolver when the user starts a fresh request and the
        most recent prior plan's outputs should be the binding source.
        """
        if plan_id is not None:
            return dict(self._recent_plan_outputs.get(plan_id, {}))
        merged: dict[str, str] = {}
        for bucket in self._recent_plan_outputs.values():
            merged.update(bucket)
        return merged

    def clear_plan_outputs(self, plan_id: str | None = None) -> None:
        if plan_id is None:
            self._recent_plan_outputs.clear()
        else:
            self._recent_plan_outputs.pop(plan_id, None)

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
