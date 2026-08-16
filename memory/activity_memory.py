"""Semantic memory for project activity — isolated from the main RAG.

Layer 3 of navigation. Layer 1 (:mod:`actions.file_controller`) finds
*files*, Layer 2 (:mod:`memory.projects`) tracks which *project* and what
was last asked, and this module gives those activity records **semantic
recall**: "which project did we build the auth flow in?" or "the study
plan we made last week" become answerable from a vector index instead of
only by recency.

Isolation is the point, and it is structural rather than by convention.
The main RAG manager (conversation chunks, uploaded documents) lives in
``memory/data/rag/vectors.db``; this store has its **own sqlite file**
(``memory/data/activity/vectors.db``), its own index, its own top-k and
its own retrieval budget. Activity chunks can never crowd conversation
memory out of a result, and conversation chunks never appear in project
memory. The coordinator queries the two stores separately and renders
activity hits in their own prompt section.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from memory.rag.config import RAGConfig
from memory.rag.manager import RAGManager
from memory.rag.models import Document
from memory.rag.stores import SQLiteVectorStore

logger = logging.getLogger("kancha.memory.activity")

_ACTIVITY_SUBDIR = "activity"


class ActivityMemory:
    """Semantic store for condensed project-activity summaries."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        base_config: RAGConfig | None = None,
        embedder: Any | None = None,
    ) -> None:
        self._base_config = base_config or RAGConfig.from_env()
        self._data_dir = (
            Path(data_dir)
            if data_dir
            else self._base_config.data_dir / _ACTIVITY_SUBDIR
        )
        self._embedder = embedder
        self._manager: RAGManager | None = None

    @property
    def ready(self) -> bool:
        return self._manager is not None and self._manager.ready

    @property
    def store_path(self) -> Path:
        return self._data_dir / "vectors.db"

    # ── lifecycle ───────────────────────────────────────────────────

    async def initialize(self) -> None:
        if self.ready:
            return
        config = replace(self._base_config, data_dir=self._data_dir)
        store = SQLiteVectorStore(
            db_path=self.store_path,
            dimensions=config.embedding_dimensions,
        )
        self._manager = RAGManager(config=config, store=store, embedder=self._embedder)
        await self._manager.initialize()
        logger.info("Activity memory ready — store=%s", self.store_path)

    # ── writes ──────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the underlying store (used by tests and shutdown)."""
        if self._manager is not None:
            try:
                await self._manager.close()
            except Exception:  # noqa: BLE001
                logger.debug("Activity memory close ignored an error", exc_info=True)
            self._manager = None

    async def index(
        self,
        summary: str,
        *,
        project: str = "",
        root: str = "",
        session: str = "default",
    ) -> bool:
        """Index one condensed activity record. Never raises, never blocks
        the caller: any failure is logged and the record is skipped."""
        if self._manager is None:
            return False
        content = (summary or "").strip()
        if not content:
            return False
        doc = Document(
            title=f"{project or 'project'} activity",
            content=content,
            source=f"session:{session}",
            doc_type="activity",
            metadata={
                "project": project,
                "root": root,
                "session": session,
            },
        )
        try:
            result = await self._manager.index_document(doc, semantic_dedupe=True)
            if result.chunks_indexed:
                logger.info(
                    "Indexed project activity (%d chunk(s), project=%r)",
                    result.chunks_indexed,
                    project,
                )
            return result.chunks_indexed > 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("Project activity index failed (non-fatal): %s", exc)
            return False

    # ── reads ───────────────────────────────────────────────────────

    def prefetch(self, query: str) -> None:
        """Start embedding *query* for the activity store in the background.

        Embedding is a network round-trip; starting it when the transcript
        lands lets it finish inside the STT/LLM window, so a later
        :meth:`search` of the same text finds it already cached (mirrors
        ``RAGManager.prefetch``). Never blocks and never raises.
        """
        if self._manager is None:
            return
        try:
            self._manager.prefetch(query)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Activity query prefetch failed (harmless): %s", exc)

    async def search(self, query: str, top_k: int = 3) -> list[Any]:
        """Semantic recall over project activity. Returns [] on any failure
        — retrieval enhances a reply, it must never break one."""
        if self._manager is None:
            return []
        try:
            return await self._manager.retrieve(query, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Project activity retrieval failed (non-fatal): %s", exc)
            return []


# ── shared instance (pipeline sets it; executor/coordinator read it) ──

_activity_memory: ActivityMemory | None = None


def get_shared_activity_memory() -> ActivityMemory:
    global _activity_memory
    if _activity_memory is None:
        # A lazily-built instance is NOT ready until initialize() runs;
        # callers gate on `.ready`, so this never queries an unopened db.
        _activity_memory = ActivityMemory()
    return _activity_memory


def set_shared_activity_memory(memory: ActivityMemory | None) -> None:
    global _activity_memory
    _activity_memory = memory
