"""Pydantic models for the WebSocket envelope and REST request/response bodies.

Keeping these in one place makes the wire format easy to audit against
answers/integration_plan.md and answers/guide.md.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ── WebSocket ────────────────────────────────────────────────────────────────


class WSIncoming(BaseModel):
    """Envelope for messages sent from the frontend to the backend over `/ws`.

    type:
        "user_text"   -> payload: {"text": str}. Injected onto the bus as a
                         TextInputReceived event (same as CLI --text mode).
        "retry_last"  -> payload: {} (re-runs the last task; coordinator's
                         retry-phrase regex already understands "retry it").
        "ping"        -> payload: {}. Server replies with "pong" (used by the
                         frontend for a crude WS round-trip latency reading).
    """

    type: Literal["user_text", "retry_last", "ping"]
    payload: dict[str, Any] = Field(default_factory=dict)
    session_id: str = "default"


# ── REST: settings ───────────────────────────────────────────────────────────


class SettingsModel(BaseModel):
    """Persisted user-facing settings (memory/data/settings.json).

    NOTE: `tts_enabled` here only controls *future* process starts today —
    TTSHandler is registered once at pipeline build time. Toggling this at
    runtime does not hot-swap the handler yet (see answers/guide.md,
    "known limitations").
    """

    tts_enabled: bool = True
    voice_mode: bool = False
    session_id: str = "default"


# ── REST: alarms ─────────────────────────────────────────────────────────────


class AlarmCreateRequest(BaseModel):
    command: str
    delay_seconds: int | None = None


# ── REST: generic action result envelope ────────────────────────────────────


class ActionResult(BaseModel):
    success: bool
    message: str


# ── REST: RAG (upload + document management) ─────────────────────────────────
#
# These mirror the dataclasses in memory/rag/models.py and
# memory/rag/ingest.py. They are re-declared here as Pydantic models rather
# than reused directly so the wire format stays owned by the API layer and
# cannot drift accidentally when an internal dataclass gains a field.


class RAGUploadResult(BaseModel):
    """Response for ``POST /api/rag/upload``."""

    success: bool
    filename: str
    message: str
    document_id: str = ""
    title: str = ""
    chunks_indexed: int = 0
    chunks_skipped: int = 0
    error: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGDocumentOut(BaseModel):
    """One indexed document, as listed by ``GET /api/rag/documents``."""

    id: str
    title: str
    source: str = ""
    doc_type: str = "document"
    created_at: str = ""
    updated_at: str = ""
    chunk_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGDeleteResult(BaseModel):
    document_id: str
    title: str = ""
    chunks_removed: int = 0


class RAGSearchHit(BaseModel):
    """One retrieval result. Matches ``RetrievedChunk.to_dict()``."""

    title: str
    type: str
    score: float
    content: str
    source: str = ""
    document_id: str = ""
    chunk_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGSearchResponse(BaseModel):
    query: str
    count: int
    results: list[RAGSearchHit] = Field(default_factory=list)


class RAGStats(BaseModel):
    """Index snapshot + effective config, for the UI and for debugging."""

    ready: bool = False
    enabled: bool = False
    embedder: str = ""
    store: str = ""
    dimensions: int = 0
    documents: int = 0
    chunks: int = 0
    top_k: int = 0
    similarity_threshold: float = 0.0
    chunk_size: int = 0
    chunk_overlap: int = 0
    supported_extensions: list[str] = Field(default_factory=list)
    max_upload_bytes: int = 0
    error: str = ""
