"""Typed data objects shared across the RAG subsystem.

The flow these types describe::

    Loader   -> Document          (whole file / whole conversation entry)
    Chunker  -> Chunk             (embeddable slice, carries provenance)
    Embedder -> VectorRecord      (Chunk + its embedding, what the store persists)
    Store    -> VectorMatch       (raw similarity hit)
    Manager  -> RetrievedChunk    (what the Conversation Manager sees)

The Conversation Manager only ever handles :class:`RetrievedChunk`; every
other type is internal to the RAG layer.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Source kinds. ``upload`` comes from the file pipeline, ``conversation``
# from the ``rag`` array of a Gemini response.
SOURCE_UPLOAD = "upload"
SOURCE_CONVERSATION = "conversation"

_WHITESPACE_RE = re.compile(r"\s+")


def new_id(prefix: str) -> str:
    """Short, readable, collision-safe identifier."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def utc_now() -> str:
    """ISO-8601 UTC timestamp used for every ``created_at`` field."""
    return datetime.now(timezone.utc).isoformat()


def content_hash(text: str) -> str:
    """Stable hash of *text* used for exact-duplicate detection.

    Whitespace is collapsed and case is folded first so that trivially
    reformatted duplicates ("the   same\\n text" vs "The same text")
    still collide. Semantic near-duplicates are caught separately by the
    manager's cosine check.
    """
    normalised = _WHITESPACE_RE.sub(" ", text or "").strip().casefold()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class Document:
    """A whole source document, before chunking.

    Every loader returns this shape regardless of the underlying file
    format, which is what lets :mod:`memory.rag.ingest` stay format
    agnostic.
    """

    title: str
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_type: str = "document"
    id: str = field(default_factory=lambda: new_id("doc"))
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        # A document with no usable title is a real possibility (a PDF
        # with no metadata, a conversation entry the model didn't label).
        # Fall back to the source's basename so the UI never shows blanks.
        if not self.title.strip():
            self.title = self.source.rsplit("/", 1)[-1] or "Untitled"

    @property
    def char_count(self) -> int:
        return len(self.content)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "doc_type": self.doc_type,
            "created_at": self.created_at,
            "char_count": self.char_count,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class Chunk:
    """One embeddable slice of a :class:`Document`.

    ``metadata`` always carries at least ``document_id``, ``source``,
    ``title``, ``doc_type`` and ``chunk_index``; loaders that know about
    pagination (PDF) additionally set ``page``.
    """

    document_id: str
    content: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("chk"))

    @property
    def hash(self) -> str:
        return content_hash(self.content)


@dataclass(slots=True)
class VectorRecord:
    """A chunk plus its embedding — the unit the vector store persists."""

    chunk_id: str
    document_id: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = content_hash(self.content)


@dataclass(slots=True)
class VectorMatch:
    """A raw similarity hit returned by a :class:`~.stores.base.VectorStore`."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    """Structured retrieval result handed to the Conversation Manager.

    This is the *only* RAG type that crosses the subsystem boundary. It
    carries everything a prompt builder could need — and deliberately no
    prompt-shaped strings, because building prompts is the Conversation
    Manager's job, not the RAG Manager's.
    """

    title: str
    doc_type: str
    content: str
    score: float
    source: str
    document_id: str
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_match(cls, match: VectorMatch) -> "RetrievedChunk":
        meta = dict(match.metadata or {})
        return cls(
            title=str(meta.get("title") or "Untitled"),
            doc_type=str(meta.get("doc_type") or "document"),
            content=match.content,
            score=round(float(match.score), 4),
            source=str(meta.get("source") or ""),
            document_id=match.document_id,
            chunk_id=match.chunk_id,
            metadata=meta,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "type": self.doc_type,
            "score": self.score,
            "content": self.content,
            "source": self.source,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class IndexResult:
    """Outcome of an indexing operation, returned by the RAG Manager."""

    document_id: str
    title: str
    chunks_indexed: int
    chunks_skipped: int
    skipped_reason: str = ""

    @property
    def indexed(self) -> bool:
        return self.chunks_indexed > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "chunks_indexed": self.chunks_indexed,
            "chunks_skipped": self.chunks_skipped,
            "skipped_reason": self.skipped_reason,
            "indexed": self.indexed,
        }
