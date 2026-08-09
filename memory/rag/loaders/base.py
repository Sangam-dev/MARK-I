"""Document loader interface.

Every loader turns one file into one :class:`~memory.rag.models.Document`
with the same four-field shape — ``title``, ``content``, ``source``,
``metadata`` — regardless of the source format. That uniformity is what
lets :mod:`memory.rag.ingest` handle PDFs, Markdown and Word documents
through a single code path, and what makes adding a new format a
self-contained change.

Adding a format:

1. Subclass :class:`DocumentLoader`, set ``extensions`` and ``name``.
2. Implement :meth:`DocumentLoader.load`.
3. Register it in :mod:`memory.rag.loaders.registry`.
4. Add the extension to ``KANCHA_RAG_EXTENSIONS`` (or the default tuple
   in :mod:`memory.rag.config`).
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..models import Document

logger = logging.getLogger("kancha.memory.rag.loaders")


class LoaderError(RuntimeError):
    """Raised when a file cannot be turned into a Document."""


class UnsupportedFormatError(LoaderError):
    """Raised when no registered loader handles a file's extension."""


class DocumentLoader(ABC):
    """Abstract single-format document loader."""

    #: Lower-case extensions this loader claims, including the leading dot.
    extensions: tuple[str, ...] = ()
    #: Short identifier recorded in ``Document.metadata["loader"]``.
    name: str = "base"

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    @abstractmethod
    async def load(
        self,
        path: Path,
        *,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Read *path* and return a populated :class:`Document`.

        Implementations must raise :class:`LoaderError` (not the
        underlying library's exception) so the ingest pipeline can report
        a clean, user-facing message.
        """

    # ── shared helpers ───────────────────────────────────────────────

    @staticmethod
    async def _read_bytes(path: Path) -> bytes:
        """Read a file off the event loop."""
        try:
            return await asyncio.to_thread(path.read_bytes)
        except OSError as exc:
            raise LoaderError(f"Could not read {path.name}: {exc}") from exc

    @staticmethod
    def decode_text(raw: bytes, path: Path) -> str:
        """Decode bytes to text, tolerating non-UTF-8 files.

        Uploaded documents routinely arrive as cp1252 or latin-1. Failing
        the whole ingest over one smart quote is not acceptable, so the
        final fallback replaces undecodable bytes rather than raising.
        """
        for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                return raw.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        logger.warning("%s: falling back to lossy UTF-8 decode", path.name)
        return raw.decode("utf-8", errors="replace")

    def build_document(
        self,
        *,
        path: Path,
        content: str,
        title: str | None,
        metadata: dict[str, Any] | None,
        extra: dict[str, Any] | None = None,
        doc_type: str = "document",
    ) -> Document:
        """Assemble the Document, merging caller and loader metadata."""
        merged: dict[str, Any] = {
            "loader": self.name,
            "filename": path.name,
            "extension": path.suffix.lower(),
        }
        if extra:
            merged.update(extra)
        if metadata:
            merged.update(metadata)

        return Document(
            title=(title or "").strip() or path.stem,
            content=content,
            source=path.name,
            metadata=merged,
            doc_type=doc_type,
        )
