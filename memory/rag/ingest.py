"""Upload / ingest pipeline.

Implements the file path of the architecture::

    Upload API -> Temporary Storage -> Document Loader -> Text Extraction
               -> Chunking -> Embedding Generation -> Vector Database

This module is deliberately **independent of the conversation
pipeline**. It does not import the EventBus, the Conversation Manager,
the coordinator or any event type, and nothing in the conversation path
imports it. The only thing the two share is the vector database, reached
through :class:`~memory.rag.manager.RAGManager` — which is exactly the
coupling the design calls for and no more.

Chunking and embedding are not re-implemented here; they belong to the
RAG Manager and are invoked through it.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import RAGConfig
from .loaders import LoaderError, LoaderRegistry, UnsupportedFormatError
from .manager import RAGManager
from .models import SOURCE_UPLOAD, new_id

logger = logging.getLogger("kancha.memory.rag.ingest")

# Anything outside this set is stripped from an uploaded filename before
# it touches the filesystem.
_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._ -]")
_MAX_STEM = 80


def sanitize_filename(filename: str) -> str:
    """Reduce an untrusted upload name to a safe, flat filename.

    Strips directory components (defeating ``../`` traversal), removes
    characters that are awkward or dangerous on any filesystem, and caps
    the length. Never returns an empty string.
    """
    name = Path(filename or "").name  # drops any path component
    name = _UNSAFE_CHARS_RE.sub("_", name).strip(" .")
    if not name:
        return "upload"
    stem, dot, suffix = name.rpartition(".")
    if not dot:
        return name[:_MAX_STEM]
    return f"{stem[:_MAX_STEM] or 'upload'}.{suffix.lower()}"


@dataclass(slots=True)
class IngestReport:
    """Result of ingesting one file — the API response shape."""

    success: bool
    filename: str
    message: str
    document_id: str = ""
    title: str = ""
    chunks_indexed: int = 0
    chunks_skipped: int = 0
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "filename": self.filename,
            "message": self.message,
            "document_id": self.document_id,
            "title": self.title,
            "chunks_indexed": self.chunks_indexed,
            "chunks_skipped": self.chunks_skipped,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class IngestService:
    """Drives uploaded bytes through loader -> chunker -> embedder -> store."""

    def __init__(
        self,
        config: RAGConfig,
        rag_manager: RAGManager,
        registry: LoaderRegistry | None = None,
    ) -> None:
        self._config = config
        self._rag = rag_manager
        self._registry = registry or LoaderRegistry(
            allowed_extensions=config.supported_extensions
        )

    # ── introspection ────────────────────────────────────────────────

    def supported_extensions(self) -> tuple[str, ...]:
        """Extensions the upload endpoint currently accepts."""
        return self._registry.available_extensions()

    @property
    def max_upload_bytes(self) -> int:
        return self._config.max_upload_bytes

    # ── validation ───────────────────────────────────────────────────

    def _validate(self, filename: str, size: int) -> str:
        """Return an error message, or ``""`` if the upload is acceptable."""
        if size <= 0:
            return "The uploaded file is empty."
        if size > self._config.max_upload_bytes:
            limit_mb = self._config.max_upload_bytes / (1024 * 1024)
            return (
                f"File is too large ({size / (1024 * 1024):.1f} MB). "
                f"The limit is {limit_mb:.0f} MB."
            )
        if not self._registry.supports(filename):
            return (
                f"Unsupported file type. Supported formats: "
                f"{', '.join(self.supported_extensions())}"
            )
        return ""

    # ── temporary storage ────────────────────────────────────────────

    async def _stage(self, filename: str, data: bytes) -> Path:
        """Write *data* to the staging directory and return its path.

        Staged under a unique id so two concurrent uploads of the same
        filename cannot overwrite one another.
        """
        staging_dir = self._config.upload_dir
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged = staging_dir / f"{new_id('up')}_{sanitize_filename(filename)}"
        await asyncio.to_thread(staged.write_bytes, data)
        logger.debug("Staged upload %s (%d bytes)", staged.name, len(data))
        return staged

    @staticmethod
    async def _discard(path: Path) -> None:
        """Remove a staged file. Never raises — cleanup must not mask errors."""
        try:
            await asyncio.to_thread(path.unlink, True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not remove staged upload %s: %s", path, exc)

    # ── public API ───────────────────────────────────────────────────

    async def ingest_upload(
        self,
        filename: str,
        data: bytes,
        *,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestReport:
        """Ingest raw uploaded bytes. The main entry point for the API.

        Always returns an :class:`IngestReport`; it never raises, so the
        HTTP layer stays a thin translation of report -> status code.
        """
        display_name = sanitize_filename(filename)

        problem = self._validate(display_name, len(data))
        if problem:
            logger.info("Rejected upload %s: %s", display_name, problem)
            return IngestReport(
                success=False, filename=display_name, message=problem, error=problem
            )

        staged = await self._stage(display_name, data)
        try:
            return await self._ingest_staged(
                staged,
                display_name=display_name,
                title=title,
                metadata=metadata,
                size=len(data),
            )
        finally:
            await self._discard(staged)

    async def ingest_path(
        self,
        path: Path | str,
        *,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestReport:
        """Ingest a file already on disk, without staging a copy.

        Used by scripts and bulk back-fills. The original file is never
        modified or removed.
        """
        source = Path(path)
        display_name = source.name

        if not source.is_file():
            message = f"{display_name} does not exist or is not a file."
            return IngestReport(
                success=False, filename=display_name, message=message, error=message
            )

        try:
            size = source.stat().st_size
        except OSError as exc:
            message = f"Could not read {display_name}: {exc}"
            return IngestReport(
                success=False, filename=display_name, message=message, error=message
            )

        problem = self._validate(display_name, size)
        if problem:
            return IngestReport(
                success=False, filename=display_name, message=problem, error=problem
            )

        return await self._ingest_staged(
            source,
            display_name=display_name,
            title=title,
            metadata=metadata,
            size=size,
        )

    # ── core ─────────────────────────────────────────────────────────

    async def _ingest_staged(
        self,
        path: Path,
        *,
        display_name: str,
        title: str | None,
        metadata: dict[str, Any] | None,
        size: int,
    ) -> IngestReport:
        """Loader -> Document -> RAGManager. Shared by both entry points."""
        try:
            loader = self._registry.get(display_name)
        except UnsupportedFormatError as exc:
            return IngestReport(
                success=False, filename=display_name, message=str(exc), error=str(exc)
            )

        merged_metadata: dict[str, Any] = {
            "origin": SOURCE_UPLOAD,
            "original_filename": display_name,
            "size_bytes": size,
        }
        if metadata:
            merged_metadata.update(metadata)

        # ── Text extraction ──────────────────────────────────────────
        try:
            document = await loader.load(
                path, title=title, metadata=merged_metadata
            )
        except LoaderError as exc:
            logger.warning("Loader failed for %s: %s", display_name, exc)
            return IngestReport(
                success=False, filename=display_name, message=str(exc), error=str(exc)
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected loader failure for %s", display_name)
            message = f"Could not read {display_name}: {exc}"
            return IngestReport(
                success=False, filename=display_name, message=message, error=str(exc)
            )

        # The uploaded file's real name is the useful source, not the
        # randomised staging path.
        document.source = display_name

        logger.info(
            "Extracted %d chars from %s via %s loader",
            document.char_count,
            display_name,
            loader.name,
        )

        # ── Chunking + embedding + persistence (all inside the manager) ──
        try:
            result = await self._rag.index_document(document)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Indexing failed for %s", display_name)
            message = f"Could not index {display_name}: {exc}"
            return IngestReport(
                success=False,
                filename=display_name,
                message=message,
                error=str(exc),
                title=document.title,
            )

        if not result.indexed:
            reason = result.skipped_reason or "nothing new to index"
            return IngestReport(
                success=False,
                filename=display_name,
                message=f"{display_name} was not indexed: {reason}",
                document_id=result.document_id,
                title=result.title,
                chunks_skipped=result.chunks_skipped,
                error=reason,
            )

        return IngestReport(
            success=True,
            filename=display_name,
            message=(
                f"Indexed {display_name} as {result.chunks_indexed} chunk(s)"
                + (
                    f" ({result.chunks_skipped} duplicate(s) skipped)"
                    if result.chunks_skipped
                    else ""
                )
            ),
            document_id=result.document_id,
            title=result.title,
            chunks_indexed=result.chunks_indexed,
            chunks_skipped=result.chunks_skipped,
            metadata={
                "loader": loader.name,
                "char_count": document.char_count,
                "doc_type": document.doc_type,
            },
        )
