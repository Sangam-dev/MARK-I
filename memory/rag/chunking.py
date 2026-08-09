"""Document chunking.

Whole documents are never embedded: a single vector cannot represent a
30-page PDF, and retrieving one would blow the prompt budget. Documents
are split into overlapping windows that each carry enough provenance
(document id, source, title, page, chunk index) to be cited back to the
user.

Splitting is *structure aware*. It tries progressively weaker separators
— paragraph break, line break, sentence end, word boundary — and only
falls back to a hard character cut when a single word exceeds the chunk
size. That keeps chunks aligned with the author's own semantic units
instead of slicing mid-sentence.

Overlap exists so a fact that straddles a boundary survives in at least
one chunk intact.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .config import RAGConfig
from .models import Chunk, Document

logger = logging.getLogger("kancha.memory.rag.chunking")

# Tried in order, strongest structural signal first.
_SEPARATORS = ("\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ")

_EXCESS_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+\n")


def normalise_text(text: str) -> str:
    """Collapse noisy whitespace without destroying paragraph structure.

    PDF extraction in particular produces ragged trailing spaces and long
    runs of blank lines; both waste tokens and weaken embeddings.
    """
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _TRAILING_SPACE_RE.sub("\n", cleaned)
    cleaned = _EXCESS_BLANK_RE.sub("\n\n", cleaned)
    return cleaned.strip()


class Chunker:
    """Splits :class:`Document` objects into embeddable :class:`Chunk` objects."""

    def __init__(self, config: RAGConfig) -> None:
        self._size = config.chunk_size
        self._overlap = config.chunk_overlap
        self._min_chars = config.min_chunk_chars

    # ── public API ───────────────────────────────────────────────────

    def split(self, document: Document) -> list[Chunk]:
        """Split *document* into chunks, preserving page numbers if present.

        Loaders that understand pagination (see
        :mod:`memory.rag.loaders.pdf_loader`) put a ``pages`` list in
        ``document.metadata``; when present each page is chunked
        independently so no chunk ever spans two pages and every chunk
        can report the page it came from.
        """
        pages = document.metadata.get("pages")
        chunks: list[Chunk] = []

        if isinstance(pages, list) and pages:
            index = 0
            for page in pages:
                page_number = page.get("page")
                page_text = normalise_text(str(page.get("text", "")))
                for body in self._split_text(page_text):
                    chunks.append(
                        self._make_chunk(document, body, index, page=page_number)
                    )
                    index += 1
        else:
            for index, body in enumerate(self._split_text(normalise_text(document.content))):
                chunks.append(self._make_chunk(document, body, index))

        if not chunks:
            logger.info(
                "Document %s (%s) produced no chunks — content too short (%d chars, min %d)",
                document.id,
                document.title,
                len(document.content or ""),
                self._min_chars,
            )
        else:
            logger.debug(
                "Split %s into %d chunk(s) (size=%d overlap=%d)",
                document.title,
                len(chunks),
                self._size,
                self._overlap,
            )
        return chunks

    # ── internals ────────────────────────────────────────────────────

    def _make_chunk(
        self,
        document: Document,
        content: str,
        index: int,
        page: Any = None,
    ) -> Chunk:
        metadata: dict[str, Any] = {
            "document_id": document.id,
            "title": document.title,
            "source": document.source,
            "doc_type": document.doc_type,
            "chunk_index": index,
        }
        if page is not None:
            metadata["page"] = page
        # Carry loader-supplied metadata through, minus the bulky page
        # payload which has already been consumed by the splitter.
        for key, value in (document.metadata or {}).items():
            if key != "pages" and key not in metadata:
                metadata[key] = value
        return Chunk(
            document_id=document.id,
            content=content,
            chunk_index=index,
            metadata=metadata,
        )

    def _split_text(self, text: str) -> list[str]:
        """Window *text* into overlapping pieces on the best available boundary."""
        if not text:
            return []

        if len(text) <= self._size:
            # The whole document fits in one chunk, so there is nothing to
            # split and ``min_chunk_chars`` does not apply.
            #
            # That floor exists to discard *fragments* thrown off by
            # splitting — page headers, dangling clauses. Applying it here
            # instead silently discarded any document shorter than it,
            # which is exactly wrong for the short, deliberate memories the
            # model writes to rag.txt ("Batching embeddings cut indexing
            # time in half." is 46 characters and matters).
            return [text]

        pieces: list[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + self._size, length)

            if end < length:
                split_at = self._best_boundary(text, start, end)
                if split_at > start:
                    end = split_at

            piece = text[start:end].strip()
            if len(piece) >= self._min_chars:
                pieces.append(piece)

            if end >= length:
                break

            # Step forward, minus the overlap. max() guarantees progress
            # even if a pathological boundary lands at `start`.
            start = max(end - self._overlap, start + 1)

        return pieces

    def _best_boundary(self, text: str, start: int, end: int) -> int:
        """Find the latest natural boundary in ``text[start:end]``.

        Boundaries closer than halfway through the window are rejected —
        accepting them would produce a chunk far smaller than configured
        and inflate the chunk count for no retrieval benefit.
        """
        minimum = start + self._size // 2
        for separator in _SEPARATORS:
            position = text.rfind(separator, minimum, end)
            if position > start:
                return position + len(separator)
        return end
