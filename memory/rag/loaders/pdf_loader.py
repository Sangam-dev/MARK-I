"""PDF loader.

Extracts text per page and hands the page list to the chunker via
``metadata["pages"]`` so every resulting chunk can report the page it came
from — which is what makes "summarize the uploaded networking PDF" able
to cite a location rather than a vague blob.

Requires ``pypdf``. It is a small pure-Python dependency; if it is
missing the loader raises a clear install message instead of failing
obscurely.

Note: scanned/image-only PDFs contain no text layer. This loader detects
that case and says so, rather than silently indexing an empty document.
OCR is deliberately out of scope.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from ..models import Document
from .base import DocumentLoader, LoaderError

logger = logging.getLogger("kancha.memory.rag.loaders.pdf")


class PdfLoader(DocumentLoader):
    """Loads ``.pdf`` files page by page."""

    extensions = (".pdf",)
    name = "pdf"

    def _extract_sync(self, path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        try:
            from pypdf import PdfReader  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise LoaderError(
                "PDF support requires pypdf. Install it with: uv add pypdf"
            ) from exc

        try:
            reader = PdfReader(str(path))
        except Exception as exc:  # noqa: BLE001
            raise LoaderError(f"Could not open {path.name} as a PDF: {exc}") from exc

        if getattr(reader, "is_encrypted", False):
            # An empty user password is common for "protected" PDFs and
            # costs nothing to try before giving up.
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise LoaderError(
                    f"{path.name} is password protected and cannot be indexed"
                ) from exc

        pages: list[dict[str, Any]] = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s page %d: extraction failed (%s)", path.name, number, exc)
                text = ""
            if text.strip():
                pages.append({"page": number, "text": text})

        info: dict[str, Any] = {"page_count": len(reader.pages)}
        try:
            raw_meta = reader.metadata or {}
            for source_key, target_key in (
                ("/Title", "pdf_title"),
                ("/Author", "pdf_author"),
                ("/Subject", "pdf_subject"),
            ):
                value = raw_meta.get(source_key)
                if value:
                    info[target_key] = str(value)
        except Exception:  # noqa: BLE001
            pass

        return pages, info

    async def load(
        self,
        path: Path,
        *,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        pages, info = await asyncio.to_thread(self._extract_sync, path)

        if not pages:
            raise LoaderError(
                f"{path.name} has no extractable text — it is most likely a "
                "scanned document, which needs OCR before it can be indexed."
            )

        content = "\n\n".join(page["text"] for page in pages)
        extra: dict[str, Any] = {"pages": pages, "text_pages": len(pages)}
        extra.update(info)

        return self.build_document(
            path=path,
            content=content,
            title=title or info.get("pdf_title") or path.stem,
            metadata=metadata,
            extra=extra,
        )
