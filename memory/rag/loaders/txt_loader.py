"""Plain-text loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import Document
from .base import DocumentLoader, LoaderError


class TxtLoader(DocumentLoader):
    """Loads ``.txt``/``.log``/``.csv`` style files as one flat document."""

    extensions = (".txt", ".text", ".log", ".csv")
    name = "txt"

    async def load(
        self,
        path: Path,
        *,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        raw = await self._read_bytes(path)
        content = self.decode_text(raw, path)
        if not content.strip():
            raise LoaderError(f"{path.name} contains no readable text")

        return self.build_document(
            path=path,
            content=content,
            title=title,
            metadata=metadata,
            extra={"line_count": content.count("\n") + 1},
        )
