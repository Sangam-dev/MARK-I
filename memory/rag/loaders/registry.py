"""Loader registry — maps a file extension to the loader that handles it.

The registry is the single place the ingest pipeline consults, so adding
a format never touches :mod:`memory.rag.ingest`. Which extensions are
actually *accepted* is still gated by
:attr:`memory.rag.config.RAGConfig.supported_extensions`, so an operator
can narrow the surface without deleting code.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .base import DocumentLoader, UnsupportedFormatError
from .docx_loader import DocxLoader
from .md_loader import MarkdownLoader
from .pdf_loader import PdfLoader
from .txt_loader import TxtLoader

logger = logging.getLogger("kancha.memory.rag.loaders.registry")

# Order matters only for ``available_extensions()`` output; lookup is by
# extension and therefore unambiguous.
DEFAULT_LOADERS: tuple[DocumentLoader, ...] = (
    PdfLoader(),
    TxtLoader(),
    MarkdownLoader(),
    DocxLoader(),
)


class LoaderRegistry:
    """Extension -> loader lookup."""

    def __init__(
        self,
        loaders: tuple[DocumentLoader, ...] = DEFAULT_LOADERS,
        allowed_extensions: tuple[str, ...] | None = None,
    ) -> None:
        self._loaders = loaders
        self._allowed = (
            tuple(e.lower() for e in allowed_extensions) if allowed_extensions else None
        )
        self._by_extension: dict[str, DocumentLoader] = {}
        for loader in loaders:
            for extension in loader.extensions:
                self._by_extension[extension.lower()] = loader

    def register(self, loader: DocumentLoader) -> None:
        """Add a loader at runtime (used by tests and plugins)."""
        for extension in loader.extensions:
            self._by_extension[extension.lower()] = loader
        self._loaders = (*self._loaders, loader)

    def supports(self, path: Path | str) -> bool:
        extension = Path(path).suffix.lower()
        if self._allowed is not None and extension not in self._allowed:
            return False
        return extension in self._by_extension

    def get(self, path: Path | str) -> DocumentLoader:
        """Return the loader for *path*, or raise :class:`UnsupportedFormatError`."""
        extension = Path(path).suffix.lower()

        if self._allowed is not None and extension not in self._allowed:
            raise UnsupportedFormatError(
                f"'{extension or 'no extension'}' is not an enabled format. "
                f"Enabled: {', '.join(self._allowed)}"
            )

        loader = self._by_extension.get(extension)
        if loader is None:
            raise UnsupportedFormatError(
                f"No loader for '{extension or 'no extension'}'. "
                f"Supported: {', '.join(self.available_extensions())}"
            )
        return loader

    def available_extensions(self) -> tuple[str, ...]:
        """Extensions that are both implemented and enabled by config."""
        extensions = sorted(self._by_extension)
        if self._allowed is not None:
            extensions = [e for e in extensions if e in self._allowed]
        return tuple(extensions)
