"""Modular document loaders.

Each loader converts one file format into the common
:class:`~memory.rag.models.Document` shape (title, content, source,
metadata). See :mod:`memory.rag.loaders.base` for the contract and
:mod:`memory.rag.loaders.registry` for extension dispatch.
"""

from __future__ import annotations

from .base import DocumentLoader, LoaderError, UnsupportedFormatError
from .docx_loader import DocxLoader
from .md_loader import MarkdownLoader
from .pdf_loader import PdfLoader
from .registry import DEFAULT_LOADERS, LoaderRegistry
from .txt_loader import TxtLoader

__all__ = [
    "DEFAULT_LOADERS",
    "DocumentLoader",
    "DocxLoader",
    "LoaderError",
    "LoaderRegistry",
    "MarkdownLoader",
    "PdfLoader",
    "TxtLoader",
    "UnsupportedFormatError",
]
