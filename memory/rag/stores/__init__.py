"""Vector store implementations and their factory.

Adding a backend: implement
:class:`~memory.rag.stores.base.VectorStore`, add a branch to
:func:`build_vector_store`, and allow its name in
:meth:`memory.rag.config.RAGConfig.from_env`.
"""

from __future__ import annotations

import logging

from ..config import RAGConfig
from .base import VectorStore
from .chroma_store import ChromaVectorStore
from .sqlite_store import SQLiteVectorStore

logger = logging.getLogger("kancha.memory.rag.stores")

__all__ = [
    "ChromaVectorStore",
    "SQLiteVectorStore",
    "VectorStore",
    "build_vector_store",
]


def build_vector_store(config: RAGConfig) -> VectorStore:
    """Instantiate the configured vector store (not yet initialized)."""
    if config.vector_store == "chroma":
        return ChromaVectorStore(
            persist_dir=config.chroma_dir,
            collection_name=config.collection_name,
            dimensions=config.embedding_dimensions,
        )
    return SQLiteVectorStore(
        db_path=config.store_path,
        dimensions=config.embedding_dimensions,
    )
