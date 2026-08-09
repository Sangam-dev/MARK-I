"""Retrieval-Augmented Generation subsystem — long-term semantic memory.

Division of labour with the rest of the memory layer:

* :class:`~memory.structured.StructuredMemory` (SQLite) holds **structured**
  information — facts, preferences, profile, settings, small key/value
  memories. Exact lookup, no embeddings.
* This package holds **semantic** information — experiences, research,
  technical documentation, project progress, learning summaries, design
  decisions, debugging solutions and uploaded documents. Approximate
  lookup by meaning.

Three independent pipelines share one vector database and nothing else:

* **Conversation** — :class:`~memory.rag.router.MemoryRouter` decides
  whether a turn needs context; :meth:`RAGManager.retrieve` fetches it;
  the Conversation Manager formats it into the prompt.
* **Upload** — :class:`~memory.rag.ingest.IngestService` takes a file
  through staging, loading, chunking, embedding and indexing. It imports
  nothing from the conversation layer.
* **Write-back** — :meth:`RAGManager.index_conversation_entries` persists
  the ``rag`` array of a model response.

Everything is reached through :class:`RAGManager`; no caller outside this
package touches the vector store, the embedder or the chunker directly.
"""

from __future__ import annotations

from .chunking import Chunker
from .config import RAGConfig
from .embeddings import EmbeddingError, EmbeddingProvider, build_embedder
from .ingest import IngestReport, IngestService
from .loaders import DocumentLoader, LoaderError, LoaderRegistry
from .manager import RAGManager
from .models import (
    Chunk,
    Document,
    IndexResult,
    RetrievedChunk,
    VectorMatch,
    VectorRecord,
)
from .router import MemoryRouter, RouteDecision, build_router
from .stores import VectorStore, build_vector_store
from .sync import RagFileSync, SyncReport, parse_rag_file

__all__ = [
    "Chunk",
    "Chunker",
    "Document",
    "DocumentLoader",
    "EmbeddingError",
    "EmbeddingProvider",
    "IndexResult",
    "IngestReport",
    "IngestService",
    "LoaderError",
    "LoaderRegistry",
    "MemoryRouter",
    "RAGConfig",
    "RAGManager",
    "RagFileSync",
    "RetrievedChunk",
    "RouteDecision",
    "SyncReport",
    "VectorMatch",
    "VectorRecord",
    "VectorStore",
    "build_embedder",
    "build_router",
    "build_vector_store",
    "parse_rag_file",
]
