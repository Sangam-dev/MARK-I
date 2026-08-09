"""ChromaDB vector store — optional alternative backend.

Enable with ``KANCHA_RAG_VECTOR_STORE=chroma`` after installing the
dependency::

    uv add chromadb

Chroma is *not* the default because it pulls a large dependency tree for
a corpus size where brute-force cosine (see :mod:`.sqlite_store`) is both
faster and simpler. It becomes the right choice once the index grows past
roughly a hundred thousand chunks and approximate search starts paying
for itself.

Embeddings are always supplied by this project's own
:class:`~memory.rag.embeddings.base.EmbeddingProvider` — Chroma's
built-in embedding functions are explicitly disabled so switching the
*store* never silently switches the *embedding model* underneath the
index.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..models import Document, VectorMatch, VectorRecord, utc_now
from .base import VectorStore

logger = logging.getLogger("kancha.memory.rag.stores.chroma")


class ChromaVectorStore(VectorStore):
    """Persistent Chroma collection driven by externally-supplied vectors."""

    def __init__(self, persist_dir: Path, collection_name: str, dimensions: int) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._dimensions = dimensions
        self._client: Any = None
        self._collection: Any = None

    @property
    def name(self) -> str:
        return "chroma"

    # ── lifecycle ────────────────────────────────────────────────────

    async def initialize(self) -> None:
        if self._collection is not None:
            return
        try:
            import chromadb  # noqa: PLC0415
            from chromadb.config import Settings  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "KANCHA_RAG_VECTOR_STORE=chroma requires chromadb. "
                "Install it with: uv add chromadb  (or use the default "
                "KANCHA_RAG_VECTOR_STORE=sqlite, which needs nothing extra)."
            ) from exc

        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            # Cosine matches the L2-normalised vectors every provider emits.
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        logger.info(
            "Chroma vector store ready at %s (collection=%s, dim=%d)",
            self._persist_dir,
            self._collection_name,
            self._dimensions,
        )

    async def close(self) -> None:
        self._collection = None
        self._client = None

    def _require(self) -> Any:
        if self._collection is None:
            raise RuntimeError("ChromaVectorStore.initialize() was not awaited")
        return self._collection

    # ── metadata helpers ─────────────────────────────────────────────
    #
    # Chroma metadata values must be scalars, so nested dicts are stored
    # as a JSON string under a single key and rehydrated on read.

    @staticmethod
    def _flatten(record: VectorRecord, document: Document) -> dict[str, Any]:
        return {
            "document_id": record.document_id,
            "content_hash": record.content_hash,
            "title": document.title,
            "source": document.source,
            "doc_type": document.doc_type,
            "created_at": document.created_at or utc_now(),
            "extra": json.dumps(record.metadata or {}),
        }

    @staticmethod
    def _rehydrate(flat: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(flat or {})
        raw_extra = metadata.pop("extra", "{}")
        try:
            metadata.update(json.loads(raw_extra))
        except (TypeError, ValueError):
            pass
        return metadata

    # ── writes ───────────────────────────────────────────────────────

    async def add_documents(
        self, document: Document, records: list[VectorRecord]
    ) -> int:
        if not records:
            return 0
        collection = self._require()
        collection.upsert(
            ids=[r.chunk_id for r in records],
            embeddings=[list(r.embedding) for r in records],
            documents=[r.content for r in records],
            metadatas=[self._flatten(r, document) for r in records],
        )
        logger.info(
            "Indexed %d chunk(s) for document %s (%s)",
            len(records),
            document.id,
            document.title,
        )
        return len(records)

    async def update_document(
        self, document: Document, records: list[VectorRecord]
    ) -> int:
        await self.delete_document(document.id)
        return await self.add_documents(document, records)

    async def delete_document(self, document_id: str) -> int:
        collection = self._require()
        existing = collection.get(where={"document_id": document_id})
        ids = existing.get("ids", []) if existing else []
        if ids:
            collection.delete(ids=ids)
        logger.info("Deleted document %s (%d chunk(s))", document_id, len(ids))
        return len(ids)

    # ── reads ────────────────────────────────────────────────────────

    async def search(
        self,
        embedding: list[float],
        top_k: int,
        threshold: float = 0.0,
        doc_types: tuple[str, ...] | None = None,
    ) -> list[VectorMatch]:
        collection = self._require()
        if collection.count() == 0:
            logger.debug("Vector search on an empty index — returning no results")
            return []

        where = {"doc_type": {"$in": list(doc_types)}} if doc_types else None
        result = collection.query(
            query_embeddings=[list(embedding)],
            n_results=max(1, top_k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        matches: list[VectorMatch] = []
        for chunk_id, content, flat_meta, distance in zip(
            ids, documents, metadatas, distances
        ):
            # Chroma's cosine "distance" is 1 - similarity.
            score = 1.0 - float(distance)
            if score < threshold:
                continue
            metadata = self._rehydrate(flat_meta)
            matches.append(
                VectorMatch(
                    chunk_id=chunk_id,
                    document_id=str(metadata.get("document_id", "")),
                    content=content or "",
                    score=score,
                    metadata=metadata,
                )
            )
        return matches

    async def list_documents(self) -> list[dict[str, Any]]:
        collection = self._require()
        everything = collection.get(include=["metadatas"])
        grouped: dict[str, dict[str, Any]] = {}
        for flat_meta in everything.get("metadatas") or []:
            metadata = self._rehydrate(flat_meta)
            document_id = str(metadata.get("document_id", ""))
            if not document_id:
                continue
            entry = grouped.setdefault(
                document_id,
                {
                    "id": document_id,
                    "title": metadata.get("title", "Untitled"),
                    "source": metadata.get("source", ""),
                    "doc_type": metadata.get("doc_type", "document"),
                    "created_at": metadata.get("created_at", ""),
                    "updated_at": metadata.get("created_at", ""),
                    "chunk_count": 0,
                    "metadata": {},
                },
            )
            entry["chunk_count"] += 1
        return sorted(
            grouped.values(), key=lambda d: d.get("updated_at", ""), reverse=True
        )

    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        for document in await self.list_documents():
            if document["id"] == document_id:
                return document
        return None

    async def existing_hashes(self, hashes: list[str]) -> set[str]:
        if not hashes:
            return set()
        collection = self._require()
        found = collection.get(where={"content_hash": {"$in": list(hashes)}})
        return {
            str(meta.get("content_hash"))
            for meta in (found.get("metadatas") or [])
            if meta and meta.get("content_hash")
        }

    async def count_chunks(self) -> int:
        return int(self._require().count())
