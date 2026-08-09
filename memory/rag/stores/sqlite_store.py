"""SQLite + numpy vector store — the default backend.

Chosen as the default because it needs **no new dependency**: the project
already uses ``aiosqlite`` for structured memory and ``numpy`` for audio.
It sits next to ``structured.db`` in ``memory/data/rag/vectors.db``.

Similarity is exact (brute-force cosine over the full index), not
approximate. For a personal assistant — thousands to low tens of
thousands of chunks — that is both faster and more accurate than an ANN
index, and it removes a whole class of index-corruption failure modes.
An in-memory matrix cache keeps repeat queries at numpy speed; it is
invalidated on every write.

If the corpus ever outgrows this (say >100k chunks), switch to the chroma
backend with ``KANCHA_RAG_VECTOR_STORE=chroma``. Nothing outside this
package changes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np

from ..models import Document, VectorMatch, VectorRecord, utc_now
from .base import VectorStore

logger = logging.getLogger("kancha.memory.rag.stores.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_documents (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT '',
    doc_type    TEXT NOT NULL DEFAULT 'document',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS rag_chunks (
    id           TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL,
    content      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding    BLOB NOT NULL,
    dimensions   INTEGER NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_document
    ON rag_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_hash
    ON rag_chunks(content_hash);
"""


class SQLiteVectorStore(VectorStore):
    """Brute-force cosine similarity over embeddings stored as BLOBs."""

    def __init__(self, db_path: Path, dimensions: int) -> None:
        self._db_path = db_path
        self._dimensions = dimensions
        self._conn: aiosqlite.Connection | None = None
        # Cached search matrix: (matrix, rows). Invalidated on write.
        self._cache: tuple[np.ndarray, list[dict[str, Any]]] | None = None

    @property
    def name(self) -> str:
        return "sqlite"

    # ── lifecycle ────────────────────────────────────────────────────

    async def initialize(self) -> None:
        if self._conn is not None:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()
        logger.info(
            "SQLite vector store ready at %s (dim=%d)", self._db_path, self._dimensions
        )

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
        self._cache = None

    def _db(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteVectorStore.initialize() was not awaited")
        return self._conn

    def _invalidate(self) -> None:
        self._cache = None

    async def warm(self) -> None:
        """Build the similarity matrix now instead of on the first query."""
        await self._load_matrix()

    # ── serialisation ────────────────────────────────────────────────

    @staticmethod
    def _pack(embedding: list[float]) -> bytes:
        return np.asarray(embedding, dtype=np.float32).tobytes()

    @staticmethod
    def _unpack(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    # ── writes ───────────────────────────────────────────────────────

    async def add_documents(
        self, document: Document, records: list[VectorRecord]
    ) -> int:
        if not records:
            return 0
        db = self._db()
        now = utc_now()

        await db.execute(
            """
            INSERT INTO rag_documents
                (id, title, source, doc_type, created_at, updated_at, chunk_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title       = excluded.title,
                source      = excluded.source,
                doc_type    = excluded.doc_type,
                updated_at  = excluded.updated_at,
                chunk_count = rag_documents.chunk_count + excluded.chunk_count,
                metadata    = excluded.metadata
            """,
            (
                document.id,
                document.title,
                document.source,
                document.doc_type,
                document.created_at or now,
                now,
                len(records),
                json.dumps(document.metadata or {}),
            ),
        )

        await db.executemany(
            """
            INSERT INTO rag_chunks
                (id, document_id, content, content_hash, embedding, dimensions, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO NOTHING
            """,
            [
                (
                    record.chunk_id,
                    record.document_id,
                    record.content,
                    record.content_hash,
                    self._pack(record.embedding),
                    len(record.embedding),
                    json.dumps(record.metadata or {}),
                    now,
                )
                for record in records
            ],
        )
        await db.commit()
        self._invalidate()
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
        """Replace every chunk of *document*, then re-add.

        Implemented as delete-then-insert inside one transaction so a
        crash mid-update cannot leave a half-reindexed document.
        """
        db = self._db()
        await db.execute("DELETE FROM rag_chunks WHERE document_id = ?", (document.id,))
        await db.execute(
            "UPDATE rag_documents SET chunk_count = 0 WHERE id = ?", (document.id,)
        )
        await db.commit()
        self._invalidate()
        return await self.add_documents(document, records)

    async def delete_document(self, document_id: str) -> int:
        db = self._db()
        cursor = await db.execute(
            "DELETE FROM rag_chunks WHERE document_id = ?", (document_id,)
        )
        removed = cursor.rowcount or 0
        await db.execute("DELETE FROM rag_documents WHERE id = ?", (document_id,))
        await db.commit()
        self._invalidate()
        logger.info("Deleted document %s (%d chunk(s))", document_id, removed)
        return removed

    # ── reads ────────────────────────────────────────────────────────

    async def _load_matrix(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Return (embeddings matrix, row metadata), using the cache."""
        if self._cache is not None:
            return self._cache

        db = self._db()
        async with db.execute(
            """
            SELECT c.id, c.document_id, c.content, c.embedding, c.dimensions,
                   c.metadata, d.title, d.source, d.doc_type
            FROM rag_chunks c
            LEFT JOIN rag_documents d ON d.id = c.document_id
            """
        ) as cursor:
            raw_rows = await cursor.fetchall()

        vectors: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []
        for row in raw_rows:
            vector = self._unpack(row["embedding"])
            if vector.size != self._dimensions:
                # A provider/dimension change invalidates old vectors.
                # Skip rather than crash; re-indexing fixes it.
                logger.warning(
                    "Skipping chunk %s: stored dim=%d, expected %d "
                    "(re-index this document after an embedding model change)",
                    row["id"],
                    vector.size,
                    self._dimensions,
                )
                continue
            metadata = json.loads(row["metadata"] or "{}")
            metadata.setdefault("title", row["title"] or "Untitled")
            metadata.setdefault("source", row["source"] or "")
            metadata.setdefault("doc_type", row["doc_type"] or "document")
            vectors.append(vector)
            rows.append(
                {
                    "chunk_id": row["id"],
                    "document_id": row["document_id"],
                    "content": row["content"],
                    "metadata": metadata,
                }
            )

        matrix = (
            np.vstack(vectors)
            if vectors
            else np.zeros((0, self._dimensions), dtype=np.float32)
        )
        self._cache = (matrix, rows)
        return self._cache

    async def search(
        self,
        embedding: list[float],
        top_k: int,
        threshold: float = 0.0,
        doc_types: tuple[str, ...] | None = None,
    ) -> list[VectorMatch]:
        matrix, rows = await self._load_matrix()
        if matrix.shape[0] == 0:
            logger.debug("Vector search on an empty index — returning no results")
            return []

        query = np.asarray(embedding, dtype=np.float32)
        if query.size != matrix.shape[1]:
            logger.warning(
                "Query dim %d != index dim %d — returning no results",
                query.size,
                matrix.shape[1],
            )
            return []

        # Providers return L2-normalised vectors, so the dot product is
        # already the cosine similarity. Normalise the query defensively
        # anyway: it costs one sqrt and protects against a provider that
        # quietly breaks the contract.
        norm = float(np.linalg.norm(query))
        if norm > 1e-12:
            query = query / norm

        scores = matrix @ query

        candidate_indices = np.argsort(-scores)
        matches: list[VectorMatch] = []
        for index in candidate_indices:
            score = float(scores[index])
            if score < threshold:
                break  # sorted descending — everything after is worse
            row = rows[index]
            if doc_types and row["metadata"].get("doc_type") not in doc_types:
                continue
            matches.append(
                VectorMatch(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    content=row["content"],
                    score=score,
                    metadata=row["metadata"],
                )
            )
            if len(matches) >= top_k:
                break
        return matches

    async def list_documents(self) -> list[dict[str, Any]]:
        db = self._db()
        async with db.execute(
            """
            SELECT id, title, source, doc_type, created_at, updated_at,
                   chunk_count, metadata
            FROM rag_documents
            ORDER BY updated_at DESC
            """
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "source": row["source"],
                "doc_type": row["doc_type"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "chunk_count": row["chunk_count"],
                "metadata": json.loads(row["metadata"] or "{}"),
            }
            for row in rows
        ]

    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        for document in await self.list_documents():
            if document["id"] == document_id:
                return document
        return None

    async def existing_hashes(self, hashes: list[str]) -> set[str]:
        if not hashes:
            return set()
        db = self._db()
        placeholders = ",".join("?" for _ in hashes)
        async with db.execute(
            f"SELECT DISTINCT content_hash FROM rag_chunks WHERE content_hash IN ({placeholders})",
            tuple(hashes),
        ) as cursor:
            rows = await cursor.fetchall()
        return {row["content_hash"] for row in rows}

    async def count_chunks(self) -> int:
        db = self._db()
        async with db.execute("SELECT COUNT(*) AS n FROM rag_chunks") as cursor:
            row = await cursor.fetchone()
        return int(row["n"]) if row else 0
