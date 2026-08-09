"""RAG Manager — the single entry point to semantic memory.

Owns exactly five things:

* embedding generation (delegated to an :class:`EmbeddingProvider`)
* indexing (chunk -> embed -> persist)
* retrieval (embed query -> search -> shape results)
* duplicate detection (exact hash + semantic similarity)
* document management (list / get / delete / stats)

Explicitly does **not** own prompt construction. :meth:`retrieve`
returns structured :class:`~memory.rag.models.RetrievedChunk` objects and
stops there; deciding how they appear in a prompt belongs to the
Conversation Manager. Keeping that line sharp is what allows the prompt
format to change without touching retrieval, and the vector backend to
change without touching the prompt.

Nothing above this class ever touches the vector store directly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any, Iterable

from .chunking import Chunker
from .config import RAGConfig
from .embeddings import EmbeddingError, EmbeddingProvider, HashingEmbedder, build_embedder
from .models import (
    SOURCE_CONVERSATION,
    Chunk,
    Document,
    IndexResult,
    RetrievedChunk,
    VectorRecord,
    content_hash,
)
from .stores import VectorStore, build_vector_store

logger = logging.getLogger("kancha.memory.rag.manager")


class RAGManager:
    """Coordinates embedder, chunker and vector store behind one API."""

    def __init__(
        self,
        config: RAGConfig,
        embedder: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
        chunker: Chunker | None = None,
    ) -> None:
        self._config = config
        # Injectable for tests; built from config otherwise.
        self._embedder = embedder or build_embedder(config)
        self._store = store or build_vector_store(config)
        self._chunker = chunker or Chunker(config)
        self._write_lock = asyncio.Lock()
        self._ready = False
        # Query-embedding LRU. A voice assistant asks the same things
        # repeatedly ("what did we decide about X"), and re-embedding an
        # identical string costs a network round-trip for a byte-identical
        # answer. Keyed on the normalised query, capped by config.
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        # Embeddings currently being computed, so a prefetch and the real
        # retrieval of the same text share one call instead of racing.
        self._inflight: dict[str, asyncio.Future[list[float]]] = {}
        self._prefetch_tasks: set[asyncio.Task] = set()

    # ── properties ───────────────────────────────────────────────────

    @property
    def config(self) -> RAGConfig:
        return self._config

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def embedder_name(self) -> str:
        return self._embedder.name

    @property
    def store_name(self) -> str:
        return self._store.name

    # ── lifecycle ────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Open the store and verify the embedder can actually embed.

        If the configured embedder fails its health check (no API key,
        Ollama not running, network down) the manager degrades to the
        offline :class:`HashingEmbedder` rather than leaving the whole
        subsystem dead. Retrieval quality drops to lexical overlap, which
        is a great deal better than every query erroring out.
        """
        if self._ready:
            return

        await self._store.initialize()

        if not await self._embedder.health_check():
            logger.warning(
                "Embedding provider %s is unavailable — falling back to the offline "
                "hashing embedder. Retrieval will be lexical, not semantic. Fix the "
                "provider and re-index to restore full quality.",
                self._embedder.name,
            )
            await self._embedder.close()
            self._embedder = HashingEmbedder(dimensions=self._config.embedding_dimensions)

        self._ready = True
        chunk_count = await self._store.count_chunks()
        logger.info(
            "RAG ready — embedder=%s store=%s chunks=%d (%s)",
            self._embedder.name,
            self._store.name,
            chunk_count,
            self._config.describe(),
        )

    async def warm(self) -> None:
        """Pull the index into memory so the first query pays no load cost.

        Without this the first retrieval of the session also pays for
        reading every embedding out of SQLite and building the similarity
        matrix — on a voice turn that shows up as a noticeable pause
        before the assistant speaks. Called once at boot from
        ``core.pipeline``; cheap and idempotent.
        """
        if not self._ready:
            return
        started = time.perf_counter()
        try:
            await self._store.warm()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Index warm-up skipped: %s", exc)
            return
        elapsed = (time.perf_counter() - started) * 1000
        logger.info(
            "RAG index warm (%d chunk(s) resident, %.0fms)",
            await self._store.count_chunks(),
            elapsed,
        )

    async def close(self) -> None:
        for task in list(self._prefetch_tasks):
            if not task.done():
                task.cancel()
        self._prefetch_tasks.clear()
        self._inflight.clear()
        await self._embedder.close()
        await self._store.close()
        self._query_cache.clear()
        self._ready = False

    def _require_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("RAGManager.initialize() was not awaited")

    # ── indexing ─────────────────────────────────────────────────────

    async def index_document(
        self,
        document: Document,
        *,
        replace: bool = False,
        semantic_dedupe: bool = False,
    ) -> IndexResult:
        """Chunk, embed and persist *document*.

        Parameters
        ----------
        replace:
            Replace an existing document with the same id instead of
            appending to it. Used by re-index / update flows.
        semantic_dedupe:
            Additionally drop chunks that are near-identical to something
            already indexed. Costs one vector search per chunk, so it is
            enabled for small conversation writes and off for bulk file
            uploads (which rely on the exact-hash pass instead).
        """
        self._require_ready()

        chunks = self._chunker.split(document)
        if not chunks:
            return IndexResult(
                document_id=document.id,
                title=document.title,
                chunks_indexed=0,
                chunks_skipped=0,
                skipped_reason="document produced no chunks (too short or empty)",
            )

        async with self._write_lock:
            kept, exact_duplicates = await self._filter_exact_duplicates(chunks)
            if not kept:
                logger.info(
                    "Document %s (%s) is entirely duplicate — nothing indexed",
                    document.id,
                    document.title,
                )
                return IndexResult(
                    document_id=document.id,
                    title=document.title,
                    chunks_indexed=0,
                    chunks_skipped=exact_duplicates,
                    skipped_reason="all chunks already indexed",
                )

            try:
                embeddings = await self._embedder.embed_documents(
                    [chunk.content for chunk in kept]
                )
            except EmbeddingError as exc:
                logger.error("Indexing %s failed: %s", document.title, exc)
                return IndexResult(
                    document_id=document.id,
                    title=document.title,
                    chunks_indexed=0,
                    chunks_skipped=len(chunks),
                    skipped_reason=f"embedding failed: {exc}",
                )

            records = [
                VectorRecord(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    content=chunk.content,
                    embedding=embedding,
                    metadata=chunk.metadata,
                    content_hash=chunk.hash,
                )
                for chunk, embedding in zip(kept, embeddings)
            ]

            semantic_duplicates = 0
            if semantic_dedupe:
                records, semantic_duplicates = await self._filter_semantic_duplicates(
                    records
                )
                if not records:
                    return IndexResult(
                        document_id=document.id,
                        title=document.title,
                        chunks_indexed=0,
                        chunks_skipped=exact_duplicates + semantic_duplicates,
                        skipped_reason="already known (semantically duplicate)",
                    )

            if replace:
                written = await self._store.update_document(document, records)
            else:
                written = await self._store.add_documents(document, records)

        return IndexResult(
            document_id=document.id,
            title=document.title,
            chunks_indexed=written,
            chunks_skipped=exact_duplicates + semantic_duplicates,
        )

    async def index_conversation_entries(
        self,
        entries: Iterable[dict[str, Any]],
        session_id: str = "default",
        *,
        semantic_dedupe: bool = True,
        origin: str = SOURCE_CONVERSATION,
    ) -> list[IndexResult]:
        """Index the ``rag`` array carried by a model response.

        Each entry is ``{"type": ..., "title": ..., "content": ...}``.
        Entries with no content are skipped silently — the model
        occasionally emits placeholder objects.

        ``semantic_dedupe`` defaults to on for live conversation, because
        the model tends to re-state the same insight across turns and the
        volume is low enough to afford one extra search per entry. The
        boot-time ``rag.txt`` replay passes ``False``: those entries were
        already de-duplicated when first written, and the hash pass alone
        keeps the backfill to a single query when nothing is new.
        """
        self._require_ready()
        results: list[IndexResult] = []

        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            content = str(entry.get("content", "")).strip()
            if not content:
                continue

            metadata: dict[str, Any] = {
                "origin": origin,
                "session_id": session_id,
            }
            # Preserve the original write time when replaying the audit
            # log, so recency ordering in the UI stays truthful.
            recorded_at = str(entry.get("timestamp", "")).strip()
            if recorded_at:
                metadata["recorded_at"] = recorded_at

            document = Document(
                title=str(entry.get("title", "")).strip() or "Untitled note",
                content=content,
                source=f"{SOURCE_CONVERSATION}:{session_id}",
                doc_type=str(entry.get("type", "")).strip() or "note",
                metadata=metadata,
            )
            try:
                result = await self.index_document(
                    document, semantic_dedupe=semantic_dedupe
                )
            except Exception as exc:  # noqa: BLE001
                # A memory write must never break the reply that produced it.
                logger.warning("Failed to index conversation entry: %s", exc)
                continue

            results.append(result)
            if result.indexed:
                logger.info(
                    "Indexed conversation memory '%s' (%s, %d chunk(s))",
                    document.title,
                    document.doc_type,
                    result.chunks_indexed,
                )
            else:
                logger.debug(
                    "Skipped conversation memory '%s': %s",
                    document.title,
                    result.skipped_reason,
                )
        return results

    # ── duplicate detection ──────────────────────────────────────────

    async def _filter_exact_duplicates(
        self, chunks: list[Chunk]
    ) -> tuple[list[Chunk], int]:
        """Drop chunks whose normalised content is already indexed.

        Cheap (one indexed SQL lookup) and runs *before* embedding, so
        re-uploading the same file costs no embedding quota at all.
        """
        hashes = [chunk.hash for chunk in chunks]
        try:
            known = await self._store.existing_hashes(hashes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Duplicate lookup failed (%s) — indexing everything", exc)
            return chunks, 0

        kept: list[Chunk] = []
        seen_in_batch: set[str] = set()
        skipped = 0
        for chunk in chunks:
            digest = chunk.hash
            # Guard against duplicates *within* the same document too —
            # repeated headers/footers are common in PDF extraction.
            if digest in known or digest in seen_in_batch:
                skipped += 1
                continue
            seen_in_batch.add(digest)
            kept.append(chunk)

        if skipped:
            logger.debug("Exact-duplicate filter dropped %d chunk(s)", skipped)
        return kept, skipped

    async def _filter_semantic_duplicates(
        self, records: list[VectorRecord]
    ) -> tuple[list[VectorRecord], int]:
        """Drop records that are near-identical to an existing chunk.

        Catches paraphrases that the hash pass cannot: "we fixed the leak
        by closing the cursor" vs "the leak was fixed by closing the
        cursor". Threshold is
        ``KANCHA_RAG_DEDUPE_THRESHOLD`` (default 0.95) — high on purpose,
        because discarding a genuinely new memory is worse than storing a
        near-duplicate.
        """
        threshold = self._config.dedupe_similarity_threshold
        kept: list[VectorRecord] = []
        skipped = 0

        for record in records:
            try:
                matches = await self._store.search(
                    embedding=record.embedding, top_k=1, threshold=threshold
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Semantic dedupe lookup failed (%s) — keeping record", exc)
                matches = []

            if matches:
                skipped += 1
                logger.debug(
                    "Semantic duplicate (score=%.3f) — skipping: %.60s",
                    matches[0].score,
                    record.content,
                )
                continue
            kept.append(record)

        return kept, skipped

    # ── retrieval ────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(query: str) -> str:
        return " ".join(query.split()).casefold()

    async def _embed_query_cached(self, query: str) -> list[float]:
        """Embed *query*, reusing a recent identical embedding if we have one.

        If a :meth:`prefetch` for the same text is already in flight this
        awaits that call rather than starting a second one — which is what
        makes prefetching actually save time instead of doubling the work.
        """
        if self._config.query_cache_size <= 0:
            return await self._embedder.embed_query(query)

        key = self._cache_key(query)

        cached = self._query_cache.get(key)
        if cached is not None:
            self._query_cache.move_to_end(key)
            logger.debug("Query embedding cache hit")
            return cached

        inflight = self._inflight.get(key)
        if inflight is not None:
            logger.debug("Joining in-flight prefetch for this query")
            return await asyncio.shield(inflight)

        return await self._embed_and_cache(key, query)

    async def _embed_and_cache(self, key: str, query: str) -> list[float]:
        """Run the embedder once, populate the LRU, clear the in-flight slot."""
        task = asyncio.ensure_future(self._embedder.embed_query(query))
        self._inflight[key] = task
        try:
            embedding = await task
        finally:
            self._inflight.pop(key, None)

        self._query_cache[key] = embedding
        while len(self._query_cache) > self._config.query_cache_size:
            self._query_cache.popitem(last=False)
        return embedding

    def prefetch(self, query: str) -> None:
        """Start embedding *query* now, in the background. Never blocks.

        The point is latency. Embedding a query is a network round-trip
        (~150-400ms on Gemini) and it is the only slow step in retrieval —
        the search itself is a warm numpy dot product. But the pipeline
        already spends time on NLU classification *before* reasoning
        begins, so if the embedding starts when the transcript arrives it
        finishes inside that existing window and
        :meth:`retrieve` later finds it already cached.

        Wired in ``core.pipeline`` off ``TranscriptReady`` /
        ``TextInputReceived``. Fire-and-forget: a prefetch failure is
        logged at debug and the real retrieval simply re-embeds.
        """
        if not self._ready or self._config.query_cache_size <= 0:
            return

        query = (query or "").strip()
        if not query:
            return

        key = self._cache_key(query)
        if key in self._query_cache or key in self._inflight:
            return

        async def _run() -> None:
            try:
                await self._embed_and_cache(key, query)
                logger.debug("Prefetched query embedding")
            except Exception as exc:  # noqa: BLE001
                logger.debug("Query prefetch failed (harmless): %s", exc)

        task = asyncio.create_task(_run(), name="rag_query_prefetch")
        # Hold a reference so the task cannot be garbage-collected mid-flight.
        self._prefetch_tasks.add(task)
        task.add_done_callback(self._prefetch_tasks.discard)

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
        doc_types: tuple[str, ...] | None = None,
        timeout_s: float | None = None,
    ) -> list[RetrievedChunk]:
        """Return the chunks most relevant to *query*.

        Returns ``[]`` — never raises — for an empty query, an empty
        index, an embedding failure, or a timeout. Retrieval enhances a
        reply; it must never be the reason a reply is late or missing.

        The whole operation is bounded by
        ``KANCHA_RAG_RETRIEVAL_TIMEOUT_MS`` (default 600ms). On a voice
        assistant an overrun is heard as dead air, so exceeding the
        budget deliberately degrades to answering without context rather
        than making the user wait.
        """
        self._require_ready()

        query = (query or "").strip()
        if not query:
            return []

        budget = (
            timeout_s if timeout_s is not None else self._config.retrieval_timeout_s
        )
        started = time.perf_counter()
        try:
            return await asyncio.wait_for(
                self._retrieve_inner(query, top_k, threshold, doc_types, started),
                timeout=budget,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Retrieval exceeded its %.0fms budget — answering without context",
                budget * 1000,
            )
            return []

    async def _retrieve_inner(
        self,
        query: str,
        top_k: int | None,
        threshold: float | None,
        doc_types: tuple[str, ...] | None,
        started: float,
    ) -> list[RetrievedChunk]:
        effective_k = top_k if top_k is not None else self._config.top_k
        effective_threshold = (
            threshold if threshold is not None else self._config.similarity_threshold
        )

        try:
            total = await self._store.count_chunks()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector store unreachable during retrieval: %s", exc)
            return []

        if total == 0:
            logger.debug("Retrieval skipped — index is empty")
            return []

        try:
            embedding = await self._embed_query_cached(query)
        except EmbeddingError as exc:
            logger.warning("Query embedding failed (%s) — skipping retrieval", exc)
            return []

        try:
            matches = await self._store.search(
                embedding=embedding,
                top_k=effective_k,
                threshold=effective_threshold,
                doc_types=doc_types,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector search failed: %s", exc)
            return []

        results = [RetrievedChunk.from_match(match) for match in matches]
        logger.info(
            "Retrieved %d/%d chunk(s) in %.0fms for %.60r "
            "(threshold=%.2f, top_score=%.3f)",
            len(results),
            total,
            (time.perf_counter() - started) * 1000,
            query,
            effective_threshold,
            results[0].score if results else 0.0,
        )
        return results

    # ── document management ──────────────────────────────────────────

    async def list_documents(self) -> list[dict[str, Any]]:
        self._require_ready()
        return await self._store.list_documents()

    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        self._require_ready()
        return await self._store.get_document(document_id)

    async def delete_document(self, document_id: str) -> int:
        self._require_ready()
        async with self._write_lock:
            return await self._store.delete_document(document_id)

    async def has_content(self, text: str) -> bool:
        """True if *text* is already indexed verbatim. Used by callers
        that want to check before doing expensive preparation work."""
        self._require_ready()
        return bool(await self._store.existing_hashes([content_hash(text)]))

    async def stats(self) -> dict[str, Any]:
        """Snapshot for the health endpoint and the UI's RAG panel."""
        try:
            documents = await self._store.list_documents()
            chunks = await self._store.count_chunks()
        except Exception as exc:  # noqa: BLE001
            return {"ready": self._ready, "error": str(exc)}

        return {
            "ready": self._ready,
            "enabled": self._config.enabled,
            "embedder": self._embedder.name,
            "store": self._store.name,
            "dimensions": self._config.embedding_dimensions,
            "documents": len(documents),
            "chunks": chunks,
            "top_k": self._config.top_k,
            "similarity_threshold": self._config.similarity_threshold,
            "chunk_size": self._config.chunk_size,
            "chunk_overlap": self._config.chunk_overlap,
        }

    async def health_check(self) -> bool:
        if not self._ready:
            return False
        return await self._store.health_check()
