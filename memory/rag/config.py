"""Configuration for the RAG subsystem.

Every tunable lives here and is sourced from the environment with a
documented default — no RAG value is hardcoded at its call site. The
config object is constructed once in :func:`core.pipeline.build_pipeline`
and threaded down into the manager, store, embedder and loaders.

Environment variables (all optional, all prefixed ``KANCHA_RAG_``)::

    KANCHA_RAG_ENABLED              1|0                (default 1)
    KANCHA_RAG_EMBEDDING_PROVIDER   gemini|ollama|hash (default gemini)
    KANCHA_RAG_EMBEDDING_MODEL      provider-specific  (default per provider)
    KANCHA_RAG_EMBEDDING_DIM        int                (default 768)
    KANCHA_RAG_EMBEDDING_BATCH      int                (default 16)
    KANCHA_RAG_OLLAMA_URL           url                (default http://localhost:11434)
    KANCHA_RAG_VECTOR_STORE         sqlite|chroma      (default sqlite)
    KANCHA_RAG_COLLECTION           str                (default kancha_rag)
    KANCHA_RAG_CHUNK_SIZE           chars              (default 1200)
    KANCHA_RAG_CHUNK_OVERLAP        chars              (default 180)
    KANCHA_RAG_MIN_CHUNK_CHARS      chars              (default 60)
    KANCHA_RAG_TOP_K                int                (default 5)
    KANCHA_RAG_SIMILARITY_THRESHOLD 0.0-1.0            (default 0.25)
    KANCHA_RAG_MAX_CONTEXT_CHARS    chars              (default 4000)
    KANCHA_RAG_DEDUPE_THRESHOLD     0.0-1.0            (default 0.95)
    KANCHA_RAG_ROUTER               heuristic|always|never (default heuristic)
    KANCHA_RAG_MAX_UPLOAD_MB        int                (default 25)
    KANCHA_RAG_EXTENSIONS           csv                (default .pdf,.txt,.md,.docx)
    KANCHA_RAG_SYNC_ON_BOOT         1|0                (default 1)
    KANCHA_RAG_RETRIEVAL_TIMEOUT_MS ms                 (default 600)
    KANCHA_RAG_QUERY_CACHE          entries            (default 256)
    KANCHA_RAG_WARM_ON_BOOT         1|0                (default 1)

The defaults are deliberately chosen so the subsystem works with **zero**
extra dependencies and zero extra configuration: ``sqlite`` vector store
(via ``aiosqlite``, already a project dependency) and ``gemini``
embeddings (via ``google-genai``, already a project dependency, reusing
the existing API key pool).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("kancha.memory.rag.config")

# Provider defaults — keyed by provider name so switching providers does
# not require also switching the model env var.
_DEFAULT_EMBEDDING_MODELS = {
    "gemini": "gemini-embedding-001",
    "ollama": "nomic-embed-text",
    "hash": "hash-768",
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("%s=%r is not an integer — using %d", name, raw, default)
        return default


def _env_float(name: str, default: float, *, low: float, high: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return min(high, max(low, float(raw)))
    except ValueError:
        logger.warning("%s=%r is not a number — using %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return default


def _natural_sort_key(path: Path) -> tuple:
    """Sort key that orders embedded numbers numerically.

    Plain lexicographic sorting puts ``rag.10.txt`` before ``rag.2.txt``,
    which would replay a rotation scheme out of order. Splitting digit
    runs out and comparing them as integers fixes that, and still handles
    date-stamped names (``rag.2026-07.txt``) correctly.
    """
    parts = re.split(r"(\d+)", path.name)
    # (0, int) sorts before (1, str) so numeric segments never compare
    # against text segments — that pairing is what raises TypeError.
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in parts)


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    normalised = tuple(p if p.startswith(".") else f".{p}" for p in parts)
    return normalised or default


@dataclass(frozen=True, slots=True)
class RAGConfig:
    """Immutable configuration bundle for the whole RAG subsystem."""

    # ── Lifecycle ────────────────────────────────────────────────────
    enabled: bool = True
    data_dir: Path = field(default_factory=lambda: Path("memory/data"))

    # ── Embeddings ───────────────────────────────────────────────────
    embedding_provider: str = "gemini"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    embedding_batch_size: int = 16
    ollama_url: str = "http://localhost:11434"

    # ── Vector store ─────────────────────────────────────────────────
    vector_store: str = "sqlite"
    collection_name: str = "kancha_rag"

    # ── Chunking ─────────────────────────────────────────────────────
    chunk_size: int = 1200
    chunk_overlap: int = 180
    min_chunk_chars: int = 60

    # ── Retrieval ────────────────────────────────────────────────────
    top_k: int = 5
    similarity_threshold: float = 0.25
    max_context_chars: int = 4000

    # ── Deduplication ────────────────────────────────────────────────
    dedupe_similarity_threshold: float = 0.95

    # ── Routing ──────────────────────────────────────────────────────
    router_strategy: str = "heuristic"

    # ── Latency budget ───────────────────────────────────────────────
    # This is a *voice* assistant: a slow lookup is worse than no lookup,
    # because the user hears silence. Retrieval is therefore hard-capped
    # and degrades to "answer without context" when it overruns.
    retrieval_timeout_ms: int = 600
    query_cache_size: int = 256
    warm_on_boot: bool = True

    # ── Boot-time sync ───────────────────────────────────────────────
    # Replay memory/rag.txt into the vector index at startup so history
    # written before the index existed (or by another process) is
    # searchable. Idempotent — the hash dedupe pass makes re-runs free.
    sync_on_boot: bool = True

    # ── Upload / ingest ──────────────────────────────────────────────
    max_upload_bytes: int = 25 * 1024 * 1024
    supported_extensions: tuple[str, ...] = (".pdf", ".txt", ".md", ".docx")

    # ── Derived paths ────────────────────────────────────────────────

    @property
    def rag_dir(self) -> Path:
        """Root directory for all RAG state (vector db + uploads)."""
        return self.data_dir / "rag"

    @property
    def store_path(self) -> Path:
        """SQLite vector-store file (unused by the chroma backend)."""
        return self.rag_dir / "vectors.db"

    @property
    def chroma_dir(self) -> Path:
        """Persist directory for the chroma backend."""
        return self.rag_dir / "chroma"

    @property
    def upload_dir(self) -> Path:
        """Temporary staging area for uploaded files (see ingest.py)."""
        return self.rag_dir / "uploads"

    @property
    def memory_dir(self) -> Path:
        """The ``memory/`` package directory, where the audit logs live."""
        return Path(__file__).resolve().parent.parent

    @property
    def rag_file_path(self) -> Path:
        """The *live* audit log written by ``MemoryManager.append_rag``.

        Lives in the ``memory/`` package rather than under ``data_dir``
        because that is where :meth:`memory.manager.MemoryManager._rag_file_path`
        has always put it. Kept in sync with that method deliberately —
        the boot sync reads exactly the file the conversation path writes.
        """
        return self.memory_dir / "rag.txt"

    @property
    def rag_file_paths(self) -> list[Path]:
        """Every audit log to replay at boot, oldest first.

        Matches ``rag*.txt`` so rotated archives (``rag.2026-07.txt``,
        ``rag.1.txt``, …) are picked up alongside the live ``rag.txt``.
        Without this a rotation scheme would quietly orphan its own
        history: the vector index would still hold everything already
        ingested, but a rebuild — new machine, deleted ``vectors.db``, or
        a change of embedding model — would recover only the current file.

        Ordering matters for one reason: when the same entry appears in
        two files the first one indexed wins and supplies the
        ``recorded_at`` metadata, so archives must be replayed before the
        live file to keep original timestamps. Sorting is *natural*
        (``rag.2.txt`` before ``rag.10.txt``) with ``rag.txt`` forced
        last, since it is always the newest.
        """
        directory = self.memory_dir
        try:
            candidates = [p for p in directory.glob("rag*.txt") if p.is_file()]
        except OSError:
            return []

        live = self.rag_file_path
        archives = sorted(
            (p for p in candidates if p != live), key=_natural_sort_key
        )
        if live.is_file():
            archives.append(live)
        return archives

    @property
    def retrieval_timeout_s(self) -> float:
        return self.retrieval_timeout_ms / 1000.0

    # ── Construction ─────────────────────────────────────────────────

    @classmethod
    def from_env(cls, data_dir: Path | None = None) -> "RAGConfig":
        """Build a config from environment variables.

        *data_dir* is the same ``memory/data`` directory the rest of the
        memory layer uses; passing it explicitly keeps the RAG store next
        to ``structured.db`` instead of guessing a location.
        """
        provider = _env_str("KANCHA_RAG_EMBEDDING_PROVIDER", "gemini").lower()
        if provider not in _DEFAULT_EMBEDDING_MODELS:
            logger.warning(
                "Unknown embedding provider %r — falling back to 'gemini'", provider
            )
            provider = "gemini"

        store = _env_str("KANCHA_RAG_VECTOR_STORE", "sqlite").lower()
        if store not in {"sqlite", "chroma"}:
            logger.warning("Unknown vector store %r — falling back to 'sqlite'", store)
            store = "sqlite"

        router = _env_str("KANCHA_RAG_ROUTER", "heuristic").lower()
        if router not in {"heuristic", "always", "never"}:
            logger.warning("Unknown router %r — falling back to 'heuristic'", router)
            router = "heuristic"

        chunk_size = _env_int("KANCHA_RAG_CHUNK_SIZE", 1200, minimum=100)
        overlap = _env_int("KANCHA_RAG_CHUNK_OVERLAP", 180, minimum=0)
        if overlap >= chunk_size:
            logger.warning(
                "chunk_overlap (%d) >= chunk_size (%d) — clamping overlap to %d",
                overlap,
                chunk_size,
                chunk_size // 4,
            )
            overlap = chunk_size // 4

        return cls(
            enabled=_env_bool("KANCHA_RAG_ENABLED", True),
            data_dir=data_dir or Path("memory/data"),
            embedding_provider=provider,
            embedding_model=_env_str(
                "KANCHA_RAG_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODELS[provider]
            ),
            embedding_dimensions=_env_int("KANCHA_RAG_EMBEDDING_DIM", 768, minimum=8),
            embedding_batch_size=_env_int("KANCHA_RAG_EMBEDDING_BATCH", 16),
            ollama_url=_env_str("KANCHA_RAG_OLLAMA_URL", "http://localhost:11434"),
            vector_store=store,
            collection_name=_env_str("KANCHA_RAG_COLLECTION", "kancha_rag"),
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            min_chunk_chars=_env_int("KANCHA_RAG_MIN_CHUNK_CHARS", 60, minimum=1),
            top_k=_env_int("KANCHA_RAG_TOP_K", 5),
            similarity_threshold=_env_float(
                "KANCHA_RAG_SIMILARITY_THRESHOLD", 0.25, low=0.0, high=1.0
            ),
            max_context_chars=_env_int("KANCHA_RAG_MAX_CONTEXT_CHARS", 4000, minimum=200),
            dedupe_similarity_threshold=_env_float(
                "KANCHA_RAG_DEDUPE_THRESHOLD", 0.95, low=0.0, high=1.0
            ),
            router_strategy=router,
            retrieval_timeout_ms=_env_int(
                "KANCHA_RAG_RETRIEVAL_TIMEOUT_MS", 600, minimum=50
            ),
            query_cache_size=_env_int("KANCHA_RAG_QUERY_CACHE", 256, minimum=0),
            warm_on_boot=_env_bool("KANCHA_RAG_WARM_ON_BOOT", True),
            sync_on_boot=_env_bool("KANCHA_RAG_SYNC_ON_BOOT", True),
            max_upload_bytes=_env_int("KANCHA_RAG_MAX_UPLOAD_MB", 25) * 1024 * 1024,
            supported_extensions=_env_csv(
                "KANCHA_RAG_EXTENSIONS", (".pdf", ".txt", ".md", ".docx")
            ),
        )

    def describe(self) -> str:
        """One-line summary for startup logs."""
        return (
            f"provider={self.embedding_provider}:{self.embedding_model} "
            f"store={self.vector_store} chunk={self.chunk_size}/{self.chunk_overlap} "
            f"top_k={self.top_k} threshold={self.similarity_threshold} "
            f"router={self.router_strategy} timeout={self.retrieval_timeout_ms}ms"
        )
