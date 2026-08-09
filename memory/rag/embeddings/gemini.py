"""Gemini embedding provider.

Default provider, chosen because it adds **no new dependency** — the
project already ships ``google-genai`` and already loads a pool of Gemini
API keys for the conversational model
(:mod:`reasoning.llm_client_mulapi`). This module reuses that same pool
so embeddings inherit key rotation, quota cooldowns and dead-key
handling for free.

Asymmetric task types are used (``RETRIEVAL_DOCUMENT`` when indexing,
``RETRIEVAL_QUERY`` when searching), which is what the model was trained
for and measurably beats embedding both sides identically.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from .base import EmbeddingError, EmbeddingProvider, l2_normalise

logger = logging.getLogger("kancha.memory.rag.embeddings.gemini")

_TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
_TASK_QUERY = "RETRIEVAL_QUERY"


def _resolve_api_key() -> str | None:
    """Find a Gemini API key without duplicating the pool's env parsing.

    Preference order:

    1. The live :class:`~reasoning.llm_client_mulapi.KeyPool` — so the
       embedder rides the same rotation/cooldown state as chat.
    2. Raw environment variables, for the case where the pool has not
       been constructed yet (e.g. a standalone ingest script).

    The pool import is deliberately lazy: ``memory`` must not acquire an
    import-time dependency on ``reasoning``.
    """
    try:
        from reasoning.llm_client_mulapi import get_pool  # noqa: PLC0415

        pool = get_pool()
        for entry in pool.entries():
            if entry.is_available:
                return entry.key
        # Every key cooling or dead — fall through to the env scan rather
        # than blocking; the caller will surface a clean EmbeddingError.
    except Exception as exc:  # noqa: BLE001
        logger.debug("Key pool unavailable for embeddings (%s) — using env", exc)

    for i in range(1, 10):
        key = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
        if key:
            return key
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        key = os.getenv(name, "").strip()
        if key:
            return key
    return None


class GeminiEmbedder(EmbeddingProvider):
    """Embeddings via the Google GenAI SDK."""

    def __init__(
        self,
        model: str = "text-embedding-004",
        dimensions: int = 768,
        batch_size: int = 16,
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._dimensions = max(8, dimensions)
        self._batch_size = max(1, batch_size)
        self._timeout = timeout
        self._client: Any = None
        # Not every embedding model accepts output_dimensionality. We
        # probe once and remember, instead of failing every call.
        self._supports_output_dim = True

    @property
    def name(self) -> str:
        return f"gemini:{self._model}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    # ── client ───────────────────────────────────────────────────────

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from google import genai  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError(
                "google-genai is not installed; run: uv add google-genai"
            ) from exc

        api_key = _resolve_api_key()
        if not api_key:
            raise EmbeddingError(
                "No Gemini API key available for embeddings. Set GEMINI_API_KEY "
                "(or GEMINI_API_KEY_1..9) in .env, or switch provider with "
                "KANCHA_RAG_EMBEDDING_PROVIDER=hash."
            )
        self._client = genai.Client(api_key=api_key)
        return self._client

    # ── core call ────────────────────────────────────────────────────

    def _embed_sync(self, texts: list[str], task_type: str) -> list[list[float]]:
        """One blocking embed call. Runs on a worker thread."""
        from google.genai import types  # noqa: PLC0415

        client = self._get_client()

        def _call(with_dim: bool):
            kwargs: dict[str, Any] = {"task_type": task_type}
            if with_dim:
                kwargs["output_dimensionality"] = self._dimensions
            return client.models.embed_content(
                model=self._model,
                contents=texts,
                config=types.EmbedContentConfig(**kwargs),
            )

        try:
            response = _call(self._supports_output_dim)
        except Exception as exc:  # noqa: BLE001
            if self._supports_output_dim:
                # Retry once without the dimensionality hint — older
                # models reject it outright.
                logger.info(
                    "Embedding model %s rejected output_dimensionality (%s); "
                    "retrying without it",
                    self._model,
                    exc,
                )
                self._supports_output_dim = False
                response = _call(False)
            else:
                raise

        embeddings = getattr(response, "embeddings", None) or []
        vectors: list[list[float]] = []
        for item in embeddings:
            values = getattr(item, "values", None)
            if not values:
                raise EmbeddingError("Gemini returned an embedding with no values")
            # Truncate defensively: when output_dimensionality is not
            # supported the model returns its native width, which must
            # still match what the store expects.
            vectors.append(l2_normalise(list(values)[: self._dimensions]))

        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"Gemini returned {len(vectors)} embeddings for {len(texts)} inputs"
            )
        return vectors

    async def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                vectors = await asyncio.wait_for(
                    asyncio.to_thread(self._embed_sync, batch, task_type),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError as exc:
                raise EmbeddingError(
                    f"Gemini embedding timed out after {self._timeout}s"
                ) from exc
            except EmbeddingError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(f"Gemini embedding failed: {exc}") from exc
            out.extend(vectors)
        return out

    # ── public API ───────────────────────────────────────────────────

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, _TASK_DOCUMENT)

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed([text], _TASK_QUERY)
        if not vectors:
            raise EmbeddingError("Gemini returned no embedding for the query")
        return vectors[0]
