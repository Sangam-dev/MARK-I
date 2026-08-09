"""Deterministic, offline, dependency-free embedding provider.

This is **not** a semantic embedder. It projects hashed word and
character-trigram features into a fixed-width vector, which gives useful
lexical-overlap retrieval and nothing more — "car" and "automobile" will
not match.

It exists for three concrete reasons:

1. The RAG subsystem must degrade gracefully rather than crash when no
   API key is configured or the network is down.
2. Tests need an embedder that is fast, deterministic and offline.
3. It documents the provider contract in ~60 readable lines.

Select it with ``KANCHA_RAG_EMBEDDING_PROVIDER=hash``. The manager also
falls back to it automatically if the configured provider fails its
startup health check.
"""

from __future__ import annotations

import hashlib
import math
import re

from .base import EmbeddingProvider, l2_normalise

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _feature_bucket(feature: str, dimensions: int) -> int:
    """Map a string feature to a stable vector index.

    ``hash()`` is deliberately avoided — Python randomises string hashing
    per process, which would make stored vectors incomparable across
    restarts.
    """
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions


class HashingEmbedder(EmbeddingProvider):
    """Hashed bag-of-features embedder with sublinear term weighting."""

    def __init__(self, dimensions: int = 768) -> None:
        self._dimensions = max(8, dimensions)

    @property
    def name(self) -> str:
        return f"hash-{self._dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        tokens = _TOKEN_RE.findall((text or "").casefold())
        if not tokens:
            return vector

        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
            # Character trigrams give partial credit for morphological
            # variants ("debugging" / "debugger") that whole-word
            # matching would miss entirely.
            for i in range(len(token) - 2):
                trigram = f"#{token[i : i + 3]}"
                counts[trigram] = counts.get(trigram, 0) + 1

        for feature, count in counts.items():
            # 1 + log(tf) damps the influence of repeated boilerplate.
            weight = 1.0 + math.log(count)
            vector[_feature_bucket(feature, self._dimensions)] += weight

        return l2_normalise(vector)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    async def health_check(self) -> bool:
        # Pure computation — always available.
        return True
