"""Embedding providers and their factory.

Adding a provider is three steps: implement
:class:`~memory.rag.embeddings.base.EmbeddingProvider`, add a branch to
:func:`build_embedder`, and add its default model name to
``_DEFAULT_EMBEDDING_MODELS`` in :mod:`memory.rag.config`. Nothing else
in the codebase needs to change.
"""

from __future__ import annotations

import logging

from ..config import RAGConfig
from .base import EmbeddingError, EmbeddingProvider, l2_normalise
from .gemini import GeminiEmbedder
from .hashing import HashingEmbedder
from .ollama import OllamaEmbedder

logger = logging.getLogger("kancha.memory.rag.embeddings")

__all__ = [
    "EmbeddingError",
    "EmbeddingProvider",
    "GeminiEmbedder",
    "HashingEmbedder",
    "OllamaEmbedder",
    "build_embedder",
    "l2_normalise",
]


def build_embedder(config: RAGConfig) -> EmbeddingProvider:
    """Instantiate the configured embedding provider.

    Never raises for an unknown provider — config validation already
    normalised it — but construction itself is lazy, so a missing API key
    surfaces on first use rather than at startup.
    """
    provider = config.embedding_provider

    if provider == "ollama":
        return OllamaEmbedder(
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
            base_url=config.ollama_url,
            batch_size=config.embedding_batch_size,
        )

    if provider == "hash":
        return HashingEmbedder(dimensions=config.embedding_dimensions)

    return GeminiEmbedder(
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
        batch_size=config.embedding_batch_size,
    )
