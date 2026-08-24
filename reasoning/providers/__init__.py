"""Multi-provider LLM backends behind one stable interface.

Public surface:
    * :class:`LLMProvider` — the contract every backend implements
    * :class:`SwappableClient` — per-lane facade handed to consumers
    * :class:`ProviderRouter` — validated hot-swap across providers
    * ``PROVIDERS`` — the registry (add a dict entry to add a provider)
"""

from reasoning.providers.base import LLMProvider
from reasoning.providers.gemini_adapter import GeminiAdapter
from reasoning.providers.openai_compat import OpenAICompatClient, ProviderConfig
from reasoning.providers.registry import (
    DEFAULT_PROVIDER,
    PROVIDERS,
    ProviderRouter,
    ProviderSpec,
    SwappableClient,
    get_shared_router,
    read_configured_provider,
    set_shared_router,
)

__all__ = [
    "DEFAULT_PROVIDER",
    "PROVIDERS",
    "LLMProvider",
    "OpenAICompatClient",
    "ProviderConfig",
    "ProviderRouter",
    "ProviderSpec",
    "GeminiAdapter",
    "SwappableClient",
    "get_shared_router",
    "read_configured_provider",
    "set_shared_router",
]
