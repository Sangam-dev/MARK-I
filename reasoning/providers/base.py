"""Provider-agnostic contract for the conversation/task LLM.

Every backend (Gemini, Groq, Ollama, ...) exposes exactly this surface.
It mirrors :class:`reasoning.llm_client.GeminiClient` method-for-method so
the existing consumers — the NLU classifier, the planner, the coordinator
and the pipeline build itself — keep working no matter which adapter sits
behind them. Adding a provider means implementing this ABC once; nothing
else in the codebase changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Any


class LLMProvider(ABC):
    """One initialised LLM backend bound to a single key lane."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Registry name of the active backend ("gemini", "groq", ...)."""

    @abstractmethod
    async def initialize(self) -> None:
        """Bind the provider to its resources (key pool, HTTP client)."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str = "",
        hedge_width: int | None = None,
        call_site: str = "conversational",
    ) -> str:
        """One-shot prompt → response text. Never raises at runtime:
        failures degrade to a spoken fallback string."""

    @abstractmethod
    async def generate_with_history(
        self,
        history: list[dict],
        system: str = "",
        hedge_width: int | None = None,
        call_site: str = "conversational",
    ) -> str:
        """Multi-turn chat completion. *history* is a list of
        ``{"role": "user"|"assistant", "content": str}`` dicts whose last
        entry is the current user message."""

    @abstractmethod
    def generate_with_history_stream(
        self,
        history: list[dict],
        system: str = "",
        hedge_width: int | None = None,
        call_site: str = "conversational",
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream a conversational response as raw text deltas.

        The caller extracts the ``message`` field incrementally via
        :func:`reasoning.llm_client.extract_streamed_message` and parses
        the full envelope at the end. Never raises — on failure it yields
        a single fallback message instead."""

    @abstractmethod
    async def generate_with_history_json(
        self,
        history: list[dict],
        system: str = "",
        hedge_width: int | None = None,
        call_site: str = "conversational",
    ) -> dict[str, Any]:
        """Conversational turn parsed into the memory envelope dict."""

    @abstractmethod
    async def generate_json(
        self,
        prompt: str,
        schema_description: str | None = None,
        system: str = "",
        hedge_width: int | None = None,
        call_site: str = "planner",
    ) -> dict[str, Any]:
        """Prompt → parsed JSON object (retries on decode errors, ``{}``
        on total failure). Used by the planner and classifier."""

    @abstractmethod
    async def health_check(self) -> bool:
        """True when the backend can serve a request right now."""

    @abstractmethod
    def has_available_key(self) -> bool:
        """Non-blocking variant of :meth:`health_check` used by the
        coordinator's preemptive-generation gate."""

    @abstractmethod
    async def prewarm(self) -> None:
        """Best-effort warmup; must never raise or delay startup."""
