"""Thin adapter presenting the battle-tested GeminiClient as an LLMProvider.

The adapter owns no logic — it delegates every call. Constructing a fresh
``GeminiClient`` is cheap even at swap time: the key pool is a process-wide
singleton (``get_pool()``), so lanes rebind instantly without re-reading
keys or re-establishing anything.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from reasoning.llm_client import GeminiClient

from reasoning.providers.base import LLMProvider


class GeminiAdapter(LLMProvider):
    """LLMProvider facade over :class:`reasoning.llm_client.GeminiClient`."""

    def __init__(self, client: GeminiClient) -> None:
        self._client = client
        self.provider_label = "gemini"

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def inner(self) -> GeminiClient:
        return self._client

    async def initialize(self) -> None:
        await self._client.initialize()

    async def generate(
        self,
        prompt: str,
        system: str = "",
        hedge_width: int | None = None,
        call_site: str = "conversational",
    ) -> str:
        return await self._client.generate(
            prompt=prompt,
            system=system,
            hedge_width=hedge_width,
            call_site=call_site,
        )

    async def generate_with_history(
        self,
        history: list[dict],
        system: str = "",
        hedge_width: int | None = None,
        call_site: str = "conversational",
    ) -> str:
        return await self._client.generate_with_history(
            history=history,
            system=system,
            hedge_width=hedge_width,
            call_site=call_site,
        )

    def generate_with_history_stream(
        self,
        history: list[dict],
        system: str = "",
        hedge_width: int | None = None,
        call_site: str = "conversational",
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        return self._client.generate_with_history_stream(
            history=history,
            system=system,
            hedge_width=hedge_width,
            call_site=call_site,
            max_output_tokens=max_output_tokens,
        )

    async def generate_with_history_json(
        self,
        history: list[dict],
        system: str = "",
        hedge_width: int | None = None,
        call_site: str = "conversational",
    ) -> dict[str, Any]:
        return await self._client.generate_with_history_json(
            history=history,
            system=system,
            hedge_width=hedge_width,
            call_site=call_site,
        )

    async def generate_json(
        self,
        prompt: str,
        schema_description: str | None = None,
        system: str = "",
        hedge_width: int | None = None,
        call_site: str = "planner",
    ) -> dict[str, Any]:
        return await self._client.generate_json(
            prompt=prompt,
            schema_description=schema_description,
            system=system,
            hedge_width=hedge_width,
            call_site=call_site,
        )

    async def health_check(self) -> bool:
        return await self._client.health_check()

    def has_available_key(self) -> bool:
        return self._client.has_available_key()

    async def prewarm(self) -> None:
        await self._client.prewarm()
