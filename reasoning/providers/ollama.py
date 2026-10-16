"""Ollama backend — native ``/api/chat`` instead of the ``/v1`` facade.

Two daemon features the app needs are silently ignored by Ollama's
OpenAI-compatible endpoint, so Ollama gets its own thin wire adapter:

* ``keep_alive`` — ``/v1/chat/completions`` drops it (ollama#11458), so
  the model unloads after the 5-minute idle default and every later turn
  pays a full cold-load TTFT again.
* ``think`` / ``reasoning_effort`` / the ``/no_think`` soft switch —
  ``/v1`` ignores all of them (ollama#17969), so qwen3 keeps thinking and
  the first *content* token is withheld until the reasoning trace ends.

The native endpoint honours both. Everything else — key lanes, model
fallback, the streaming contract, token accounting — is inherited
unchanged from :class:`~reasoning.providers.openai_compat.OpenAICompatClient`;
only the wire format differs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator

from reasoning.providers.openai_compat import (
    OpenAICompatClient,
    ProviderConfig,
    _Entry,
)

logger = logging.getLogger("kancha.reasoning.providers.ollama")

# Longer than the OpenAI-compatible default: a cold 8B load can take far
# more than the stock 12s, and prewarm is detached from startup so a long
# budget never delays boot.
_DEFAULT_TIMEOUT = float(os.getenv("KANCHA_OLLAMA_TIMEOUT", "60.0"))
_PREWARM_TIMEOUT = float(os.getenv("KANCHA_OLLAMA_PREWARM_TIMEOUT", "180.0"))

_TRUTHY = {"1", "true", "yes", "on"}


def thinking_enabled() -> bool:
    """Whether qwen3's thinking step is left on (default: off)."""
    return os.getenv("KANCHA_OLLAMA_THINK", "0").strip().lower() in _TRUTHY


def keep_alive() -> int | str:
    """``keep_alive`` value sent with every request.

    ``-1`` (the default) means "never unload"; any other integer is a
    seconds count, and a non-numeric string is passed through verbatim so
    Ollama duration syntax like ``"30m"`` still works.
    """
    raw = os.getenv("KANCHA_OLLAMA_KEEP_ALIVE", "-1").strip()
    if not raw:
        return -1
    try:
        return int(raw)
    except ValueError:
        return raw


def normalize_base_url(base_url: str) -> str:
    """Drop a trailing ``/v1`` so native paths resolve at the daemon root."""
    url = base_url.strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url or "http://localhost:11434"


class OllamaClient(OpenAICompatClient):
    """LLMProvider over Ollama's native ``/api/chat`` endpoint."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config,
            timeout=timeout if timeout is not None else _DEFAULT_TIMEOUT,
            **kwargs,
        )
        # Accept either form of the env override: with or without ``/v1``.
        self.config.base_url = normalize_base_url(self.config.base_url)

    # ── Wire format ──────────────────────────────────────────────────────

    def _body(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        stream: bool,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "keep_alive": keep_alive(),
        }
        # Omitting ``think`` leaves the model default (thinking on); when
        # the knob is off we must send it explicitly to suppress it.
        if not thinking_enabled():
            body["think"] = False
        if max_tokens is not None:
            body["options"] = {"num_predict": max_tokens}
        return body

    async def _post_chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int | None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        assert self._client is not None
        body = self._body(messages, model, stream=False, max_tokens=max_tokens)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        resp = await self._client.post(
            "/api/chat", json=body, headers=headers, timeout=timeout
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        return data.get("message", {}).get("content") or ""

    async def _open_stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        entry: _Entry,
    ) -> AsyncIterator[str]:
        """Open one native NDJSON chat stream and return the delta generator.

        Same contract as the base SSE variant: a plain coroutine so the
        request fires immediately and key-level failures surface to the
        rotation guard before a generator object exists.
        """
        assert self._client is not None
        body = self._body(messages, model, stream=True, max_tokens=None)
        req = self._client.build_request("POST", "/api/chat", json=body)
        if entry.api_key:
            req.headers["Authorization"] = f"Bearer {entry.api_key}"
        resp = await self._client.send(req, stream=True)

        async def _gen() -> AsyncIterator[str]:
            try:
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("error"):
                        logger.warning(
                            "%s stream error: %s",
                            self.provider_label,
                            chunk["error"],
                        )
                        return
                    delta = (chunk.get("message") or {}).get("content")
                    if delta:
                        yield delta
                    if chunk.get("done"):
                        return
            finally:
                await resp.aclose()

        try:
            resp.raise_for_status()
        except BaseException:
            await resp.aclose()
            raise
        return _gen()

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def prewarm(self) -> None:
        """Load the model once and pin it resident for later turns.

        A cold qwen3:8b load is the dominant TTFT cost, so unlike the base
        implementation this waits long enough for the load to finish and
        relies on ``keep_alive`` to keep it in memory afterwards.
        """
        if self._lane is None or not self.has_available_key():
            return
        try:
            await asyncio.wait_for(
                self._post_chat(
                    [{"role": "user", "content": "Reply with the single word: ready."}],
                    model=self.models[0],
                    max_tokens=1,
                    timeout=_PREWARM_TIMEOUT,
                ),
                timeout=_PREWARM_TIMEOUT,
            )
            logger.info(
                "%s prewarm complete (think=%s, keep_alive=%s)",
                self.provider_label,
                thinking_enabled(),
                keep_alive(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("%s prewarm skipped (non-fatal): %s", self.provider_label, exc)
