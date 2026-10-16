"""OpenAI-compatible chat backend — one adapter for Groq, Ollama and friends.

Groq (`https://api.groq.com/openai/v1`) and Ollama (`http://localhost:11434/v1`)
speak the same ``/chat/completions`` protocol, so a single httpx-based client
covers both plus any future OpenAI-style endpoint (OpenRouter, vLLM, LM
Studio). The rotation semantics mirror the Gemini key pool in
:mod:`reasoning.llm_client_mulapi`: sticky key, per-key cooldown on 429/503,
permanent death on 401/403, per-model 404 capability cache, opposite-walking
lanes so the Conversation and Task LLMs stop contending for one key.

No network happens at import or construction time; keys are discovered from
the environment exactly like the Gemini pool (``<PREFIX>_1..9`` then
``<PREFIX>``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from memory.token_log import TokenLog

from reasoning.providers.base import LLMProvider

logger = logging.getLogger("kancha.reasoning.providers.openai_compat")

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "12.0"))
KEY_COOLDOWN_SECS = float(os.getenv("KEY_COOLDOWN_SECS", "60.0"))

FORWARD = 1
REVERSE = -1


def _log(msg: str) -> None:
    sys.stderr.write(msg + "\n")


@dataclass
class ProviderConfig:
    """Static description of one OpenAI-compatible endpoint."""

    name: str
    base_url: str
    default_models: list[str]
    key_prefix: str = ""  # "" -> no API keys needed (local daemons)
    model_env: str = ""
    fallbacks_env: str = ""

    def models(self) -> list[str]:
        """Model priority list: env override first, then defaults."""
        primary = os.getenv(self.model_env, "").strip() if self.model_env else ""
        fallbacks_raw = (
            os.getenv(self.fallbacks_env, "").strip() if self.fallbacks_env else ""
        )
        fallbacks = [m.strip() for m in fallbacks_raw.split(",") if m.strip()]
        ordered = [primary] + fallbacks + list(self.default_models)
        return [m for m in dict.fromkeys(ordered) if m]


class _Entry:
    """One credential slot: an optional API key plus its health state.

    Local endpoints (Ollama) run with a single keyless entry — rotation
    degenerates to a no-op there, which is exactly what we want.
    """

    def __init__(self, api_key: str | None, index: int) -> None:
        self.api_key = api_key
        self.index = index
        self.cooling_until: float = 0.0
        self.dead: bool = False
        self.unsupported_models: set[str] = set()

    @property
    def is_available(self) -> bool:
        return not self.dead and time.monotonic() >= self.cooling_until

    def cool_down(self, secs: float, reason: str) -> None:
        self.cooling_until = time.monotonic() + secs
        logger.info("provider key[%d] cooling %.0fs (%s)", self.index, secs, reason)

    def mark_dead(self, reason: str) -> None:
        self.dead = True
        logger.warning("provider key[%d] dead (%s) — removed from rotation", self.index, reason)


class _Lane:
    """A consumer's sticky cursor over the shared entry list.

    Mirrors :class:`reasoning.llm_client_mulapi.KeyLane`: health state is
    shared through the entries themselves; only the starting position and
    walk direction are per-lane.
    """

    def __init__(self, entries: list[_Entry], direction: int = FORWARD) -> None:
        self._entries = entries
        self._direction = direction if direction != 0 else FORWARD
        n = len(entries)
        self.active_index = (n - 1) if self._direction < 0 and n else 0

    def entries(self) -> list[_Entry]:
        return self._entries

    async def acquire_active(self) -> _Entry:
        if not self._entries:
            raise RuntimeError("No provider credentials configured")
        # Skip cooling/dead entries forward from the active position.
        n = len(self._entries)
        for offset in range(n):
            idx = (self.active_index + offset * (1 if self._direction > 0 else -1)) % n
            entry = self._entries[idx]
            if entry.is_available:
                self.active_index = idx
                return entry
        # Everything is cooling — wait for the nearest recovery rather
        # than erroring; mirrors the Gemini lane's patience.
        await asyncio.sleep(min(KEY_COOLDOWN_SECS, 2.0))
        entry = self._entries[self.active_index]
        return entry

    async def advance_active(self) -> None:
        n = len(self._entries)
        if not n:
            return
        step = 1 if self._direction > 0 else -1
        self.active_index = (self.active_index + step) % n


def discover_keys(prefix: str) -> list[str]:
    """Scan ``<PREFIX>_1..9`` then ``<PREFIX>`` (+ ``<PREFIX>_KEY``),
    deduplicated — the same contract as the Gemini pool loader."""
    keys: list[str] = []
    seen: set[str] = set()
    for i in range(1, 10):
        k = os.getenv(f"{prefix}_{i}", "").strip()
        if k and k not in seen:
            keys.append(k)
            seen.add(k)
    for name in (prefix, f"{prefix}_KEY"):
        k = os.getenv(name, "").strip()
        if k and k not in seen:
            keys.append(k)
            seen.add(k)
    return keys


_RETRY_AFTER_RE = re.compile(r"retry[-_ ]after\D*(\d+(?:\.\d+)?)", re.IGNORECASE)


class OpenAICompatClient(LLMProvider):
    """LLMProvider over any OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        token_log: TokenLog | None = None,
        lane: str = "",
        lane_direction: int = FORWARD,
        timeout: float = REQUEST_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
        api_keys: list[str] | None = None,
    ) -> None:
        self.config = config
        self.provider_label = config.name
        self.models = config.models()
        self.model = self.models[0]
        self.timeout = timeout
        self.token_log = token_log
        self.lane_name = lane
        self.lane_direction = lane_direction
        self._transport = transport
        # Explicit credential injection (tests, programmatic use). When
        # None, credentials come from the environment via discover_keys().
        self._explicit_keys = list(api_keys) if api_keys is not None else None
        self._client: httpx.AsyncClient | None = None
        self._lane: _Lane | None = None

    @property
    def provider_name(self) -> str:
        return self.config.name

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        headers = {"Content-Type": "application/json"}
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.timeout,
                headers=headers,
                transport=self._transport,
            )
        if self._explicit_keys is not None:
            keys = list(self._explicit_keys)
        elif self.config.key_prefix:
            keys = discover_keys(self.config.key_prefix)
        else:
            keys = []

        # Credentials are required when a key prefix is configured; their
        # absence must surface as an unhealthy provider (refused swaps,
        # startup fallback) rather than an anonymous keyless entry.
        if not keys and self.config.key_prefix:
            entries: list[_Entry] = []
        else:
            entries = [_Entry(key, i) for i, key in enumerate(keys)] or [
                _Entry(None, 0)
            ]
        self._lane = _Lane(entries, self.lane_direction)
        start = self._lane.active_index
        logger.info(
            "%s client initialized on lane '%s' (key[%d], %d credential(s), %s)",
            self.provider_label,
            self.lane_name or "default",
            start,
            len(keys) or 1,
            "backward" if self.lane_direction < 0 else "forward",
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── Message assembly ─────────────────────────────────────────────────

    @staticmethod
    def _messages(prompt: str, system: str) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    @staticmethod
    def _messages_from_history(
        history: list[dict], system: str
    ) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        for msg in history:
            role = "assistant" if msg["role"] != "user" else "user"
            msgs.append({"role": role, "content": msg["content"]})
        # OpenAI-compatible APIs tolerate assistant-final turns, but keep
        # parity with the Gemini path's user-final guarantee.
        if msgs and msgs[-1]["role"] != "user":
            msgs.append({"role": "user", "content": ""})
        return msgs

    # ── Error classification ─────────────────────────────────────────────

    @staticmethod
    def _status(exc: Exception) -> int | None:
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code
        return None

    def _classify(self, exc: Exception, entry: _Entry, model: str) -> str:
        """Map an exception onto the rotation action, mutating *entry*.

        Returns one of: "dead", "cool", "unsupported", "raise".
        """
        status = self._status(exc)
        if status in (401, 403):
            entry.mark_dead(f"HTTP {status}")
            return "dead"
        if status == 404:
            entry.unsupported_models.add(model)
            return "unsupported"
        if status in (429, 503) or status is None:
            retry_after: float | None = None
            if isinstance(exc, httpx.HTTPStatusError):
                header = exc.response.headers.get("Retry-After", "")
                try:
                    retry_after = float(header) if header else None
                except ValueError:
                    retry_after = None
            if retry_after is None:
                match = _RETRY_AFTER_RE.search(str(exc))
                retry_after = float(match.group(1)) if match else None
            entry.cool_down(retry_after or KEY_COOLDOWN_SECS, reason=f"HTTP {status}")
            return "cool"
        return "raise"

    # ── Core request drivers ─────────────────────────────────────────────

    async def _post_chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int | None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        assert self._client is not None
        body: dict[str, Any] = {"model": model, "messages": messages}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        resp = await self._client.post(
            "/chat/completions", json=body, headers=headers
        )
        resp.raise_for_status()
        return resp.json()

    async def _try_with_active_key(
        self,
        messages: list[dict[str, str]],
        *,
        is_stream: bool,
        max_tokens: int | None = None,
    ) -> Any:
        """Sticky-key + model-fallback driver, semantically matching
        ``reasoning.llm_client_mulapi._try_with_active_key``."""
        assert self._lane is not None
        n = len(self._lane.entries())
        for _ in range(max(1, n)):
            entry = await self._lane.acquire_active()
            key_failed = False
            saw_any_attempt = False
            for model in self.models:
                if model in entry.unsupported_models:
                    continue
                saw_any_attempt = True
                try:
                    if is_stream:
                        gen = await self._open_stream(messages, model, entry)
                        # Pull the first chunk *inside* the try so a
                        # key-level failure (quota, auth, timeout)
                        # surfaces here and rotates to the next key.
                        try:
                            first = await anext(gen)
                        except StopAsyncIteration:
                            # An empty stream is not a failure — keep it.
                            first = None

                        async def _replay() -> AsyncIterator[str]:
                            if first is not None:
                                yield first
                            async for chunk in gen:
                                yield chunk

                        return _replay()
                    data = await self._post_chat(
                        messages, model, max_tokens, api_key=entry.api_key
                    )
                    return self._extract_content(data) or ""
                except Exception as exc:  # noqa: BLE001
                    action = self._classify(exc, entry, model)
                    if action == "dead":
                        key_failed = True
                        break
                    if action == "cool":
                        key_failed = True
                        break
                    if action == "unsupported":
                        continue
                    raise
            if not saw_any_attempt and not key_failed:
                key_failed = True
            if not key_failed:
                key_failed = True
            await self._lane.advance_active()
        raise RuntimeError(
            f"{self.provider_label}: all credentials + models exhausted"
        )

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        """Pull the assistant text out of a non-stream response body.

        Overridable so wire-format variants (see
        :class:`reasoning.providers.ollama.OllamaClient`) can parse their
        own envelope; the default is the OpenAI ``choices`` shape.
        """
        return (
            data.get("choices", [{}])[0].get("message", {}).get("content") or ""
        )

    async def _open_stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        entry: _Entry,
    ) -> AsyncIterator[str]:
        """Open one SSE chat completion and return the delta generator.

        A plain coroutine (not an async generator) so the HTTP request
        executes immediately and any status/auth error surfaces to the
        rotation guard before a generator object even exists. The
        response is always closed — by the generator on success, here on
        failure.
        """
        assert self._client is not None
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        req = self._client.build_request("POST", "/chat/completions", json=body)
        if entry.api_key:
            req.headers["Authorization"] = f"Bearer {entry.api_key}"
        resp = await self._client.send(req, stream=True)

        async def _gen() -> AsyncIterator[str]:
            try:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        return
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    delta = (
                        chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                    )
                    if delta:
                        yield delta
            finally:
                await resp.aclose()

        try:
            resp.raise_for_status()
        except BaseException:
            await resp.aclose()
            raise
        return _gen()

    # ── Token accounting ─────────────────────────────────────────────────

    def _record(self, call_site: str, prompt_chars: int, output_chars: int) -> None:
        if self.token_log is not None:
            self.token_log.record(
                call_site=call_site,
                model=self.model,
                input_tokens=max(1, prompt_chars // 4),
                output_tokens=max(1, output_chars // 4),
            )

    # ── LLMProvider surface ──────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system: str = "",
        hedge_width: int | None = None,
        call_site: str = "conversational",
    ) -> str:
        _ = hedge_width
        full_prompt = prompt
        if system:
            full_prompt = f"System Instruction: {system}\n\nUser Prompt: {prompt}"
        try:
            response = await self._try_with_active_key(
                self._messages(full_prompt, ""),
                is_stream=False,
            )
            self._record(call_site, len(full_prompt), len(response))
            return response
        except Exception as exc:  # noqa: BLE001
            logger.exception("generate failed: %s", exc)
            return "I'm having trouble thinking right now. Could you try again?"

    async def generate_with_history(
        self,
        history: list[dict],
        system: str = "",
        hedge_width: int | None = None,
        call_site: str = "conversational",
    ) -> str:
        _ = hedge_width
        if not history:
            return await self.generate(prompt="", system=system)
        try:
            response = await self._try_with_active_key(
                self._messages_from_history(history, system),
                is_stream=False,
            )
            self._record(call_site, len(str(history)), len(response))
            return response
        except Exception as exc:  # noqa: BLE001
            logger.exception("generate_with_history failed: %s", exc)
            return "I'm having trouble thinking right now. Could you try again?"

    async def generate_with_history_stream(
        self,
        history: list[dict],
        system: str = "",
        hedge_width: int | None = None,
        call_site: str = "conversational",
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        from reasoning.llm_client import MEMORY_RESPONSE_FORMAT

        _ = hedge_width
        full_system = system.strip()
        if full_system:
            full_system = f"{full_system}\n\n{MEMORY_RESPONSE_FORMAT}"
        else:
            full_system = MEMORY_RESPONSE_FORMAT

        if not history:
            yield await self.generate(prompt="", system=full_system)
            return

        buffer: list[str] = []
        stream_started = time.perf_counter()
        try:
            gen = await self._try_with_active_key(
                self._messages_from_history(history, full_system),
                is_stream=True,
                max_tokens=max_output_tokens,
            )
            logger.info(
                "%s TTFT: first token after %.2fs",
                self.provider_label,
                time.perf_counter() - stream_started,
            )
            async for chunk in gen:
                buffer.append(chunk)
                yield chunk
            self._record(call_site, len(str(history)), sum(len(c) for c in buffer))
        except Exception as exc:  # noqa: BLE001
            logger.exception("generate_with_history_stream failed: %s", exc)
            yield "I'm having trouble thinking right now. Could you try again?"

    async def generate_with_history_json(
        self,
        history: list[dict],
        system: str = "",
        hedge_width: int | None = None,
        call_site: str = "conversational",
    ) -> dict[str, Any]:
        from reasoning.llm_client import MEMORY_RESPONSE_FORMAT, parse_memory_response

        _ = hedge_width
        full_system = system.strip()
        if full_system:
            full_system = f"{full_system}\n\n{MEMORY_RESPONSE_FORMAT}"
        else:
            full_system = MEMORY_RESPONSE_FORMAT
        raw = await self.generate_with_history(
            history=history,
            system=full_system,
            call_site=call_site,
        )
        return parse_memory_response(raw)

    async def generate_json(
        self,
        prompt: str,
        schema_description: str | None = None,
        system: str = "",
        hedge_width: int | None = None,
        call_site: str = "planner",
    ) -> dict[str, Any]:
        from reasoning.llm_client import _strip_fences

        _ = hedge_width
        full_prompt = prompt
        if schema_description:
            full_prompt = (
                f"{prompt}\n\n"
                f"You MUST return valid JSON matching the following schema description:\n"
                f"{schema_description}\n"
                f"Ensure the response is wrapped in standard JSON format, with NO markdown formatting, no preamble, and no code blocks (do NOT use ```json)."
            )
        else:
            full_prompt = (
                f"{prompt}\n\n"
                f"You MUST return valid JSON with no preamble, and no markdown formatting (do NOT use ```json)."
            )
        if system:
            full_prompt = f"System Instruction: {system}\n\nUser Prompt: {full_prompt}"

        try:
            response = await self._try_with_active_key(
                self._messages(full_prompt, ""),
                is_stream=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("generate_json failed: %s", exc)
            return {}

        cleaned = _strip_fences(response.strip())
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as je:
            logger.warning(
                "JSON decode failed: %s. Raw response: %s",
                je,
                response[:400],
            )
            return {}
        self._record(call_site, len(full_prompt), len(response))
        return parsed if isinstance(parsed, dict) else {}

    async def health_check(self) -> bool:
        if self._lane is None:
            return False
        return any(e.is_available for e in self._lane.entries())

    def has_available_key(self) -> bool:
        if self._lane is None:
            return False
        return any(e.is_available for e in self._lane.entries())

    async def prewarm(self) -> None:
        """One tiny throwaway completion; failures swallowed by design."""
        if self._lane is None or not self.has_available_key():
            return
        try:
            await asyncio.wait_for(
                self._post_chat(
                    [{"role": "user", "content": "Reply with the single word: ready."}],
                    model=self.models[0],
                    max_tokens=1,
                ),
                timeout=min(8.0, self.timeout),
            )
            logger.debug("%s prewarm complete", self.provider_label)
        except Exception as exc:  # noqa: BLE001
            logger.debug("%s prewarm skipped (non-fatal): %s", self.provider_label, exc)
