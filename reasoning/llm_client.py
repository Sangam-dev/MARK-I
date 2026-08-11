from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator

from memory.token_log import TokenLog

from reasoning.llm_client_mulapi import (
    ALL_MODELS,
    get_pool,
    hedged_generate,
    hedged_generate_conv,
    hedged_stream_conv,

)

logger = logging.getLogger("kancha.reasoning.llm_client")

# Format the model must follow for conversational responses. Memory (SQL
# facts + RAG entries) is carried inline in the primary response — there
# is no separate extraction step anymore.
MEMORY_RESPONSE_FORMAT = """\
Respond with ONLY one JSON object, no markdown, no code fences, no preamble:

{
  "task": {
    "task_type": "...",
    "instruction": "...",
    "parameters": {},
    "expected_result": "...",
    "context": {},
    "follow_up": false
  },
  "message": "Your reply to the user",
  "sql": [{"key": "...", "value": "..."}],
  "rag": [{"type": "...", "title": "...", "content": "..."}]
}

Rules:
- "task" is OPTIONAL and delegates one unit of work to the execution
  layer. Include it ONLY when the request needs a tool, a system action,
  or live data. Omit it entirely for ordinary conversation. The task
  layer never sees the conversation — "instruction" must therefore be
  self-contained, with every reference ("it", "that", "the second one")
  already resolved into explicit values.

- When you include "task", it MUST be the FIRST key in the object,
  before "message". Ordinary replies start directly with "message".
  This ordering is load-bearing: it is how the assistant knows, while
  your reply is still being generated, whether it is about to start
  work — which decides whether your acknowledgement is spoken at all.

- "message" is REQUIRED and is the only text shown to the user.

- "sql" is OPTIONAL and holds STRUCTURED memory: short key/value facts.
  Use it for preferences, profile details, settings, relationships and
  other small stable values.
    Good: {"key": "favorite_language", "value": "Rust"}
          {"key": "current_project", "value": "AI Assistant"}
          {"key": "preferred_editor", "value": "Neovim"}
  Existing keys are updated in place; new keys are inserted. Never
  duplicate a key. Values must be short — a phrase, not a paragraph.

- "rag" is OPTIONAL and holds SEMANTIC memory: self-contained passages
  worth recalling. Use it for experiences, research,
  technical documentation, project progress, learning summaries, design
  decisions and debugging solutions , all of this are mandatory. 
  Each entry needs "type", "title"
  and "content", where "content" is a complete standalone paragraph
  that still makes sense with no surrounding conversation. 
    Good: {"type": "debugging", "title": "Fixed the WebSocket drop",
           "content": "The socket closed after 60s because the proxy
           idle timeout was shorter than the heartbeat interval.
           Lowering the heartbeat to 25s resolved it."}
  NEVER put greetings, chit-chat, one-off questions, temporary context,
  or restatements of the user's request in "rag".

- Choosing between them: if it fits in a short key/value pair it is
  "sql"; if it needs a paragraph to be useful later it is "rag". Most
  turns need neither.

- Omit "sql" and "rag" entirely when there is nothing to store.
  Never return empty arrays and never return null.

- Omit "task" entirely when no action is required. Never return null.
"""
# JSON string escapes that can appear inside the streamed ``message`` value.
# Decoding them the same way ``json.loads`` does keeps the streamed display
# text byte-identical to the authoritative parsed message — which lets the
# TTS overlap layer dedupe reliably (never re-speaking already-streamed text).
_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
    '"': '"',
    "\\": "\\",
    "/": "/",
}
_HEX4_RE = re.compile(r"[0-9a-fA-F]{4}")


_MSG_KEY_RE = re.compile(r'"message"\s*:\s*"')

def extract_streamed_message(buffer: str) -> str:
    """Best-effort extraction of the ``message`` field from a partially
    streamed JSON envelope.

    The model streams the full envelope (``{"message": ..., "sql"?: [...],
    "rag"?: [...]}``), but the frontend must only see the *message* text
    as it arrives. This incremental extractor walks the buffer, finds the
    ``"message": "`` key, and captures the JSON string value seen so far
    (handling ``\\`` escapes and stopping at the closing quote).

    Returns ``""`` until the message value has started. This is display-
    only — the authoritative parse still happens at the end via
    :func:`parse_memory_response` on the complete response.
    """
    if not buffer:
        return ""
    match = _MSG_KEY_RE.search(buffer)
    if not match:
        return ""

    i = match.end()
    out: list[str] = []
    while i < len(buffer):
        ch = buffer[i]
        if ch == "\\":
            if i + 1 >= len(buffer):
                break  # escape marker only — wait for the next chunk
            esc = buffer[i + 1]
            if esc == "u":
                hex_part = buffer[i + 2 : i + 6]
                if len(hex_part) < 4 or not _HEX4_RE.fullmatch(hex_part):
                    break  # \uXXXX incomplete — wait for the rest
                out.append(chr(int(hex_part, 16)))
                i += 6
                continue
            out.append(_ESCAPES.get(esc, esc))
            i += 2
            continue
        if ch == '"':
            break
        out.append(ch)
        i += 1
    return "".join(out)


def parse_memory_response(raw: str) -> dict[str, Any]:
    """Parse the model's JSON envelope into ``{message, task?, sql?, rag?}``.

    Tolerant parser — never raises. Falls back to ``{"message": <raw>}``
    when the model returns plain text or malformed JSON. ``task``/``sql``/
    ``rag`` are omitted (never empty, never null) so callers can safely
    test ``if "task" in response``.

    ``task`` is the Conversation LLM's delegation to the Task LLM; the
    coordinator turns it into a :class:`core.events.TaskRequested`. It is
    only surfaced when it is a dict carrying something actionable — an
    ``instruction`` or a ``task_type`` — so a hallucinated empty object
    can never trigger execution.
    """
    text = (raw or "").strip()
    if not text:
        return {"message": ""}

    cleaned = text
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"message": text}

    if not isinstance(data, dict):
        return {"message": text}

    message = data.get("message")
    if not isinstance(message, str):
        message = text

    payload: dict[str, Any] = {"message": message}

    task = data.get("task")
    if isinstance(task, dict) and (
        str(task.get("instruction") or "").strip()
        or str(task.get("task_type") or "").strip()
    ):
        payload["task"] = task

    sql = data.get("sql")
    if isinstance(sql, list) and sql:
        payload["sql"] = sql

    rag = data.get("rag")
    if isinstance(rag, list) and rag:
        payload["rag"] = rag

    return payload


class GeminiClient:
    """Gemini Client wrapping the multi-API key rotation and hedged generation logic."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        timeout: float = 12.0,
        default_hedge_width: int = 1,
        token_log: TokenLog | None = None,
    ) -> None:
        self.model = model or ALL_MODELS[0]
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_hedge_width = default_hedge_width
        self.token_log = token_log
        self.pool = None

    async def initialize(self) -> None:
        """Initialize the client by loading the key pool."""
        self.pool = get_pool()
        logger.info("GeminiClient initialized successfully.")

    async def generate(
        self,
        prompt: str,
        system: str = "",
        hedge_width: int | None = None,
        call_site: str = "conversational",
    ) -> str:
        """Generate a response from the model, optionally injecting system instructions."""
        full_prompt = prompt
        if system:
            full_prompt = f"System Instruction: {system}\n\nUser Prompt: {prompt}"

        if self.pool is None:
            raise RuntimeError("GeminiClient is not initialized")

        try:
            response = await hedged_generate(
                pool=self.pool,
                models=ALL_MODELS,
                prompt=full_prompt,
                hedge_width=(
                    hedge_width if hedge_width is not None else self.default_hedge_width
                ),
                timeout=self.timeout,
            )
            if self.token_log is not None:
                self.token_log.record(
                    call_site=call_site,
                    model=self.model,
                    input_tokens=max(1, len(full_prompt) // 4),
                    output_tokens=max(1, len(response) // 4),
                )
            return response
        except Exception as e:
            logger.exception("Failed to generate response: %s", e)
            return "I'm having trouble thinking right now. Could you try again?"

    async def generate_with_history(
        self,
        history: list[dict],
        system: str = "",
        hedge_width: int | None = None,
        call_site: str = "conversational",
    ) -> str:
        """
        Generate a response using proper Gemini multi-turn conversation format.

        Args:
            history: Ordered list of conversation turns.
                     Each dict must have "role" ("user" or "assistant") and "content".
                     The LAST item must have role "user" — it is the current user message.
            system:  Optional system instruction (passed via GenerateContentConfig,
                     not embedded in the prompt string).

        Returns:
            The assistant's response text.
        """
        from google.genai import types

        if not history:
            return await self.generate(prompt="", system=system)

        # Convert to Gemini native Content objects.
        # Gemini uses "model" for assistant turns (not "assistant").
        contents = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])],
                )
            )

        # Ensure conversation ends with a user turn (Gemini requirement).
        if not contents or contents[-1].role != "user":
            logger.warning(
                "generate_with_history: last turn is not 'user' — appending empty user turn"
            )
            contents.append(types.Content(role="user", parts=[types.Part(text="")]))

        config = None
        if system:
            config = types.GenerateContentConfig(system_instruction=system)

        if self.pool is None:
            raise RuntimeError("GeminiClient is not initialized")

        try:
            response = await hedged_generate_conv(
                pool=self.pool,
                models=ALL_MODELS,
                contents=contents,
                config=config,
                hedge_width=(
                    hedge_width if hedge_width is not None else self.default_hedge_width
                ),
                timeout=self.timeout,
            )
            if self.token_log is not None:
                self.token_log.record(
                    call_site=call_site,
                    model=self.model,
                    input_tokens=max(1, len(str(history)) // 4),
                    output_tokens=max(1, len(response) // 4),
                )
            return response
        except Exception as e:
            logger.exception("generate_with_history failed: %s", e)
            return "I'm having trouble thinking right now. Could you try again?"

    async def generate_with_history_stream(
        self,
        history: list[dict],
        system: str = "",
        hedge_width: int | None = None,
        call_site: str = "conversational",
    ) -> AsyncIterator[str]:
        """Stream a conversational response as text deltas.

        Augments *system* with :data:`MEMORY_RESPONSE_FORMAT` exactly like
        :meth:`generate_with_history_json`, then streams the raw token
        deltas from the model (the JSON envelope, token by token). The
        caller is responsible for extracting the ``message`` field via
        :func:`extract_streamed_message` while it streams and parsing the
        full envelope at the end via :func:`parse_memory_response`.

        On failure, yields a single fallback message (never raises), so the
        conversational path always produces something for the user.
        """
        full_system = system.strip()
        if full_system:
            full_system = f"{full_system}\n\n{MEMORY_RESPONSE_FORMAT}"
        else:
            full_system = MEMORY_RESPONSE_FORMAT

        from google.genai import types

        if not history:
            # No history: fall back to a plain (non-stream) call so we still
            # return something; rare on the conversational path.
            yield await self.generate(prompt="", system=full_system)
            return

        # Convert to Gemini native Content objects (same as generate_with_history).
        contents = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])],
                )
            )
        if not contents or contents[-1].role != "user":
            contents.append(types.Content(role="user", parts=[types.Part(text="")]))

        config = (
            types.GenerateContentConfig(system_instruction=full_system)
            if full_system
            else None
        )

        if self.pool is None:
            raise RuntimeError("GeminiClient is not initialized")

        buffer: list[str] = []
        try:
            async for chunk in hedged_stream_conv(
                pool=self.pool,
                models=ALL_MODELS,
                contents=contents,
                config=config,
                hedge_width=(
                    hedge_width if hedge_width is not None else self.default_hedge_width
                ),
                timeout=self.timeout,
            ):
                buffer.append(chunk)
                yield chunk

            if self.token_log is not None:
                self.token_log.record(
                    call_site=call_site,
                    model=self.model,
                    input_tokens=max(1, len(str(history)) // 4),
                    output_tokens=max(1, sum(len(c) for c in buffer) // 4),
                )
        except Exception as exc:
            logger.exception("generate_with_history_stream failed: %s", exc)
            yield "I'm having trouble thinking right now. Could you try again?"

    async def generate_with_history_json(
        self,
        history: list[dict],
        system: str = "",
        hedge_width: int | None = None,
        call_site: str = "conversational",
    ) -> dict[str, Any]:
        """Generate a conversational response and parse the JSON envelope.

        Augments *system* with :data:`MEMORY_RESPONSE_FORMAT` so the
        model returns ``{"message": ..., "sql"?: [...], "rag"?: [...]}``.
        Returns the parsed dict — ``sql``/``rag`` keys are absent when
        there is nothing to store.
        """
        full_system = system.strip()
        if full_system:
            full_system = f"{full_system}\n\n{MEMORY_RESPONSE_FORMAT}"
        else:
            full_system = MEMORY_RESPONSE_FORMAT

        raw = await self.generate_with_history(
            history=history,
            system=full_system,
            hedge_width=hedge_width,
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
        """Generate a JSON response from the model, retrying on decode errors."""
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

        if self.pool is None:
            raise RuntimeError("GeminiClient is not initialized")

        for attempt in range(self.max_retries + 1):
            try:
                response = await hedged_generate(
                    pool=self.pool,
                    models=ALL_MODELS,
                    prompt=full_prompt,
                    hedge_width=(
                        hedge_width if hedge_width is not None else self.default_hedge_width
                    ),
                    timeout=self.timeout,
                )
                if self.token_log is not None:
                    self.token_log.record(
                        call_site=call_site,
                        model=self.model,
                        input_tokens=max(1, len(full_prompt) // 4),
                        output_tokens=max(1, len(response) // 4),
                    )
                cleaned = response.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    cleaned = "\n".join(lines).strip()

                return json.loads(cleaned)
            except json.JSONDecodeError as je:
                logger.warning(
                    "JSON decode failed (attempt %d/%d): %s. Raw response: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    je,
                    response,
                )
                if attempt == self.max_retries:
                    logger.exception("Failed to generate valid JSON after retries.")
                    return {}
            except Exception as e:
                logger.exception("Error during generate_json: %s", e)
                return {}
        return {}

    async def health_check(self) -> bool:
        """Return True if the key pool has available keys."""
        if not self.pool:
            return False
        return any(e.is_available for e in self.pool._entries)
