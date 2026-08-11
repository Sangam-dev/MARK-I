"""Natural-language pass for plan outcomes.

Sits between the Scheduler's :class:`PlanCompleted` event and the
``ResponseReady`` that the user sees / hears. Rewrites the raw,
machine-style task results into one short, JARVIS-style reply.

Three paths, cheapest first
--------------------------
- **Short, already-conversational output** — the per-tool voice from
  :mod:`reasoning.tool_voice` and nothing else. Zero cost, instant.
  "Firefox is open, sir." needs no help.
- **Anything machine-shaped** — a listing, a process table, an email
  with a hex id, a multi-line result: handed to :mod:`reasoning.groq_voice`
  to be said the way a person would say it. This is the path that stops
  the assistant reading ``id=19ff02e7eaf03a37`` out loud.
- **Fallback** — no Groq key, a timeout, an error: the deterministic
  text goes out unchanged. Blunter, never broken.

Gemini is deliberately not used here. This pass runs on every completed
task, and the Gemini keys are the scarce resource; Groq's 8b model is
fast enough to sit in the response path and cheap enough not to matter.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from memory.token_log import TokenLog
from reasoning.groq_voice import GroqVoice, get_shared_voice, needs_naturalizing
from reasoning.llm_client import GeminiClient
from reasoning.tool_voice import naturalize_single_tool

logger = logging.getLogger("kancha.reasoning.naturalize")


async def naturalize_plan_response(
    llm: GeminiClient,
    user_request: str,
    task_results: list[dict[str, Any]],
    status: str,
    token_log: TokenLog | None = None,
    voice: GroqVoice | None = None,
) -> str:
    """Return a user-facing reply for a finished plan.

    Parameters
    ----------
    llm:
        Kept for signature compatibility; the phrasing pass runs on Groq
        (see :mod:`reasoning.groq_voice`), not on the Gemini keys.
    user_request:
        Original user input. Passed to the voice so it knows what was
        actually asked — "how many unread?" and "read me the latest"
        want very different amounts of the same tool output.
    task_results:
        Per-task result entries from :class:`PlanCompleted.task_results`.
        Each dict has ``{"tool": str, "result": str}``. Failures are
        passed through with their error text in ``result``.
    status:
        One of ``"completed" | "partial" | "failed" | "cancelled"``.
    voice:
        Override for tests. Defaults to the shared Groq voice.
    """
    _ = (llm, token_log)

    if not task_results:
        return _status_fallback(status)

    if status == "cancelled":
        return "Cancelled, sir."

    # Deterministic pass first: it knows each tool's output shape better
    # than a general model does, and it gives the voice cleaner input.
    rendered = _render(task_results)
    if not rendered.strip():
        return _status_fallback(status)

    # Short, conversational, single-task output goes straight out. This
    # is most tasks, and it stays free and instant.
    if (
        len(task_results) == 1
        and status == "completed"
        and not needs_naturalizing(rendered)
    ):
        return rendered

    speaker = voice if voice is not None else get_shared_voice()
    return await speaker.speak(
        user_request=user_request,
        tool_output=_raw_fallback(task_results, status),
        status=status,
        fallback=rendered,
    )


def _render(task_results: list[dict[str, Any]]) -> str:
    """Apply the per-tool voice to every result and join them."""
    parts: list[str] = []
    for entry in task_results:
        message = str(entry.get("result", "")).strip()
        if not message:
            continue
        parts.append(
            naturalize_single_tool(
                str(entry.get("tool", "")), message, entry.get("arguments") or {}
            )
        )
    if len(parts) == 2:
        combined = _join_two(parts[0], parts[1])
        if combined is not None:
            return combined
    return "\n".join(parts)


def _join_two(first: str, second: str) -> str | None:
    """Deterministically combine two compact results into one sentence."""
    if not first or not second:
        return None
    if len(first) > 500 or len(second) > 500:
        return None
    return f"{first} And {second[0].lower() + second[1:]}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw_fallback(task_results: list[dict[str, Any]], status: str) -> str:
    """Concatenate raw results when the LLM path isn't available."""
    parts: list[str] = []
    for entry in task_results:
        message = str(entry.get("result", "")).strip()
        if message:
            parts.append(message)
    if not parts:
        return _status_fallback(status)
    return "\n".join(parts)


def _status_fallback(status: str) -> str:
    if status == "completed":
        return "Done, sir."
    if status == "partial":
        return "Some steps didn't complete, sir."
    if status == "failed":
        return "I couldn't complete that, sir."
    if status == "cancelled":
        return "Cancelled, sir."
    return f"Plan finished ({status})."