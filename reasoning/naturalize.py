"""Natural-language pass for plan outcomes.

Sits between the Scheduler's :class:`PlanCompleted` event and the
``ResponseReady`` that the user sees / hears. Rewrites the raw,
machine-style task results into one short, JARVIS-style reply.

Two paths
---------
- **Single task, completed** — runs the per-tool voice function from
  :mod:`reasoning.tool_voice` synchronously. Zero LLM cost. This is
  the common case ("open firefox", "what's the weather in London?").
- **Multi-task or any failure** — sends the tool results to the LLM
  with the JARVIS persona and asks for one short reply. On any error
  the function falls back to the raw joined summary so the user still
  sees something.

This module is intentionally small: ~80 lines, no I/O besides the
optional LLM call.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from memory.token_log import TokenLog
from reasoning.llm_client import GeminiClient
from reasoning.prompt import persona
from reasoning.tool_voice import naturalize_single_tool

logger = logging.getLogger("kancha.reasoning.naturalize")


# Per-call timeout for the LLM paraphrase. Lower than the GeminiClient
# default (12s) so a slow LLM doesn't hold up the response path —
# we always fall back to the raw summary.
_LLM_TIMEOUT_S = 4.0


async def naturalize_plan_response(
    llm: GeminiClient,
    user_request: str,
    task_results: list[dict[str, Any]],
    status: str,
    token_log: TokenLog | None = None,
) -> str:
    """Return a user-facing reply for a finished plan.

    Parameters
    ----------
    llm:
        The shared GeminiClient. Used only when there are ≥2 tasks or
        any failures.
    user_request:
        Original user input — included in the LLM prompt for context.
    task_results:
        Per-task result entries from :class:`PlanCompleted.task_results`.
        Each dict has ``{"tool": str, "result": str}``. Failures are
        passed through with their error text in ``result``.
    status:
        One of ``"completed" | "partial" | "failed" | "cancelled"``.
    """
    if not task_results:
        return _status_fallback(status)

    # Fast path: single successful task — apply the per-tool voice and
    # ship it. No LLM call.
    if len(task_results) == 1 and status == "completed":
        entry = task_results[0]
        tool = str(entry.get("tool", ""))
        message = str(entry.get("result", ""))
        arguments = entry.get("arguments") or {}
        return naturalize_single_tool(tool, message, arguments)

    if status == "cancelled":
        return "Cancelled, sir."

    if len(task_results) == 2 and status == "completed":
        combined = _naturalize_two_tasks(task_results[0], task_results[1])
        if combined is not None:
            return combined

    # Multi-task or failure path — ask the LLM to combine.
    try:
        return await asyncio.wait_for(
            _ask_llm(llm, user_request, task_results, status, token_log),
            timeout=_LLM_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning("naturalize_plan_response: LLM timeout, using raw summary")
        return _raw_fallback(task_results, status)
    except Exception as exc:  # noqa: BLE001
        logger.warning("naturalize_plan_response failed: %s — using raw summary", exc)
        return _raw_fallback(task_results, status)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _ask_llm(
    llm: GeminiClient,
    user_request: str,
    task_results: list[dict[str, Any]],
    status: str,
    token_log: TokenLog | None = None,
) -> str:
    """Call the LLM with a JARVIS-voiced rewrite prompt."""
    lines = []
    for entry in task_results:
        tool = str(entry.get("tool", ""))
        result = str(entry.get("result", "")).strip()
        if not result:
            continue
        # Pre-naturalize each line locally — gives the LLM a cleaner
        # input to work with and saves tokens.
        arguments = entry.get("arguments") or {}
        natural = naturalize_single_tool(tool, result, arguments)
        lines.append(f"- {tool}: {natural}")
    if not lines:
        return _raw_fallback(task_results, status)

    body = (
        f"The user asked: \"{user_request.strip()}\"\n\n"
        f"Tool results (in order):\n"
        + "\n".join(lines)
        + f"\n\nStatus: {status}\n\n"
        "Reply with one short natural-language message, in your JARVIS "
        "voice, that combines these results for the user. If only one "
        "result is present, return it unchanged (lightly rephrased only "
        "if it improves flow). No bullet points, no status counters, no "
        "markdown."
    )

    response = await llm.generate(
        prompt=body,
        system=persona.strip(),
        hedge_width=1,
        call_site="naturalize",
    )
    return (response or "").strip() or _raw_fallback(task_results, status)


def _naturalize_two_tasks(first: dict[str, Any], second: dict[str, Any]) -> str | None:
    """Deterministically combine two completed tasks when the results are compact."""
    first_text = naturalize_single_tool(
        str(first.get("tool", "")),
        str(first.get("result", "")),
        first.get("arguments") or {},
    )
    second_text = naturalize_single_tool(
        str(second.get("tool", "")),
        str(second.get("result", "")),
        second.get("arguments") or {},
    )
    if not first_text or not second_text:
        return None
    if len(first_text) > 500 or len(second_text) > 500:
        return None
    return f"{first_text} And {second_text[0].lower() + second_text[1:]}"


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