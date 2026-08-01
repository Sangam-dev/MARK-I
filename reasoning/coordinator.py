from __future__ import annotations

import logging
import re
from typing import Any
import asyncio
from core.bus import EventBus
from core.events import (
    Intent,
    IntentIdentified,
    MemoryRetrieved,
    PartialResponse,
    PlanCompleted,
    ReasoningRequested,
    ResponseReady,
    TaskCompleted,
)
from memory.manager import MemoryManager
from memory.token_log import TokenLog
from reasoning.llm_client import (
    GeminiClient,
    extract_streamed_message,
    parse_memory_response,
)
from reasoning.naturalize import naturalize_plan_response
from reasoning.prompt_builder import JARVIS_PERSONA

logger = logging.getLogger("kancha.reasoning.coordinator")

_RETRY_RE = re.compile(
    r"^\s*(?:redo(?: it)?|retry(?: it)?|try again|do it again|run it again|"
    r"execute it again|again)\s*[.!?]*\s*$",
    re.IGNORECASE,
)


class ReasoningCoordinator:
    """
    Orchestrates context retrieval, LLM prompt construction, tool dispatch,
    and response generation — all through the EventBus.

    Conversation history ownership
    --------------------------------
    The coordinator owns the short-term in-memory buffer directly.
    It adds user/assistant turns synchronously so the context is
    always up-to-date before the LLM call.  Durable memory (SQL facts
    and RAG entries) is carried inline by the primary Gemini response
    JSON envelope and persisted here via MemoryManager.save_sql /
    append_rag — there is no separate extraction step anymore.
    """

    def __init__(
        self,
        bus: EventBus,
        gemini_client: GeminiClient,
        memory_manager: MemoryManager,
        token_log: TokenLog | None = None,
    ) -> None:
        self.bus = bus
        self.gemini_client = gemini_client
        self.memory_manager = memory_manager
        self.token_log = token_log
        # Pending task turns keyed by session_id
        self._pending_turns: dict[str, dict[str, Any]] = {}
        self._last_tasks: dict[str, dict[str, Any]] = {}

    def register(self) -> None:
        self.bus.subscribe(ReasoningRequested, self.on_reasoning_requested)
        self.bus.subscribe(TaskCompleted, self.on_task_completed)
        self.bus.subscribe(PlanCompleted, self.on_plan_completed)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_system_prompt(self, memory_event: MemoryRetrieved | None) -> str:
        """
        Build the system prompt: KANCHA persona + explicit user facts only.

        Conversation history is passed separately as a message list —
        it must NEVER appear here as "structured facts", which confuses
        the LLM into anchoring on old turns. RAG/vector context is disabled.
        """
        parts = [JARVIS_PERSONA]

        if memory_event:
            # Only inject items that have a key+value structure — these are
            # explicit user facts (e.g. name, language preference).
            # Recent interaction rows from SQLite (which have only "content")
            # are intentionally excluded: they belong in the message list.
            fact_lines = [
                f"- {item['key']}: {item['value']}"
                for item in memory_event.structured_context
                if "key" in item and "value" in item
            ]
            if fact_lines:
                parts.append("User facts:\n" + "\n".join(fact_lines))

        return "\n\n".join(parts)

    def _get_history(self) -> list[dict[str, str]]:
        """Return short_term buffer as a clean [{role, content}] list."""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.memory_manager.short_term.get_recent()
        ]

    async def _persist_response_memory(self, payload: dict[str, Any]) -> None:
        """Persist sql/rag memory carried by the primary Gemini response.

        Best-effort: a failure to persist must never break the reply.
        Absent keys ("sql"/"rag") are simply skipped.
        """
        sql = payload.get("sql")
        if sql:
            try:
                saved = await self.memory_manager.save_sql(sql)
                if saved:
                    logger.info("Persisted %d SQL fact(s) from response", saved)
            except Exception as exc:
                logger.warning("SQL memory persist failed (non-fatal): %s", exc)

        rag = payload.get("rag")
        if rag:
            try:
                appended = await self.memory_manager.append_rag(rag)
                if appended:
                    logger.info("Appended %d RAG entries to rag.txt", appended)
            except Exception as exc:
                logger.warning("RAG memory append failed (non-fatal): %s", exc)

    def _is_retry_request(self, user_input: str) -> bool:
        return bool(_RETRY_RE.match(user_input))

    def _format_task_response(self, event: TaskCompleted) -> str:
        """Return a factual response based only on the executor result."""
        details = (event.result if event.success else event.error).strip()
        if event.success:
            if details:
                return details
            return f"The '{event.task_name}' action completed successfully."

        if details:
            return f"I couldn't complete '{event.task_name}': {details}"
        return f"I couldn't complete '{event.task_name}'."

    # ── Main handler ──────────────────────────────────────────────────────────

    async def on_reasoning_requested(self, event: ReasoningRequested) -> None:
        session_id = event.session_id
        intent_event = event.intent_event
        user_input = intent_event.raw_input

        # ── Step 1: Add user turn to short_term SYNCHRONOUSLY ─────────────
        # This must happen BEFORE get_recent() so the LLM sees the current
        # message as part of the conversation (not just appended externally).
        self.memory_manager.short_term.add("user", user_input)

        # ── Step 2: Task path — defer to Planner ────────────────────────
        # The Planner (planning/planner.py) is already subscribed to
        # IntentIdentified and will decompose this request into an
        # ExecutionPlan, dispatch it via the PlanScheduler, and emit
        # PlanCompleted when it's done. We just record the user input
        # for memory; we do NOT emit TaskExecutionRequested ourselves.
        if intent_event.requires_task:
            logger.info(
                "Task intent detected: %s  params=%s — handed off to Planner",
                intent_event.task_type,
                intent_event.task_params,
            )
            self._last_tasks[session_id] = {
                "task_type": intent_event.task_type,
                "task_params": dict(intent_event.task_params),
                "user_input": user_input,
                "memory_event": (
                    event.memory_events[0] if event.memory_events else None
                ),
            }
            return

        # ── Step 3b: Retry path — re-plan from the last user request ─────
        # "Do it again" used to re-emit a single TaskExecutionRequested.
        # We now treat it as a fresh request to the Planner: the user
        # wants the same plan re-run. The Planner decides whether to
        # one-shot it (single task) or decompose.
        if self._is_retry_request(user_input):
            last_task = self._last_tasks.get(session_id)
            if last_task is None:
                response_text = "I don't have a previous action to retry."
                self.memory_manager.short_term.add("assistant", response_text)
                self.bus.emit(ResponseReady(text=response_text, session_id=session_id))
                return

            # Rebuild a synthetic IntentIdentified from the last known
            # task so the Planner's existing subscription picks it up.
            retry_intent = IntentIdentified(
                intent=Intent.TASK,
                raw_input=last_task.get("user_input", user_input),
                confidence=1.0,
                session_id=session_id,
                requires_task=True,
                task_type=last_task.get("task_type") or "",
                task_params=last_task.get("task_params") or {},
            )
            logger.info(
                "Retrying via Planner: %s params=%s",
                retry_intent.task_type,
                retry_intent.task_params,
            )
            self.bus.emit(retry_intent)
            return

        # ── Step 3: Conversational path — stream LLM (JSON envelope) ──────
        memory_event = event.memory_events[0] if event.memory_events else None
        system = self._build_system_prompt(memory_event)

        # history already ends with the user's current message (added in step 1)
        history = self._get_history()

        logger.debug(
            "Calling LLM (streaming) | history_turns=%d | system_len=%d",
            len(history),
            len(system),
        )

        # Stream the JSON envelope token-by-token. While it arrives, strip
        # the scaffolding and forward ONLY the visible `message` text as
        # PartialResponse events so the UI renders words as they generate.
        # The full buffer is parsed once at the end for the authoritative
        # message + sql/rag persistence (see _persist_response_memory).
        raw_parts: list[str] = []
        buffer = ""
        prev_visible = ""
        try:
            async for chunk in self.gemini_client.generate_with_history_stream(
                history=history,
                system=system,
                hedge_width=1,
                call_site="conversational",
            ):
                raw_parts.append(chunk)
                buffer += chunk
                visible = extract_streamed_message(buffer)
                if len(visible) > len(prev_visible):
                    self.bus.emit(
                        PartialResponse(
                            text=visible[len(prev_visible) :],
                            session_id=session_id,
                        )
                    )
                    prev_visible = visible
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Streaming LLM call failed (session=%s): %s", session_id, exc
            )

        raw_response = "".join(raw_parts)
        payload = parse_memory_response(raw_response)
        response_text = payload.get("message", "").strip()
        # Guard against the raw-JSON fallback: if the stream died mid-way,
        # parse_memory_response returns {"message": <raw partial JSON>}.
        # Never surface scaffolding — fall back to what was already shown.
        if (
            not response_text
            or response_text == raw_response.strip()
            or response_text.startswith("{")
        ):
            response_text = prev_visible

        # ── Step 4: Add assistant turn to short_term SYNCHRONOUSLY ────────
        self.memory_manager.short_term.add("assistant", response_text)

        # ── Step 5: Persist memory from the primary Gemini response ───────
        # Memory now comes directly from the response JSON envelope — no
        # separate extraction step after the conversation.
        await self._persist_response_memory(payload)

        # ── Step 6: Mark stream done + emit final response ────────────────
        self.bus.emit(PartialResponse(text="", done=True, session_id=session_id))
        self.bus.emit(ResponseReady(text=response_text, session_id=session_id))

    # ── Task completion handler ───────────────────────────────────────────────

    async def on_task_completed(self, event: TaskCompleted) -> None:
        session_id = event.session_id
        pending = self._pending_turns.pop(session_id, None)

        if not pending:
            logger.warning(
                "TaskCompleted for session %s but no pending turn found — ignoring.",
                session_id,
            )
            return

        response_text = self._format_task_response(event)

        # Add assistant response to short_term.
        self.memory_manager.short_term.add("assistant", response_text)

        self.bus.emit(ResponseReady(text=response_text, session_id=session_id))

    async def on_plan_completed(self, event: PlanCompleted) -> None:
        """Emit a user-facing response when a plan finishes.

        Single-task plans run through :mod:`reasoning.tool_voice`
        synchronously (zero LLM cost). Multi-task and failure paths go
        through :mod:`reasoning.naturalize` which calls the LLM to
        combine the per-tool results into one JARVIS-style reply.
        """
        session_id = event.session_id
        summary = (event.summary or "").strip()

        response_text = await naturalize_plan_response(
            llm=self.gemini_client,
            user_request=event.user_request or summary,
            task_results=list(event.task_results or []),
            status=event.status,
            token_log=self.token_log,
        )

        self.memory_manager.short_term.add("assistant", response_text)

        self.bus.emit(ResponseReady(text=response_text, session_id=session_id))
