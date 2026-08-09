from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from core.audio_state import audio_state
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
from memory.rag import MemoryRouter, RAGManager, RetrievedChunk
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

    RAG ownership
    -------------
    This class is the **Conversation Manager** in the RAG architecture.
    It never touches the vector database: retrieval goes through
    :class:`~memory.rag.manager.RAGManager`, gated by a
    :class:`~memory.rag.router.MemoryRouter`. Conversely the RAG Manager
    never builds prompts — turning retrieved chunks into prompt text is
    :meth:`_format_rag_context`, and it lives here on purpose.
    """

    def __init__(
        self,
        bus: EventBus,
        gemini_client: GeminiClient,
        memory_manager: MemoryManager,
        token_log: TokenLog | None = None,
        rag_manager: RAGManager | None = None,
        rag_router: MemoryRouter | None = None,
    ) -> None:
        self.bus = bus
        self.gemini_client = gemini_client
        self.memory_manager = memory_manager
        self.token_log = token_log
        # Both None when RAG is disabled or failed to start. Every use
        # site null-checks, so the conversational path is unchanged in
        # that case.
        self.rag_manager = rag_manager
        self.rag_router = rag_router
        # Pending task turns keyed by session_id
        self._pending_turns: dict[str, dict[str, Any]] = {}
        self._last_tasks: dict[str, dict[str, Any]] = {}
        # Sessions whose thinking gate is held and will be released by a
        # terminal handler (on_task_completed or on_plan_completed)
        # rather than by on_reasoning_requested itself. The conversational
        # branch releases inline; the task/retry branches add here.
        self._gated_session_ids: set[str] = set()

    def register(self) -> None:
        self.bus.subscribe(ReasoningRequested, self.on_reasoning_requested)
        self.bus.subscribe(TaskCompleted, self.on_task_completed)
        self.bus.subscribe(PlanCompleted, self.on_plan_completed)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_system_prompt(
        self,
        memory_event: MemoryRetrieved | None,
        retrieved: list[RetrievedChunk] | None = None,
    ) -> str:
        """
        Build the system prompt: persona + structured facts + retrieved memory.

        Three distinct layers, deliberately kept apart:

        * **persona** — who the assistant is.
        * **User facts** — structured key/value memory from SQLite.
        * **Relevant long-term memory** — semantic chunks from the vector
          store, present only when the Memory Router asked for retrieval
          and something scored above the similarity threshold.

        Conversation history is passed separately as a message list —
        it must NEVER appear here as "structured facts", which confuses
        the LLM into anchoring on old turns.
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

        rag_block = self._format_rag_context(retrieved)
        if rag_block:
            parts.append(rag_block)

        return "\n\n".join(parts)

    def _format_rag_context(self, retrieved: list[RetrievedChunk] | None) -> str:
        """Render retrieved chunks into a prompt block.

        This is the Conversation Manager's job, not the RAG Manager's —
        which is why it takes structured :class:`RetrievedChunk` objects
        and produces the string, rather than receiving a pre-baked
        prompt fragment.

        The block is budgeted by ``KANCHA_RAG_MAX_CONTEXT_CHARS``.
        Chunks arrive sorted by score, so truncation always drops the
        least relevant material first.
        """
        if not retrieved:
            return ""

        budget = 4000
        if self.rag_manager is not None:
            budget = self.rag_manager.config.max_context_chars

        lines: list[str] = []
        used = 0
        included = 0

        for index, chunk in enumerate(retrieved, start=1):
            source = f", source: {chunk.source}" if chunk.source else ""
            page = chunk.metadata.get("page")
            page_note = f", page {page}" if page else ""
            header = (
                f"[{index}] {chunk.title} "
                f"({chunk.doc_type}, relevance {chunk.score:.2f}{source}{page_note})"
            )
            body = chunk.content.strip()
            entry = f"{header}\n{body}"

            if used + len(entry) > budget:
                if included == 0:
                    # Always include at least one chunk, truncated to fit,
                    # rather than retrieving something and then showing
                    # the model nothing.
                    entry = entry[:budget].rstrip() + " […]"
                    lines.append(entry)
                    included += 1
                break

            lines.append(entry)
            used += len(entry)
            included += 1

        if not lines:
            return ""

        if included < len(retrieved):
            logger.debug(
                "RAG context truncated to %d/%d chunk(s) by the %d-char budget",
                included,
                len(retrieved),
                budget,
            )

        return (
            "Relevant long-term memory (retrieved for this message):\n\n"
            + "\n\n".join(lines)
            + "\n\nUse this material only where it genuinely answers the user. "
            "Ignore anything irrelevant, and never mention that you looked "
            "anything up or refer to these numbered entries."
        )

    async def _retrieve_rag_context(
        self, user_input: str, intent_event: IntentIdentified | None
    ) -> list[RetrievedChunk]:
        """Ask the router whether to retrieve, then retrieve if so.

        Fully guarded: retrieval enriches a reply, so any failure here
        must degrade the answer rather than break the turn.
        """
        if self.rag_manager is None or self.rag_router is None:
            return []

        try:
            decision = await self.rag_router.decide(
                user_input,
                intent=intent_event.intent.value if intent_event else None,
                is_task=bool(intent_event.requires_task) if intent_event else False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory router failed (%s) — skipping retrieval", exc)
            return []

        if not decision.retrieve:
            logger.debug("RAG skipped: %s", decision.reason)
            return []

        logger.info("RAG retrieval triggered: %s", decision.reason)
        try:
            return await self.rag_manager.retrieve(decision.query or user_input)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG retrieval failed (%s) — answering without it", exc)
            return []

    def _get_history(self) -> list[dict[str, str]]:
        """Return short_term buffer as a clean [{role, content}] list."""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.memory_manager.short_term.get_recent()
        ]

    async def _persist_response_memory(
        self, payload: dict[str, Any], session_id: str = "default"
    ) -> None:
        """Persist sql/rag memory carried by the primary Gemini response.

        Best-effort: a failure to persist must never break the reply.
        Absent keys ("sql"/"rag") are simply skipped.

        ``rag`` entries go to two places with different standing:

        * the **vector database** — the authoritative semantic store,
          written through the RAG Manager so chunking, embedding and
          duplicate detection all apply;
        * ``memory/rag.txt`` — a human-readable audit log, for
          inspection and debugging only. Nothing ever reads it back.
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
        if not rag:
            return

        # 1. Authoritative store.
        if self.rag_manager is not None:
            try:
                results = await self.rag_manager.index_conversation_entries(
                    rag, session_id=session_id
                )
                indexed = sum(r.chunks_indexed for r in results)
                if indexed:
                    logger.info(
                        "Indexed %d RAG chunk(s) from response into the vector store",
                        indexed,
                    )
            except Exception as exc:
                logger.warning("RAG vector index failed (non-fatal): %s", exc)
        else:
            logger.debug("RAG disabled — response rag entries go to the audit log only")

        # 2. Audit log (unchanged behaviour).
        try:
            appended = await self.memory_manager.append_rag(rag)
            if appended:
                logger.info("Appended %d RAG entries to rag.txt (audit log)", appended)
        except Exception as exc:
            logger.warning("RAG audit append failed (non-fatal): %s", exc)

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
        # Defensive: if NLU was bypassed (e.g. direct bus.emit from a
        # caller that didn't go through _process_text) and the gate is
        # not yet held, claim it so the mic can't reopen mid-think.
        gate_held = audio_state.thinking_active.is_set()
        if not gate_held:
            audio_state.thinking_started()

        try:
            return await self._handle_reasoning(event)
        finally:
            # The conversational branch emits ResponseReady itself, but
            # the task/retry branches hand off and the response for
            # those paths arrives later via on_task_completed /
            # on_plan_completed. Releasing here would let the mic
            # reopen during task/plan execution — wrong. Instead we
            # release on every branch's *terminal* handler. So this
            # block is intentionally empty unless an exception escaped.
            if gate_held:
                # We didn't claim it ourselves; don't release it.
                pass

    async def _handle_reasoning(self, event: ReasoningRequested) -> None:
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
            # Task path: keep the thinking gate raised until either
            # on_task_completed or on_plan_completed fires (they release
            # it). If the planner doesn't pick this up at all, the gate
            # would otherwise stay held forever — log a warning so the
            # deadlock is visible.
            self._gated_session_ids.add(session_id)
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
                # No Planner handoff; release the gate here.
                audio_state.thinking_finished()
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
            # Same gated-release contract as the task path above.
            self._gated_session_ids.add(session_id)
            return

        # ── Step 3: Conversational path — stream LLM (JSON envelope) ──────
        try:
            await self._stream_conversational_reply(event, session_id)
        finally:
            # Conversational path emits ResponseReady itself; release here
            # so the mic can reopen after TTS finishes.
            audio_state.thinking_finished()

    async def _stream_conversational_reply(
        self, event: ReasoningRequested, session_id: str
    ) -> None:
        memory_event = event.memory_events[0] if event.memory_events else None

        # ── Memory Router -> RAG retrieval ────────────────────────────
        # Gated on purpose: retrieving for "hi" or "open chrome" wastes
        # an embedding call and pollutes the prompt. See memory/rag/router.py.
        retrieved = await self._retrieve_rag_context(
            event.intent_event.raw_input if event.intent_event else "",
            event.intent_event,
        )

        system = self._build_system_prompt(memory_event, retrieved)

        # history already ends with the user's current message (added in step 1)
        history = self._get_history()

        logger.debug(
            "Calling LLM (streaming) | history_turns=%d | system_len=%d | rag_chunks=%d",
            len(history),
            len(system),
            len(retrieved),
        )

        # Stream the JSON envelope token-by-token. While it arrives, strip
        # the scaffolding and forward ONLY the visible `message` text as
        # PartialResponse events so the UI renders words as they generate.
        # The full buffer is parsed once at the end for the authoritative
        # message + sql/rag persistence (see _persist_response_memory).
        raw_parts: list[str] = []
        buffer = ""
        prev_visible = ""
        stream_failed = False
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
            stream_failed = True
            logger.warning(
                "Streaming LLM call failed (session=%s): %s", session_id, exc
            )

        raw_response = "".join(raw_parts)
        payload = parse_memory_response(raw_response)
        response_text = payload.get("message", "").strip()

        # Decide if the parsed message is *actually* the message or a
        # raw-text fallback that ``parse_memory_response`` substitutes
        # when the JSON envelope is missing or broken. We treat the
        # message as suspicious only if:
        #   - it is empty
        #   - OR it equals the raw text AND the raw text looks like a
        #     partial/whole JSON envelope (``{...}`` shape), which means
        #     parse_memory_response gave us back the literal JSON it
        #     couldn't decode.
        # We deliberately do NOT fall back when the raw text is plain
        # English (e.g. the streaming LLM client yielded a single fallback
        # chunk like "I'm having trouble thinking right now. Could you
        # try again?" on an upstream exception). Before this guard,
        # every such fallback was silently discarded because
        # ``response_text == raw_response`` was treated as a failure signal
        # even when the raw text was perfectly good conversational copy.
        raw_stripped = raw_response.strip()
        looks_like_json = raw_stripped.startswith("{") and raw_stripped.endswith("}")
        raw_text_fallback = response_text == raw_stripped and looks_like_json
        # The third condition (`response_text.startswith("{")`) catches the
        # case where the parse succeeded but returned a dict-as-message
        # (which shouldn't happen but historically did on schema drift).
        if not response_text or raw_text_fallback or response_text.startswith("{"):
            response_text = prev_visible

        # If the stream returned nothing usable — first chunk won the race
        # but no real text followed (Gemini occasionally streams just the
        # opening JSON and then stalls), or the call raised before any
        # chunk arrived — retry once before giving up. Without this, the
        # mic opens into dead silence and the user sees the assistant
        # "ignore" their question.
        if not response_text:
            if stream_failed or raw_parts:
                logger.info(
                    "Empty LLM stream (session=%s, raw=%d chars, prev_visible=%d chars); retrying once",
                    session_id,
                    len(raw_response),
                    len(prev_visible),
                )
                try:
                    raw_parts.clear()
                    prev_visible = ""
                    async for chunk in self.gemini_client.generate_with_history_stream(
                        history=history,
                        system=system,
                        hedge_width=1,
                        call_site="conversational-retry",
                    ):
                        raw_parts.append(chunk)
                        buffer = "".join(raw_parts)
                        visible = extract_streamed_message(buffer)
                        if len(visible) > len(prev_visible):
                            self.bus.emit(
                                PartialResponse(
                                    text=visible[len(prev_visible) :],
                                    session_id=session_id,
                                )
                            )
                            prev_visible = visible
                    raw_response = "".join(raw_parts)
                    payload = parse_memory_response(raw_response)
                    response_text = payload.get("message", "").strip()
                    raw_stripped = raw_response.strip()
                    looks_like_json = raw_stripped.startswith(
                        "{"
                    ) and raw_stripped.endswith("}")
                    raw_text_fallback = (
                        response_text == raw_stripped and looks_like_json
                    )
                    if (
                        not response_text
                        or raw_text_fallback
                        or response_text.startswith("{")
                    ):
                        response_text = prev_visible
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Retry of streaming LLM call failed (session=%s): %s",
                        session_id,
                        exc,
                    )

        # Last resort: surface a user-visible fallback so the mic doesn't
        # open into dead silence. TTS will speak this short apology and
        # the mic reopens for the user's next attempt.
        if not response_text:
            logger.error(
                "LLM produced no usable text after retry (session=%s, user_input=%r); surfacing fallback",
                session_id,
                event.intent_event.raw_input if event.intent_event else None,
            )
            response_text = (
                "Sorry, sir — my response came back blank. Could you say that again?"
            )

        # ── Step 4: Add assistant turn to short_term SYNCHRONOUSLY ────────
        self.memory_manager.short_term.add("assistant", response_text)

        # ── Step 5: Persist memory from the primary Gemini response ───────
        # Memory now comes directly from the response JSON envelope — no
        # separate extraction step after the conversation. `sql` lands in
        # SQLite, `rag` in the vector store (+ the rag.txt audit log).
        await self._persist_response_memory(payload, session_id=session_id)

        # ── Step 6: Mark stream done + emit final response ────────────────
        self.bus.emit(PartialResponse(text="", done=True, session_id=session_id))
        self.bus.emit(ResponseReady(text=response_text, session_id=session_id))

    # ── Task completion handler ───────────────────────────────────────────────

    async def on_task_completed(self, event: TaskCompleted) -> None:
        session_id = event.session_id
        pending = self._pending_turns.pop(session_id, None)

        try:
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
        finally:
            # Release the gate if this session was the one holding it
            # open across a deferred task handoff. Conversational turns
            # never enter _gated_session_ids, so this is a no-op for them.
            self._maybe_release_gate(session_id)

    async def on_plan_completed(self, event: PlanCompleted) -> None:
        """Emit a user-facing response when a plan finishes.

        Single-task plans run through :mod:`reasoning.tool_voice`
        synchronously (zero LLM cost). Multi-task and failure paths go
        through :mod:`reasoning.naturalize` which calls the LLM to
        combine the per-tool results into one JARVIS-style reply.
        """
        session_id = event.session_id
        summary = (event.summary or "").strip()

        try:
            response_text = await naturalize_plan_response(
                llm=self.gemini_client,
                user_request=event.user_request or summary,
                task_results=list(event.task_results or []),
                status=event.status,
                token_log=self.token_log,
            )

            self.memory_manager.short_term.add("assistant", response_text)

            self.bus.emit(ResponseReady(text=response_text, session_id=session_id))
        finally:
            self._maybe_release_gate(session_id)

    def _maybe_release_gate(self, session_id: str) -> None:
        """Release the thinking gate if this session was holding it open
        across a deferred task/plan handoff. Idempotent and safe to call
        from any handler — conversational turns never enter the gated
        set, so this is a no-op for them."""
        if session_id in self._gated_session_ids:
            self._gated_session_ids.discard(session_id)
            audio_state.thinking_finished()
