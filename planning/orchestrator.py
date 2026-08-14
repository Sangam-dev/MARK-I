"""The Task Orchestrator — state and traffic control between the two LLMs.

Before this layer, the Conversation LLM and the Task LLM were connected
but not *coordinated*: a delegation went out, a result came back, and
anything in between — a missing parameter, a request for approval — had
nowhere to live. A task that needed to ask the user a question had to
either guess or fail.

The Orchestrator is that in-between::

    User ─► Conversation LLM ─► ORCHESTRATOR ─► Task LLM ─► tools
                    ▲                │                        │
                    └──────── ORCHESTRATOR ◄──────────────────┘

It owns three things nobody else may own:

**Task state.** Every user-level task has a
:class:`~planning.task_state.TaskState` that survives across turns, so
"what should the body say?" and the answer three seconds later are the
same task rather than two unrelated ones.

**Follow-up routing.** A message arriving while a task waits is
classified — answer, confirmation, rejection, cancellation, modification
or a genuinely new task — and routed to the existing task where that is
what the user meant. The Conversation LLM proposes the classification
because it can see the conversation; the Orchestrator *validates* it
against real state, because a model claiming "the user confirmed" when
nothing is awaiting confirmation is exactly the failure this layer
exists to prevent.

**Approval.** ``confirm`` is stripped from every parameter set arriving
from either LLM, and re-applied only from
:attr:`~planning.task_state.TaskState.user_confirmed`, which only a real
subsequent user message can set. A model cannot approve its own
dangerous action here, no matter what it emits — and if this layer ever
missed one, the per-tool two-phase gates in :mod:`actions.gmail_tool`
and :mod:`actions.system_tool` still refuse it.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.bus import EventBus
from core.events import (
    TaskDispatched,
    TaskProtocolResponse,
    TaskRequested,
    TaskResultReady,
)
from planning.protocol import (
    Completed,
    ConfirmationRequired,
    Execute,
    Failed,
    InputRequired,
    ProtocolError,
    parse_task_response,
)
from planning.task_state import (
    MAX_ATTEMPTS,
    TaskPhase,
    TaskState,
    TaskStateStore,
)

logger = logging.getLogger("kancha.planning.orchestrator")

# Modes the Conversation LLM may propose for a turn.
MODE_NEW = "new"
MODE_ANSWER = "answer"
MODE_CONFIRM = "confirm"
MODE_REJECT = "reject"
MODE_CANCEL = "cancel"
MODE_MODIFY = "modify"

_RESUME_MODES = frozenset({MODE_ANSWER, MODE_CONFIRM, MODE_REJECT, MODE_MODIFY})

# Deterministic safety net for the two cases where getting it wrong is
# worst: a bare "yes" that runs nothing, or a bare "no" that runs the
# thing anyway. Only consulted when a task is actually awaiting
# confirmation, and only for messages that are *nothing but* assent or
# refusal — "yes, but send it to Bob instead" is not matched here and is
# left to the Conversation LLM, which can see what "but" changed.
#
# Each side is a bare interjection ("yes"), a bare action phrase
# ("send it"), or the two combined with ordinary punctuation between
# them ("yeah, send it.", "no, don't send it") — which is how people
# actually answer a yes/no question, not just the single-word case.
_YES_WORD = (
    r"(?:yes|yeah|yep|yup|ok|okay|sure|alright|certainly|definitely|"
    r"affirmative|confirmed?)"
)
_YES_ACTION = (
    r"(?:go\s*ahead(?:\s+and\s+(?:do|send)\s+it)?|do\s+it|do\s+that|"
    r"send\s+it|proceed|please\s+do(?:\s+it)?|go\s+for\s+it|go\s+on)"
)
_BARE_YES_RE = re.compile(
    rf"^(?:{_YES_WORD}(?:[,!.\s]+{_YES_ACTION})?|{_YES_ACTION})"
    rf"(?:[,\s]+(?:now|please))?[.!]?$",
    re.IGNORECASE,
)
_NO_WORD = r"(?:no|nope|nah|negative)"
_NO_ACTION = (
    r"(?:don'?t(?:\s+send\s+it)?|do\s+not(?:\s+send\s+it)?|stop|"
    r"cancel(?:\s+(?:it|that))?|never\s*mind|forget\s+it|abort|hold\s+off)"
)
_BARE_NO_RE = re.compile(
    rf"^(?:{_NO_WORD}(?:[,!.\s]+{_NO_ACTION})?|{_NO_ACTION})[.!]?$",
    re.IGNORECASE,
)


def looks_like_assent(text: str) -> bool:
    """True if *text* is nothing but agreement — "yes", "go ahead", "do it",
    or a natural combination like "yeah, send it."

    Public because the Conversation LLM side needs the same answer: a
    plain confirmation must reach the open task even if the controller
    replied conversationally (or attached a stray task of its own)
    instead of proposing ``mode: "confirm"``.
    """
    return bool(_BARE_YES_RE.match((text or "").strip()))


def looks_like_refusal(text: str) -> bool:
    """True if *text* is nothing but refusal or cancellation — "no", "stop",
    or a natural combination like "no, don't send it"."""
    return bool(_BARE_NO_RE.match((text or "").strip()))


def _strip_confirm(params: dict[str, Any]) -> dict[str, Any]:
    """Remove any self-granted approval flag.

    Neither LLM gets to set this. It is re-applied by the Orchestrator
    alone, from state that only a real user message can produce.
    """
    return {k: v for k, v in (params or {}).items() if k != "confirm"}


class TaskOrchestrator:
    """Owns task state; the only path between the two LLMs."""

    def __init__(self, bus: EventBus, store: TaskStateStore | None = None) -> None:
        self._bus = bus
        self.store = store or TaskStateStore()

    def register(self) -> None:
        """Subscribe to both directions of the delegation path.

        The Task LLM no longer listens for :class:`TaskRequested` — it
        listens for :class:`TaskDispatched`, which only this class
        emits. That is what makes "neither LLM bypasses the
        orchestrator" a property of the wiring rather than a convention.
        """
        self._bus.subscribe(TaskRequested, self.on_task_requested)
        self._bus.subscribe(TaskProtocolResponse, self.on_protocol_response)

    # ── Conversation LLM ─► Orchestrator ──────────────────────────────

    async def on_task_requested(self, event: TaskRequested) -> None:
        """Start a task, or route this turn into the one already running."""
        session_id = event.session_id or "default"
        mode = self._resolve_mode(event, session_id)
        active = self.store.active(session_id)

        if mode == MODE_CANCEL or mode == MODE_REJECT:
            self._cancel(active, session_id, mode)
            return

        if mode in _RESUME_MODES and active is not None:
            self._resume(active, event, mode)
            return

        self._start(event, session_id)

    def _resolve_mode(self, event: TaskRequested, session_id: str) -> str:
        """Decide what this turn does, trusting state over the model.

        The Conversation LLM's proposal is a hint. It is honoured only
        when the task state can actually support it — a "confirm" with
        nothing awaiting confirmation becomes a new task, not an
        approval of whatever ran last.
        """
        proposed = (event.mode or MODE_NEW).strip().lower()
        awaiting = self.store.awaiting(session_id)
        active = self.store.active(session_id)

        if proposed == MODE_CANCEL:
            # Cancelling with nothing running is answered, not turned into
            # a task — "cancel that" is never an instruction to execute.
            return MODE_CANCEL

        if proposed in (MODE_CONFIRM, MODE_REJECT):
            if awaiting is None or awaiting.status is not TaskPhase.WAITING_FOR_CONFIRMATION:
                logger.info(
                    "Ignoring proposed mode %r — no task is awaiting confirmation "
                    "in session %s",
                    proposed,
                    session_id,
                )
                # A stray "no" must not become a task either. A stray
                # "yes" may legitimately carry a fresh instruction, so it
                # falls through to being treated as new.
                return MODE_CANCEL if proposed == MODE_REJECT else MODE_NEW
            return proposed

        if proposed == MODE_ANSWER:
            if awaiting is None or awaiting.status is not TaskPhase.WAITING_FOR_INPUT:
                logger.info(
                    "Ignoring proposed mode 'answer' — no task is awaiting input "
                    "in session %s",
                    session_id,
                )
                return MODE_NEW
            return MODE_ANSWER

        if proposed == MODE_MODIFY:
            return MODE_MODIFY if active is not None else MODE_NEW

        # Proposed "new". If something is waiting on a yes/no and the
        # user said exactly that, believe the user over the classifier.
        if awaiting is not None and awaiting.status is TaskPhase.WAITING_FOR_CONFIRMATION:
            text = (event.user_request or "").strip()
            if _BARE_YES_RE.match(text):
                logger.info(
                    "Reclassified turn as confirmation of task %s (bare assent)",
                    awaiting.task_id,
                )
                return MODE_CONFIRM
            if _BARE_NO_RE.match(text):
                logger.info(
                    "Reclassified turn as rejection of task %s (bare refusal)",
                    awaiting.task_id,
                )
                return MODE_REJECT

        return MODE_NEW

    def _start(self, event: TaskRequested, session_id: str) -> None:
        state = self.store.create(
            session_id=session_id,
            instruction=event.instruction,
            action=event.task_type,
            params=_strip_confirm(event.parameters),
            user_request=event.user_request,
            expected_result=event.expected_result,
            context=dict(event.context or {}),
            follow_up=event.follow_up,
            task_id=event.task_id,
        )
        self._dispatch(state)

    def _resume(self, state: TaskState, event: TaskRequested, mode: str) -> None:
        """Continue an existing task with what the user just supplied."""
        if mode == MODE_CONFIRM:
            state.user_confirmed = True
            logger.info(
                "task_resumed | task_id=%s reason=confirmation_granted",
                state.task_id,
            )
        elif mode == MODE_ANSWER:
            state.merge_params(_strip_confirm(event.parameters))
            state.missing_fields = []
            state.question = ""
            logger.info(
                "task_resumed | task_id=%s reason=answer fields=%s",
                state.task_id,
                sorted(_strip_confirm(event.parameters)),
            )
        elif mode == MODE_MODIFY:
            state.merge_params(_strip_confirm(event.parameters))
            if event.instruction:
                state.instruction = event.instruction
            # A modified task is a different request, so any approval
            # given for the previous shape no longer applies.
            state.user_confirmed = False
            state.confirmation_data = {}
            logger.info(
                "task_resumed | task_id=%s reason=modified | %r",
                state.task_id,
                state.instruction,
            )

        if event.user_request:
            state.user_request = event.user_request
        self._dispatch(state)

    def _cancel(self, state: TaskState | None, session_id: str, mode: str) -> None:
        if state is None:
            logger.info("Nothing to cancel in session %s", session_id)
            self._bus.emit(
                TaskResultReady(
                    task_id="",
                    status="cancelled",
                    error="there was nothing running to cancel",
                    session_id=session_id,
                )
            )
            return

        reason = (
            "the user declined the confirmation"
            if mode == MODE_REJECT
            else "the user cancelled it"
        )
        self.store.transition(
            state, TaskPhase.CANCELLED, event="task_cancelled", error=reason
        )
        self._emit_result(state, status="cancelled", error=reason)

    # ── Orchestrator ─► Task LLM ──────────────────────────────────────

    def _dispatch(self, state: TaskState) -> None:
        """Hand the task to the Task LLM with everything known so far."""
        state.attempts += 1
        if state.attempts > MAX_ATTEMPTS:
            error = (
                f"gave up after {MAX_ATTEMPTS} attempts without completing the task"
            )
            logger.warning("task_failed | task_id=%s %s", state.task_id, error)
            self.store.transition(
                state, TaskPhase.FAILED, event="task_failed", error=error
            )
            self._emit_result(state, status="failed", error=error)
            return

        self.store.transition(state, TaskPhase.RUNNING, event="task_executing")
        logger.info(
            "task_dispatched | task_id=%s attempt=%d confirmed=%s params=%s",
            state.task_id,
            state.attempts,
            state.user_confirmed,
            sorted(state.params),
        )
        self._bus.emit(
            TaskDispatched(
                task_id=state.task_id,
                task_type=state.action,
                instruction=state.instruction,
                # Stripped again at the boundary: the only approval that
                # travels is the explicit user_confirmed flag below.
                parameters=_strip_confirm(state.params),
                expected_result=state.expected_result,
                context=dict(state.context),
                follow_up=state.follow_up,
                user_request=state.user_request,
                user_confirmed=state.user_confirmed,
                attempt=state.attempts,
                session_id=state.session_id,
            )
        )

    # ── Task LLM ─► Orchestrator ──────────────────────────────────────

    async def on_protocol_response(self, event: TaskProtocolResponse) -> None:
        """Validate one Task LLM response and advance the task."""
        state = self.store.get(event.task_id)
        if state is None:
            logger.warning(
                "Protocol response for unknown task %s — ignoring", event.task_id
            )
            return

        try:
            response = parse_task_response(event.type, event.payload, event.task_id)
        except ProtocolError as exc:
            # A malformed response must never become a question to the
            # user or an unvetted execution. Fail the task instead.
            logger.error(
                "task_failed | task_id=%s invalid Task LLM response: %s",
                event.task_id,
                exc,
            )
            self.store.transition(
                state,
                TaskPhase.FAILED,
                event="task_failed",
                error=f"the execution layer returned an invalid response ({exc})",
            )
            self._emit_result(state, status="failed", error=state.error)
            return

        if isinstance(response, InputRequired):
            self._handle_input_required(state, response)
        elif isinstance(response, ConfirmationRequired):
            self._handle_confirmation_required(state, response)
        elif isinstance(response, Execute):
            # Informational: the plan is running. Record what for logs.
            state.action = response.action or state.action
            if response.params:
                state.merge_params(_strip_confirm(response.params))
            logger.info(
                "task_executing | task_id=%s action=%s",
                state.task_id,
                state.action or "(unspecified)",
            )
        elif isinstance(response, Completed):
            self._handle_completed(state, response)
        elif isinstance(response, Failed):
            self.store.transition(
                state,
                TaskPhase.FAILED,
                event="task_failed",
                error=response.error or "the task failed",
            )
            self._emit_result(state, status="failed", error=state.error)

    def _handle_input_required(self, state: TaskState, response: InputRequired) -> None:
        """Park the task and pass the question up to the conversation."""
        repeated = state.already_asked(response.missing_fields)
        if repeated:
            # We asked, the user answered, and the Task LLM is asking
            # again for something now sitting in params. Answering again
            # would loop forever, so stop.
            error = (
                "the execution layer asked again for "
                + ", ".join(repeated)
                + " after it had already been provided"
            )
            logger.error("task_failed | task_id=%s %s", state.task_id, error)
            self.store.transition(
                state, TaskPhase.FAILED, event="task_failed", error=error
            )
            self._emit_result(state, status="failed", error=error)
            return

        for name in response.missing_fields:
            if name not in state.asked_fields:
                state.asked_fields.append(name)

        self.store.transition(
            state,
            TaskPhase.WAITING_FOR_INPUT,
            event="waiting_for_input",
            missing_fields=list(response.missing_fields),
            question=response.question,
        )
        self._emit_result(
            state,
            status="waiting_for_input",
            question=response.question,
            missing_fields=list(response.missing_fields),
        )

    def _handle_confirmation_required(
        self, state: TaskState, response: ConfirmationRequired
    ) -> None:
        self.store.transition(
            state,
            TaskPhase.WAITING_FOR_CONFIRMATION,
            event="waiting_for_confirmation",
            confirmation_required=True,
            confirmation_data=dict(response.confirmation_data),
            confirmation_description=response.description,
            action=response.action or state.action,
        )
        self._emit_result(
            state,
            status="waiting_for_confirmation",
            description=response.description,
            confirmation_data=dict(response.confirmation_data),
        )

    def _handle_completed(self, state: TaskState, response: Completed) -> None:
        results = response.result.get("results")
        results = list(results) if isinstance(results, list) else []
        status = str(response.result.get("status") or "completed")
        error = str(response.result.get("error") or "")

        phase = TaskPhase.COMPLETED if status != "failed" else TaskPhase.FAILED
        self.store.transition(
            state,
            phase,
            event="task_completed" if phase is TaskPhase.COMPLETED else "task_failed",
            result=results,
            error=error,
            # Approval is spent. A later "yes" must not re-run this.
            user_confirmed=False,
        )
        self._emit_result(state, status=status, results=results, error=error)

    # ── Orchestrator ─► Conversation LLM ──────────────────────────────

    def _emit_result(
        self,
        state: TaskState,
        status: str,
        results: list[dict[str, Any]] | None = None,
        error: str = "",
        question: str = "",
        missing_fields: list[str] | None = None,
        description: str = "",
        confirmation_data: dict[str, Any] | None = None,
    ) -> None:
        """Report upward. The only thing the Conversation LLM receives."""
        self._bus.emit(
            TaskResultReady(
                task_id=state.task_id,
                status=status,
                results=results or [],
                error=error,
                task_type=state.action,
                instruction=state.instruction,
                user_request=state.user_request,
                question=question,
                missing_fields=missing_fields or [],
                description=description,
                confirmation_data=confirmation_data or {},
                session_id=state.session_id,
            )
        )
