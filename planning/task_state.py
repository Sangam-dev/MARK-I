"""Task state — the memory that makes a multi-turn task one task.

Before this layer existed, a delegation was a fire-and-forget event: the
Conversation LLM emitted a :class:`~core.events.TaskRequested`, the Task
LLM planned it, and a result came back. That works for "open Firefox"
and breaks for anything the assistant needs to *ask about*, because
there was nowhere to put a half-finished task while the user answered.

A :class:`TaskState` is that somewhere. It holds the fields the spec
requires — ``task_id``, ``status``, ``action``, ``params``,
``missing_fields``, ``confirmation_required``, ``confirmation_data``,
``result``, ``error`` — plus what is needed to resume intelligently:
the original instruction, which fields have already been asked about,
and how many times the task has gone round.

The state machine
-----------------
::

    PENDING ──► RUNNING ──┬──► COMPLETED
        ▲                 ├──► FAILED
        │                 ├──► WAITING_FOR_INPUT ────────┐
        │                 └──► WAITING_FOR_CONFIRMATION ─┤
        └─────────────────────────────────────────────────┘
                        (user answers / confirms)

    any non-terminal ──► CANCELLED

Terminal states are terminal. A COMPLETED task cannot be resumed by a
later "yes" — that is the difference between an assistant that
remembers and one that re-sends an email because the user thanked it.

Why not reuse :class:`planning.models.TaskStatus`
-------------------------------------------------
That enum tracks one *tool call* inside an ExecutionPlan. This tracks
one *user-level task* which may span several tool calls and several
conversational turns. Two different lifetimes, deliberately two types;
the enum here is called :class:`TaskPhase` so the two never get
confused at an import site.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("kancha.planning.task_state")

# How many times one task may be dispatched to the Task LLM before the
# Orchestrator gives up. Each answered question costs one attempt, so
# this bounds the "assistant keeps asking questions forever" failure.
MAX_ATTEMPTS = 6

# How long a task may sit unanswered before a new utterance is treated
# as a new task rather than an answer to it. Someone who ignores
# "what should the body say?" for ten minutes and then says "yes" is
# not answering that question.
STALE_AFTER_S = 600.0


class TaskPhase(str, Enum):
    """Lifecycle of one user-level task."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_FOR_INPUT = "waiting_for_input"
    WAITING_FOR_CONFIRMATION = "waiting_for_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL

    @property
    def is_waiting(self) -> bool:
        return self in _WAITING


_TERMINAL = frozenset(
    {TaskPhase.COMPLETED, TaskPhase.FAILED, TaskPhase.CANCELLED}
)
_WAITING = frozenset(
    {TaskPhase.WAITING_FOR_INPUT, TaskPhase.WAITING_FOR_CONFIRMATION}
)


@dataclass(slots=True)
class TaskState:
    """One user-level task, across however many turns it takes."""

    task_id: str
    session_id: str = "default"
    status: TaskPhase = TaskPhase.PENDING

    # What is being done.
    action: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    instruction: str = ""
    user_request: str = ""
    expected_result: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    follow_up: bool = False

    # What is being waited on.
    missing_fields: list[str] = field(default_factory=list)
    question: str = ""
    confirmation_required: bool = False
    confirmation_data: dict[str, Any] = field(default_factory=dict)
    confirmation_description: str = ""

    # How it turned out.
    result: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    # Bookkeeping.
    attempts: int = 0
    #: Fields already asked about, so the same question is never asked
    #: twice. See :meth:`already_asked`.
    asked_fields: list[str] = field(default_factory=list)
    #: Set only by the Orchestrator, only from a real user message.
    user_confirmed: bool = False
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.updated_at = time.monotonic()

    @property
    def is_stale(self) -> bool:
        return (time.monotonic() - self.updated_at) > STALE_AFTER_S

    @property
    def is_resumable(self) -> bool:
        """True if a follow-up message may continue this task."""
        return not self.status.is_terminal and not self.is_stale

    def already_asked(self, fields: list[str]) -> list[str]:
        """Of *fields*, those we have asked about and already have.

        The loop guard: if the Task LLM asks for ``body`` when ``body``
        is sitting in ``params`` because the user answered it last turn,
        something upstream is not reading the state it was given, and
        re-asking would trap the user in a loop.
        """
        return [
            name
            for name in fields
            if name in self.asked_fields and str(self.params.get(name, "")).strip()
        ]

    def merge_params(self, new: dict[str, Any]) -> None:
        """Fold newly-learned parameters in, newest wins."""
        for key, value in (new or {}).items():
            if value is None or value == "":
                continue
            self.params[key] = value
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        """Serialisable snapshot — for logs, tests and prompt context."""
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "action": self.action,
            "params": dict(self.params),
            "missing_fields": list(self.missing_fields),
            "confirmation_required": self.confirmation_required,
            "confirmation_data": dict(self.confirmation_data),
            "result": list(self.result),
            "error": self.error,
            "instruction": self.instruction,
            "attempts": self.attempts,
        }


class TaskStateStore:
    """Holds the active task per session, and a bounded history.

    In-process and per-run, matching how the rest of the session layer
    behaves (:class:`memory.manager.MemoryManager` short-term buffer).
    A task that outlives the process is not a case this assistant has:
    the conversation it belongs to is gone too.
    """

    def __init__(self, history_limit: int = 20) -> None:
        # session_id -> task_id of the task a follow-up would continue.
        self._active: dict[str, str] = {}
        # task_id -> state, for every task this run, active or finished.
        self._tasks: dict[str, TaskState] = {}
        # session_id -> task_ids, oldest first.
        self._history: dict[str, list[str]] = {}
        self._history_limit = history_limit

    # ── creation and lookup ───────────────────────────────────────────

    def create(
        self,
        session_id: str,
        instruction: str,
        action: str = "",
        params: dict[str, Any] | None = None,
        user_request: str = "",
        expected_result: str = "",
        context: dict[str, Any] | None = None,
        follow_up: bool = False,
        task_id: str = "",
    ) -> TaskState:
        if task_id and task_id in self._tasks:
            # The caller offered an id we already know. This happens when
            # the controller proposes a resume ("mode": "answer") that the
            # Orchestrator downgrades to a new task because the original
            # had already finished — it still sends the old id along.
            # Reusing it would overwrite that finished task's record and
            # lose the very history that proves it finished.
            logger.info(
                "task_id %s is already in use — minting a new one for this task",
                task_id,
            )
            task_id = ""

        state = TaskState(
            task_id=task_id or f"task-{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            instruction=instruction,
            action=action,
            params=dict(params or {}),
            user_request=user_request,
            expected_result=expected_result,
            context=dict(context or {}),
            follow_up=follow_up,
        )
        self._tasks[state.task_id] = state
        self._active[session_id] = state.task_id
        history = self._history.setdefault(session_id, [])
        history.append(state.task_id)
        self._prune(session_id)
        logger.info(
            "task_created | task_id=%s session=%s action=%s | %r",
            state.task_id,
            session_id,
            action or "(unspecified)",
            instruction,
        )
        return state

    def get(self, task_id: str) -> TaskState | None:
        return self._tasks.get(task_id)

    def active(self, session_id: str) -> TaskState | None:
        """The task a follow-up in this session would continue.

        Returns None once the task is terminal or stale, which is what
        stops "yes" from re-running something that already finished.
        """
        task_id = self._active.get(session_id)
        if not task_id:
            return None
        state = self._tasks.get(task_id)
        if state is None:
            return None
        if not state.is_resumable:
            return None
        return state

    def awaiting(self, session_id: str) -> TaskState | None:
        """The active task, only if it is actually waiting on the user."""
        state = self.active(session_id)
        if state is not None and state.status.is_waiting:
            return state
        return None

    def _prune(self, session_id: str) -> None:
        history = self._history.get(session_id, [])
        while len(history) > self._history_limit:
            dropped = history.pop(0)
            self._tasks.pop(dropped, None)

    # ── transitions ───────────────────────────────────────────────────

    def transition(
        self,
        state: TaskState,
        phase: TaskPhase,
        *,
        event: str = "",
        **updates: Any,
    ) -> TaskState:
        """Move a task to *phase*, applying *updates* and logging it.

        Every lifecycle log line carries ``task_id``, so one grep
        reconstructs a task's whole history across turns.
        """
        if state.status.is_terminal and phase is not state.status:
            # Not an exception: a late PlanCompleted for a cancelled task
            # is normal, and must not crash the turn. It is just ignored.
            logger.warning(
                "task_transition_refused | task_id=%s %s -> %s (already terminal)",
                state.task_id,
                state.status.value,
                phase.value,
            )
            return state

        previous = state.status
        state.status = phase
        for key, value in updates.items():
            setattr(state, key, value)
        state.touch()

        if phase.is_terminal:
            # Free the session slot so the next utterance starts fresh.
            if self._active.get(state.session_id) == state.task_id:
                self._active.pop(state.session_id, None)

        logger.info(
            "%s | task_id=%s session=%s %s -> %s%s",
            event or _EVENT_FOR_PHASE.get(phase, "task_updated"),
            state.task_id,
            state.session_id,
            previous.value,
            phase.value,
            f" error={state.error!r}" if state.error else "",
        )
        return state

    # ── introspection ─────────────────────────────────────────────────

    def snapshot(self, session_id: str) -> list[dict[str, Any]]:
        """Every known task for a session, oldest first. Tests and debug."""
        return [
            self._tasks[task_id].to_dict()
            for task_id in self._history.get(session_id, [])
            if task_id in self._tasks
        ]


_EVENT_FOR_PHASE = {
    TaskPhase.PENDING: "task_created",
    TaskPhase.RUNNING: "task_executing",
    TaskPhase.WAITING_FOR_INPUT: "waiting_for_input",
    TaskPhase.WAITING_FOR_CONFIRMATION: "waiting_for_confirmation",
    TaskPhase.COMPLETED: "task_completed",
    TaskPhase.FAILED: "task_failed",
    TaskPhase.CANCELLED: "task_cancelled",
}
