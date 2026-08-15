from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Intent(str, Enum):
    QUERY = "query"
    CONVERSATIONAL = "conversational"
    TASK = "task"


class MemoryLayer(str, Enum):
    SHORT_TERM = "short_term"
    STRUCTURED = "structured"
    EPISODIC = "episodic"


@dataclass(frozen=True)
class BaseEvent:
    """
    baseclass is for all other class events.
    all events class will be recorded
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: str = field(default="default")


@dataclass(frozen=True)
class WakeWordDetected(BaseEvent):
    """
    Event triggered when a wake word is detected.

    emitted by: STT module
    consumed by: STT module, Intent module

    """

    audio_path: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class TextInputReceived(BaseEvent):
    """
    Event triggered when text input is received.

    emitted by: UI module
    consumed by: Intent module
    """

    text: str = ""


@dataclass(frozen=True)
class TranscriptReady(BaseEvent):
    """
    Event triggered when a transcript is ready.

    emitted by: STT module
    consumed by: Intent module
    """

    text: str = ""
    word_error_rate: float = 0.0
    language: str = "en"


@dataclass(frozen=True)
class PartialTranscriptReady(BaseEvent):
    """
    Event fired with a *provisional* transcript of the speech recorded so
    far, produced before the authoritative :class:`TranscriptReady`.

    The voice path emits this from an interim (fast, lightweight) ASR pass
    that races ahead of the final Whisper transcription. It exists purely
    so the ReasoningCoordinator can start generating a speculative reply
    while the user's final transcript is still being produced. Nothing
    here is authoritative — ``final`` must remain False, and consumers
    must treat the text as a guess that can be discarded.

    emitted by: input/stt.py (interim ASR pass)
    consumed by: reasoning/coordinator.py (preemptive generation)
    """

    text: str = ""
    final: bool = False
    language: str = "en"


@dataclass(frozen=True)
class IntentIdentified(BaseEvent):
    """
    Event triggered when an intent is identified.

    emitted by: Intent module
    consumed by: Response module
    """

    raw_input: str = ""
    intent: Intent = Intent.CONVERSATIONAL
    confidence: float = 1.0
    entities: dict[str, Any] = field(default_factory=dict)
    # Task execution fields (populated when intent == TASK)
    requires_task: bool = False
    task_type: str | None = None
    task_params: dict[str, Any] = field(default_factory=dict)


# -------------- Memory Events -----------------#


@dataclass(frozen=True)
class MemoryUpdateNeeded(BaseEvent):
    """
    Event triggered when an update to memory is needed.

    NOTE: currently unused by the active pipeline. Durable memory is fed
    directly by the primary Gemini response JSON envelope (sql/rag) in
    reasoning/coordinator.py. Kept as part of the event schema for the
    disconnected episodic memory implementation.

    emitted by: Intent module
    consumed by: Memory module
    """

    content: str = ""
    layer: MemoryLayer = MemoryLayer.EPISODIC
    metadata: dict[str, Any] = field(default_factory=dict)
    role: str = "user"


@dataclass(frozen=True)
class MemoryRetrieved(BaseEvent):
    """
    Event triggered when memory is retrieved.

    emitted by: Memory module
    consumed by: Intent module
    """

    query: str = ""
    structured_context: list[dict[str, Any]] = field(default_factory=list)
    episodic_context: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReasoningRequested(BaseEvent):
    """
    emitted by: Intent module
    consumed by: Reasoning module
    """

    intent_event: IntentIdentified = field(default_factory=IntentIdentified)
    memory_events: list[MemoryRetrieved] = field(default_factory=list)


@dataclass(frozen=True)
class ResponseReady(BaseEvent):
    """
    Event triggered when a response is ready.

    emitted by: Response module
    consumed by: UI module
    """

    text: str = ""
    llm_raw: str = ""
    requires_task: bool = False


@dataclass(frozen=True)
class PartialResponse(BaseEvent):
    """
    Event emitted as a conversational response streams in.

    Carries the latest chunk of the assistant's *message* text (JSON
    envelope scaffolding already stripped) so the UI can render tokens
    as they arrive instead of waiting for the full ``ResponseReady``.

    emitted by: ReasoningCoordinator (streaming conversational path)
    consumed by: api/bridge.py (forwards to WebSocket clients)
    """

    text: str = ""
    done: bool = False


# -------------- UI / Presentation Events -----------------#


class AssistantState(str, Enum):
    SLEEPING = "sleeping"  # wake word required to re-activate
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass(frozen=True)
class AssistantStateChanged(BaseEvent):
    """
    Event triggered when the assistant's high-level UI state changes.

    This exists purely to drive presentation layers (e.g. the jarvis_frontend
    orb animation) — it carries no business data, only a state label.

    emitted by: input/stt.py (listening -> thinking), output/tts.py
                (speaking -> idle), api/server.py (thinking, for text input
                which has no "listening" phase; idle fallback when TTS is
                disabled)
    consumed by: api/bridge.py (forwards to WebSocket clients)
    """

    state: AssistantState = AssistantState.IDLE


@dataclass(frozen=True)
class TaskRequested(BaseEvent):
    """The **Task Request** — the Conversation LLM delegating to the Task LLM.

    This is the ONLY way work reaches the Task LLM. Raw user input never
    does: the Conversation LLM (``reasoning/coordinator.py``) decides
    whether a task is needed, resolves conversational references
    ("do it", "send that") against the session history, and emits this
    event with a self-contained instruction.

    Fields
    ------
    task_id:
        Correlation id. Echoed back on :class:`TaskResultReady`.
    task_type:
        Optional tool hint from ``tasks.registry.TASK_REGISTRY``. The
        Task LLM validates it and is free to override — it owns *how* to
        execute, the Conversation LLM owns *whether* and *what*.
    instruction:
        Imperative, self-contained description of the work. Must never
        contain unresolved references ("it", "that").
    parameters:
        Arguments the Conversation LLM already knows. Validated against
        the registry; a bad set simply falls through to decomposition.
    expected_result:
        What the Conversation LLM expects back, in words. Passed to the
        planner as context.
    context:
        Free-form resolution trace, e.g.
        ``{"referenced_object": "previous_email"}``. Prompt context only.
    follow_up:
        When true the Conversation LLM wants to see the result and decide
        the next action itself (task chaining across turns) rather than
        just having it phrased for the user.
    user_request:
        The raw user text that triggered the delegation. Logging and
        naturalisation context only — the Task LLM must plan from
        ``instruction``.
    mode:
        What this turn is doing to the conversation's task. One of
        ``"new"``, ``"answer"`` (supplying a field the Task LLM asked
        for), ``"confirm"``, ``"reject"``, ``"cancel"``, ``"modify"``.
        The Conversation LLM proposes it; the Orchestrator validates it
        against the real task state and may override — a "confirm" with
        nothing awaiting confirmation is not a confirmation.
    resume_task_id:
        The task this turn continues, when known. Empty on a new task.
        The Orchestrator resolves it from session state when the
        Conversation LLM leaves it blank.

    emitted by: reasoning/coordinator.py (Conversation LLM)
    consumed by: planning/orchestrator.py (Task Orchestrator)

    Note the consumer: this no longer reaches the Task LLM directly.
    The Orchestrator owns task state and re-emits :class:`TaskDispatched`.
    """

    task_id: str = ""
    task_type: str = ""
    instruction: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    expected_result: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    follow_up: bool = False
    user_request: str = ""
    mode: str = "new"
    resume_task_id: str = ""


@dataclass(frozen=True)
class TaskDispatched(BaseEvent):
    """The **Orchestrator handing work to the Task LLM**.

    Distinct from :class:`TaskRequested` on purpose: a Task Request is
    what the *conversation* asked for, while a Dispatch is what the
    Orchestrator decided to actually run, after merging answers from
    earlier turns and deciding whether approval exists.

    ``user_confirmed`` is the only channel through which approval
    reaches execution, and the Orchestrator sets it **only** from a real
    subsequent user message. Nothing the Task LLM generates can set it —
    that is what stops a model from approving its own dangerous action.

    ``attempt`` counts how many times this task has been dispatched, so
    a Task LLM that keeps asking for information it already has can be
    stopped rather than looped.

    emitted by: planning/orchestrator.py
    consumed by: planning/planner.py (Task LLM)
    """

    task_id: str = ""
    task_type: str = ""
    instruction: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    expected_result: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    follow_up: bool = False
    user_request: str = ""
    user_confirmed: bool = False
    attempt: int = 1


@dataclass(frozen=True)
class TaskProtocolResponse(BaseEvent):
    """The **Task LLM's structured reply** to a dispatch.

    The Task LLM never speaks to the user. Everything it wants to say —
    "I need the body", "this will send mail, are you sure", "here is the
    result" — comes back through this one event, in the closed vocabulary
    of :mod:`planning.protocol`:

    ``input_required`` | ``confirmation_required`` | ``execute`` |
    ``completed`` | ``failed``

    ``payload`` holds the type-specific fields and is validated by
    :func:`planning.protocol.parse_task_response` before the Orchestrator
    acts on it. An unparseable payload fails the task; it never becomes
    a question to the user.

    emitted by: planning/planner.py (Task LLM)
    consumed by: planning/orchestrator.py
    """

    task_id: str = ""
    type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskResultReady(BaseEvent):
    """The **Task Result** — the Task LLM reporting back to the Conversation LLM.

    Execution data only. The Task LLM must never put a user-facing
    sentence here; turning this into speech is the Conversation LLM's
    job (see ``reasoning/coordinator.py:on_task_result``).

    ``status`` is one of ``"completed" | "partial" | "failed" |
    "cancelled"`` — the same vocabulary :class:`PlanCompleted` uses —
    plus the two *interactive* states the Orchestrator introduces:
    ``"waiting_for_input"`` and ``"waiting_for_confirmation"``. Those two
    are not outcomes; they are the Task LLM asking the user something
    through the only party allowed to speak to them.

    ``results`` carries the per-tool outcomes in
    :class:`PlanCompleted.task_results` shape
    (``{"tool", "result", "arguments"}``) so the existing
    :func:`reasoning.naturalize.naturalize_plan_response` consumes it
    unchanged.

    ``question``/``missing_fields`` are set on ``waiting_for_input``;
    ``description``/``confirmation_data`` on
    ``waiting_for_confirmation``. The Conversation LLM turns whichever
    is present into natural speech — it must not read them out verbatim.

    emitted by: planning/orchestrator.py
    consumed by: reasoning/coordinator.py (Conversation LLM)
    """

    task_id: str = ""
    status: str = "completed"
    results: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    task_type: str = ""
    instruction: str = ""
    user_request: str = ""
    question: str = ""
    missing_fields: list[str] = field(default_factory=list)
    description: str = ""
    confirmation_data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskExecutionRequested(BaseEvent):
    """
    Event triggered when a task execution is requested.

    emitted by: Response module
    consumed by: Task module
    """

    task_name: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskCompleted(BaseEvent):
    """
    Event triggered when a task is completed.

    emitted by: Task module
    consumed by: Response module
    """

    task_name: str = ""
    success: bool = True
    result: str = ""
    error: str = ""
    #: Structured failure class (see :mod:`core.failures`). ``None`` on
    #: success or when the failure is transient/unknown and therefore
    #: retryable. Never a user-facing sentence.
    error_type: str | None = None


@dataclass(frozen=True)
class SystemError(BaseEvent):
    """
    Event triggered when a system error occurs.

    emitted by: any module
    consumed by: Error handling module
    """

    source_module: str = ""
    error_message: str = ""
    recoverable: bool = True


@dataclass(frozen=True)
class ShutdownRequested(BaseEvent):
    """
    Event triggered when a shutdown is requested.

    emitted by: any module
    consumed by: any module

    """

    reason: str = "user requested"


@dataclass(frozen=True)
class SystemMonitorAlert(BaseEvent):
    """
    Event triggered when the background SystemMonitorLoop detects a metric
    crossing a configured threshold.

    The ``text`` field carries the same ``[SYSTEM_ALERT] …`` phrasing that
    ``actions.system_monitor.SystemMonitor.check()`` returns, so the bridge
    to ``ResponseReady`` (which feeds TTS and the console formatter) is a
    one-line copy.

    ``metrics`` is optional structured metadata for the WebSocket bridge
    (``api/bridge.py``) — the voice path ignores it. Use it to render a
    sparkline or coloured badge on the web front-end.

    emitted by: core/system_monitor_loop.py
    consumed by: a small handler registered in core/pipeline.py that emits
                 ``ResponseReady(text=event.text)`` so the existing TTS and
                 ResponseFormatter subscribers pick it up.
    """

    text: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


# -------------- Planning Events -----------------#


@dataclass(frozen=True)
class PlanCreated(BaseEvent):
    """
    Event triggered when the Planner has produced an ExecutionPlan.

    emitted by: planning/planner.py
    consumed by: planning/scheduler.py

    The plan is serialized to a dict (events are frozen dataclasses)
    and rebuilt into an ExecutionPlan inside the Scheduler.
    """

    plan: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskStarted(BaseEvent):
    """
    Event triggered when the PlanExecutor dispatches a single task.

    emitted by: planning/executor.py
    consumed by: UI / telemetry (optional)
    """

    plan_id: str = ""
    task_id: str = ""
    tool: str = ""


@dataclass(frozen=True)
class PlanReplanRequested(BaseEvent):
    """
    Event triggered when a task has permanently failed and the Planner
    should rebuild the remaining steps.

    emitted by: planning/scheduler.py
    consumed by: planning/planner.py
    """

    plan_id: str = ""
    failed_task_id: str = ""
    reason: str = ""
    remaining_tasks: list[dict[str, Any]] = field(default_factory=list)
    #: True when the failed task's failure class is permanent (see
    #: :mod:`core.failures`). The replanner then only proceeds with a
    #: genuinely different strategy — never the same plan again.
    terminal: bool = False
    error_type: str | None = None


@dataclass(frozen=True)
class PlanCompleted(BaseEvent):
    """
    Event triggered when an ExecutionPlan reaches a terminal state.

    emitted by: planning/scheduler.py
    consumed by: reasoning/coordinator.py (to emit ResponseReady)

    status: "completed" | "failed" | "partial" | "cancelled"

    task_results carries the per-task outcome so the coordinator can
    run the natural-language pass without re-tracking plan state.
    Each entry is ``{"tool": str, "result": str, "arguments": dict}``
    where ``result`` is the raw tool output (success text or error
    text on failure). Empty entries are omitted.

    user_request is the original user input — passed through so the
    naturalize helper can include it in the LLM paraphrase prompt.
    """

    plan_id: str = ""
    status: str = "completed"
    summary: str = ""
    task_results: list[dict[str, Any]] = field(default_factory=list)
    user_request: str = ""


@dataclass(frozen=True)
class PlanCancelled(BaseEvent):
    """
    Event triggered when an in-flight plan is cancelled (e.g. shutdown).

    emitted by: any module
    consumed by: planning/scheduler.py
    """

    plan_id: str = ""
    reason: str = ""


# -------------------------------------- TESTING --------------------------------------#

# if __name__ == "__main__":
#     # Stage 1
#     assert Intent.QUERY == "query"
#     assert MemoryLayer.EPISODIC == "episodic"

#     # Stage 2
#     e1 = BaseEvent()
#     e2 = BaseEvent()
#     assert e1.event_id != e2.event_id
#     try:
#         e1.session_id = "x"
#         assert False, "should have raised"
#     except Exception:
#         pass

#     # Stage 3
#     t = TranscriptReady(text="hello kancha")
#     assert t.text == "hello kancha"
#     assert t.language == "en"

#     i = IntentIdentified(intent=Intent.TASK)
#     assert i.intent == Intent.TASK
#     assert i.entities == {}

#     # Stage 4
#     m = MemoryRetrieved()
#     assert m.results == []
#     assert m.episodic_chunks == []

#     r = ResponseReady(response="I am KANCHA")
#     assert r.requires_task == False

#     s = SystemError(source_module="stt", error_message="mic not found")
#     assert s.recoverable == True

#     # Unique IDs across different types
#     a = TextInputReceived(text="hello")
#     b = TranscriptReady(text="hello")
#     assert a.event_id != b.event_id

#     print("All events verified.")
