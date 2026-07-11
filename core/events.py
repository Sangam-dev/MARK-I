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
