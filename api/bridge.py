"""Bridges the KANCHA EventBus to connected WebSocket clients.

This is intentionally the ONLY module in the codebase that knows both "bus
event shapes" (core/events.py) and "WebSocket wire format"
(answers/integration_plan.md, section 3.4) — every domain module (NLU,
reasoning, tasks, memory, TTS, STT) stays fully decoupled from the transport
layer, exactly like it was when the only consumer was the CLI.

`attach_bridge(pipeline)` subscribes a handful of async handlers onto the
pipeline's EventBus once, at server startup. When adding a new backend
feature that the frontend should react to in real time, add a new
`bus.subscribe(...)` line here rather than teaching NLU/reasoning/tasks about
WebSockets. See answers/guide.md for a worked example.
"""

from __future__ import annotations

import logging

from core.events import (
    AppQuitRequested,
    AssistantState,
    AssistantStateChanged,
    PartialResponse,
    ResponseReady,
    SystemError,
    TaskCompleted,
    TextInputReceived,
    TranscriptReady,
)
from core.pipeline import Pipeline

from .ws_manager import manager

logger = logging.getLogger("kancha.api.bridge")


def attach_bridge(pipeline: Pipeline) -> None:
    """Subscribe WS-forwarding handlers onto `pipeline.bus`. Call once per Pipeline."""
    bus = pipeline.bus

    async def _on_state_changed(event: AssistantStateChanged) -> None:
        await manager.broadcast("state", {"state": event.state.value}, event.session_id)

    async def _on_text_input(event: TextInputReceived) -> None:
        # Echo back what the backend received so the UI can render it as a
        # user chat bubble (this event is emitted directly by the /ws
        # handler in server.py for text input, and by TextInputHandler for
        # CLI --text mode).
        await manager.broadcast(
            "transcript", {"text": event.text, "source": "text"}, event.session_id
        )

    async def _on_transcript_ready(event: TranscriptReady) -> None:
        await manager.broadcast(
            "transcript", {"text": event.text, "source": "voice"}, event.session_id
        )

    async def _on_partial_response(event: PartialResponse) -> None:
        # Streamed assistant text — the UI appends `text` until `done` is
        # true, then the authoritative `response` message (from
        # _on_response_ready) finalizes the bubble.
        await manager.broadcast(
            "response_partial",
            {"text": event.text, "done": event.done},
            event.session_id,
        )

    async def _on_response_ready(event: ResponseReady) -> None:
        await manager.broadcast("response", {"text": event.text}, event.session_id)
        if not pipeline.tts_enabled:
            # No TTSHandler is registered to bring the state back to idle —
            # do it here so the orb doesn't stay stuck on "thinking".
            bus.emit(
                AssistantStateChanged(
                    state=AssistantState.IDLE, session_id=event.session_id
                )
            )

    async def _on_task_completed(event: TaskCompleted) -> None:
        # Supplemental structured data for panels like TASKS/AUTOMATION.
        # The spoken/printed reply for this task already goes out separately
        # as a "response" message once ReasoningCoordinator formats it into
        # a ResponseReady event — this is not a duplicate.
        await manager.broadcast(
            "task_update",
            {
                "task_name": event.task_name,
                "success": event.success,
                "result": event.result,
                "error": event.error,
            },
            event.session_id,
        )

    async def _on_system_error(event: SystemError) -> None:
        await manager.broadcast(
            "error",
            {
                "message": event.error_message,
                "recoverable": event.recoverable,
                "source": event.source_module,
            },
            event.session_id,
        )

    async def _on_app_quit_requested(event: AppQuitRequested) -> None:
        # Forward the quit to the frontend over the same channel as every
        # other server->client message. The renderer (App.tsx) waits for the
        # spoken confirmation to finish, then asks the Electron main process
        # to stop the backend and exit.
        await manager.broadcast(
            "app_command", {"command": event.command}, event.session_id
        )

    bus.subscribe(AssistantStateChanged, _on_state_changed)
    bus.subscribe(TextInputReceived, _on_text_input)
    bus.subscribe(TranscriptReady, _on_transcript_ready)
    bus.subscribe(PartialResponse, _on_partial_response)
    bus.subscribe(ResponseReady, _on_response_ready)
    bus.subscribe(TaskCompleted, _on_task_completed)
    bus.subscribe(SystemError, _on_system_error)
    bus.subscribe(AppQuitRequested, _on_app_quit_requested)

    logger.info(
        "WebSocket bridge attached to pipeline bus (session=%s)", pipeline.session_id
    )
