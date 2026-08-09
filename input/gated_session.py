from __future__ import annotations

import asyncio
import logging
import os
import threading
import time

from core.bus import EventBus
from core.events import (
    AssistantState,
    AssistantStateChanged,
    ResponseReady,
    ShutdownRequested,
    TextInputReceived,
    TranscriptReady,
    WakeWordDetected,
)
from input.stt import MicrophoneListener
from input.wake_word import WakeWordDetector

logger = logging.getLogger("kancha.input.gated_session")

_DEFAULT_IDLE_TIMEOUT = 120  # 2 minutes


class WakeWordGatedSession:
    """Manages the full SLEEPING ↔ ACTIVE voice input lifecycle."""

    def __init__(
        self,
        bus: EventBus,
        session_id: str = "default",
        idle_timeout_secs: int | None = None,
        wake_word_detector: WakeWordDetector | None = None,
    ) -> None:
        self._bus = bus
        self._session_id = session_id
        self._idle_timeout = idle_timeout_secs or int(
            os.getenv("KANCHA_IDLE_TIMEOUT", str(_DEFAULT_IDLE_TIMEOUT))
        )
        self._detector = wake_word_detector or WakeWordDetector()
        self._running = False
        self._last_activity = 0.0
        self._stop_wakeword: threading.Event | None = None

    # ── Activity tracking ──────────────────────────────────────────────────

    def _touch(self) -> None:
        """Reset the idle timer."""
        self._last_activity = time.monotonic()

    def _idle_for(self) -> float:
        return time.monotonic() - self._last_activity

    # ── Bus event handlers ─────────────────────────────────────────────────

    async def _on_transcript_ready(self, event: TranscriptReady) -> None:
        if event.session_id == self._session_id:
            self._touch()

    async def _on_text_input(self, event: TextInputReceived) -> None:
        if event.session_id == self._session_id:
            self._touch()

    async def _on_response_ready(self, event: ResponseReady) -> None:
        if event.session_id == self._session_id:
            self._touch()

    async def _on_shutdown(self, event: ShutdownRequested) -> None:
        self._running = False
        if self._stop_wakeword is not None:
            self._stop_wakeword.set()

    def _register_activity_listeners(self) -> None:
        self._bus.subscribe(TranscriptReady, self._on_transcript_ready)
        self._bus.subscribe(TextInputReceived, self._on_text_input)
        self._bus.subscribe(ResponseReady, self._on_response_ready)
        self._bus.subscribe(ShutdownRequested, self._on_shutdown)

    def _unregister_activity_listeners(self) -> None:
        self._bus.unsubscribe(TranscriptReady, self._on_transcript_ready)
        self._bus.unsubscribe(TextInputReceived, self._on_text_input)
        self._bus.unsubscribe(ResponseReady, self._on_response_ready)
        self._bus.unsubscribe(ShutdownRequested, self._on_shutdown)

    # ── Mic lifecycle helpers ───────────────────────────────────────────────

    async def _start_mic(self) -> tuple[MicrophoneListener, asyncio.Task]:
        mic = MicrophoneListener(
            bus=self._bus,
            session_id=self._session_id,
            wake_word_gated=False,
        )
        mic.register()
        task = asyncio.create_task(mic.run(), name="microphone_active")
        return mic, task

    async def _stop_mic(self, mic: MicrophoneListener, task: asyncio.Task) -> None:
        mic.stop()
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass
        mic.unregister()

    # ── Main loop ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Run the gated session until ShutdownRequested."""
        self._running = True
        self._register_activity_listeners()
        logger.info(
            "WakeWordGatedSession started (idle_timeout=%ds, session=%s)",
            self._idle_timeout,
            self._session_id,
        )

        try:
            while self._running:
                # ── SLEEPING phase ───────────────────────────────────────
                logger.info("Entering SLEEPING state — say 'hey jarvis' to activate")
                self._bus.emit(
                    AssistantStateChanged(
                        state=AssistantState.SLEEPING,
                        session_id=self._session_id,
                    )
                )

                stop_wakeword = threading.Event()
                self._stop_wakeword = stop_wakeword

                detected = await self._detector.listen_for_wake_word(stop_wakeword)

                self._stop_wakeword = None

                if not self._running:
                    break

                if not detected:
                    # stop_event was set externally (shutdown)
                    break

                # ── Transition to ACTIVE ──────────────────────────────────
                logger.info("Wake word detected — entering ACTIVE state")
                self._bus.emit(
                    WakeWordDetected(
                        session_id=self._session_id,
                        confidence=1.0,
                    )
                )
                self._bus.emit(
                    AssistantStateChanged(
                        state=AssistantState.IDLE,
                        session_id=self._session_id,
                    )
                )
                self._touch()  # reset idle clock

                mic, mic_task = await self._start_mic()

                # ── ACTIVE phase: idle-timer loop ────────────────────────
                try:
                    while self._running:
                        await asyncio.sleep(1.0)
                        idle = self._idle_for()
                        if idle >= self._idle_timeout:
                            logger.info(
                                "No activity for %.0fs — returning to sleep",
                                idle,
                            )
                            break
                finally:
                    await self._stop_mic(mic, mic_task)

                # (next iteration will re-emit SLEEPING and restart detector)

        finally:
            self._unregister_activity_listeners()
            logger.info("WakeWordGatedSession stopped")

    def stop(self) -> None:
        """Request a graceful stop (e.g. server shutdown)."""
        self._running = False
        if self._stop_wakeword is not None:
            self._stop_wakeword.set()
