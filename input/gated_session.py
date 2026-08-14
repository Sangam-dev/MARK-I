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
        # Set by toggle_manual() (spacebar in the frontend) to distinguish a
        # deliberate manual trigger from listen_for_wake_word() returning
        # False because of a real shutdown, and to break out of the ACTIVE
        # idle-timer loop early.
        self._manual_wake_requested = False
        self._manual_sleep_requested = False

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

    # ── Manual override (spacebar) ──────────────────────────────────────────
    #
    # openwakeword occasionally mishears or just doesn't catch "hey jarvis"
    # in a noisy room. This gives the frontend a deterministic fallback: one
    # key toggles between SLEEPING and ACTIVE, independent of the mic.

    def toggle_manual(self) -> str:
        """Manually wake or sleep, mirroring whatever the mic would have done.

        Returns ``"waking"`` if we were SLEEPING and are now being woken,
        ``"sleeping"`` if we were ACTIVE and are now being sent back to
        sleep, or ``"ignored"`` if the session is mid-transition and there
        is nothing sensible to do.
        """
        if self._stop_wakeword is not None:
            # Blocked inside listen_for_wake_word() — interrupt it exactly
            # like a real detection, but flag it as manual so run() doesn't
            # mistake this for an external shutdown.
            self._manual_wake_requested = True
            self._stop_wakeword.set()
            return "waking"
        if self._running:
            # ACTIVE (idle/listening/thinking/speaking) — end the idle-timer
            # loop early and fall back to SLEEPING, same as an idle timeout.
            self._manual_sleep_requested = True
            return "sleeping"
        return "ignored"

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
                manual_wake = self._manual_wake_requested
                self._manual_wake_requested = False

                if not self._running:
                    break

                if not detected:
                    if not manual_wake:
                        # stop_event was set externally (shutdown)
                        break
                    # Spacebar (toggle_manual()) interrupted the detector —
                    # treat it exactly like a real wake word.

                # ── Transition to ACTIVE ──────────────────────────────────
                logger.info(
                    "%s — entering ACTIVE state",
                    "Manual wake requested" if manual_wake else "Wake word detected",
                )
                self._bus.emit(
                    WakeWordDetected(
                        session_id=self._session_id,
                        confidence=1.0 if not manual_wake else 0.0,
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

                # ── ACTIVE phase: idle-timer loop ──────────────────
                try:
                    while self._running:
                        await asyncio.sleep(1.0)
                        if self._manual_sleep_requested:
                            self._manual_sleep_requested = False
                            logger.info("Manual sleep requested — returning to sleep")
                            break
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
