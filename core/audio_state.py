"""Shared audio coordination state."""
from __future__ import annotations

import asyncio
import time


DEFAULT_TTS_COOLDOWN_SECS = 0.8


class AudioState:
    """
    Shared audio coordination state.

    Prevents the microphone from listening while the assistant is speaking
    OR while a user turn is still being reasoned about. Both gates must
    be clear before the mic reopens; otherwise a stray utterance (or echo
    of the assistant's own audio) could be captured mid-think and re-fed
    as its own turn — which is what caused "first command ignored, second
    command answered with the first one's context" in voice mode.
    """

    def __init__(self, tts_cooldown_secs: float = DEFAULT_TTS_COOLDOWN_SECS) -> None:
        # Set while TTS is actively playing audio.
        self.tts_active = asyncio.Event()
        self._active_speakers = 0
        self._quiet_until = 0.0
        self._tts_cooldown_secs = max(0.0, tts_cooldown_secs)

        # Set from the moment a user turn is emitted until the response
        # has been fully handed to TTS. Held in addition to ``tts_active``
        # so the mic cannot reopen during NLU classification, the LLM
        # stream, or task/plan execution — only when both gates are clear
        # AND the post-speech cooldown has elapsed does the mic cycle.
        self.thinking_active = asyncio.Event()
        self._active_thinking = 0

    @property
    def is_speaking(self) -> bool:
        """True if TTS is currently speaking."""
        return self.tts_active.is_set()

    @property
    def is_audio_input_blocked(self) -> bool:
        """True while mic input should be ignored to avoid self-transcription
        or capturing a turn that's still being reasoned about."""
        return (
            self.tts_active.is_set()
            or self.thinking_active.is_set()
            or time.monotonic() < self._quiet_until
        )

    def speaking_started(self) -> None:
        """Called before TTS playback begins."""
        self._active_speakers += 1
        self._quiet_until = 0.0
        self.tts_active.set()

    def speaking_finished(self) -> None:
        """Called after TTS playback has completely finished."""
        if self._active_speakers > 0:
            self._active_speakers -= 1
        if self._active_speakers == 0:
            self._quiet_until = time.monotonic() + self._tts_cooldown_secs
            self.tts_active.clear()

    def thinking_started(self) -> None:
        """Mark that a user turn is being reasoned about.

        Called as soon as a transcript/text event is dispatched into the
        pipeline. Counter-balanced by :meth:`thinking_finished`; the gate
        only clears when every in-flight reasoning turn has fully released.
        """
        self._active_thinking += 1
        self.thinking_active.set()

    def thinking_finished(self) -> None:
        """Mark a reasoning turn complete. No-op if no turn is active.

        Safe to call multiple times (e.g., once when ``ResponseReady``
        fires and again as a safety net when TTS actually starts); the
        guard on ``_active_thinking > 0`` matches the speaker pattern
        and prevents underflow.
        """
        if self._active_thinking > 0:
            self._active_thinking -= 1
        if self._active_thinking == 0:
            self.thinking_active.clear()

    async def wait_until_idle(self) -> None:
        """
        Block until both gates have cleared and the post-speech cooldown
        has elapsed.
        """
        while self.is_audio_input_blocked:
            remaining = self._quiet_until - time.monotonic()
            await asyncio.sleep(max(0.02, min(remaining, 0.1)))


# Global singleton used by both STT and TTS.
audio_state = AudioState()
