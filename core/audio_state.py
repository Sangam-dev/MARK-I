from __future__ import annotations

import asyncio
import time


DEFAULT_TTS_COOLDOWN_SECS = 0.8


class AudioState:
    """
    Shared audio coordination state.

    Prevents the microphone from listening while TTS is speaking or while
    speaker echo/audio-driver buffers may still be draining.
    """

    def __init__(self, tts_cooldown_secs: float = DEFAULT_TTS_COOLDOWN_SECS) -> None:
        # Set while TTS is actively playing audio.
        self.tts_active = asyncio.Event()
        self._active_speakers = 0
        self._quiet_until = 0.0
        self._tts_cooldown_secs = max(0.0, tts_cooldown_secs)

    @property
    def is_speaking(self) -> bool:
        """True if TTS is currently speaking."""
        return self.tts_active.is_set()

    @property
    def is_audio_input_blocked(self) -> bool:
        """True while mic input should be ignored to avoid self-transcription."""
        return self.tts_active.is_set() or time.monotonic() < self._quiet_until

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

    async def wait_until_idle(self) -> None:
        """
        Block until TTS has finished speaking and the post-speech cooldown has
        elapsed.
        """
        while self.is_audio_input_blocked:
            remaining = self._quiet_until - time.monotonic()
            await asyncio.sleep(max(0.02, min(remaining, 0.1)))


# Global singleton used by both STT and TTS.
audio_state = AudioState()
