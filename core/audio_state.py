"""Shared audio coordination state."""
from __future__ import annotations

import asyncio
import os
import threading
import time

import numpy as np


DEFAULT_TTS_COOLDOWN_SECS = 0.3

# Reference kept of the exact PCM the speaker played (resampled to the mic
# rate), so STT can detect its own echo by correlation instead of by text.
_MIC_RATE = 16000
_PLAYBACK_REF_SECS = float(os.getenv("KANCHA_ECHO_REF_S", "2.0"))
# How long after playback ends the echo gate stays armed (reverb tail).
_PLAYBACK_TAIL_SECS = float(os.getenv("KANCHA_ECHO_TAIL_S", "0.5"))
# Correlation stride when matching a mic block against the playback buffer.
_CORR_STRIDE = 160
# Normalized cross-correlation above which a mic block is our own voice.
# Kept HIGH (backstop only): on this hardware AEC decorrelates the linear
# echo, so low thresholds also fire on real user speech and would break
# barge-in. The primary self-echo defense is the playback-time volume floor
# in input/stt.py (PLAYBACK_SPEECH_FLOOR); this gate only catches strong
# linear echo leaks (AEC off / imperfect).
_CORR_THRESHOLD = float(os.getenv("KANCHA_ECHO_CORR", "0.5"))


class AudioState:
    """Shared audio coordination state.

    Gates the mic while the assistant is speaking OR while a user turn is
    still being reasoned about, so a stray utterance (or echo) captured
    mid-think can't be re-fed as its own turn.
    """

    def __init__(self, tts_cooldown_secs: float = DEFAULT_TTS_COOLDOWN_SECS) -> None:
        self.tts_active = asyncio.Event()  # set while TTS is playing
        self._active_speakers = 0
        self._quiet_until = 0.0
        self._tts_cooldown_secs = max(0.0, tts_cooldown_secs)

        # Set from user-turn dispatch until the response is handed to TTS,
        # so the mic can't reopen during NLU/LLM/task execution.
        self.thinking_active = asyncio.Event()
        self._active_thinking = 0

        # Recently played sentences (timestamp, text) — the mic could still
        # be echoing these; STT's text guard reads them. Bounded, newest last.
        self.recent_spoken: list[tuple[float, str]] = []
        self._max_recent_spoken = 8

        # Exact played waveform (16k float32 mono) + when last written, for
        # STT's correlation echo gate. Guarded by a lock: ``_play`` feeds it
        # from the executor thread while the PortAudio callback reads it.
        self._playback_ref = np.zeros(0, dtype=np.float32)
        self._last_playback_at = 0.0
        self._playback_lock = threading.Lock()

    def note_spoken(self, text: str) -> None:
        """Record a sentence that actually started playing aloud."""
        text = (text or "").strip()
        if not text:
            return
        self.recent_spoken.append((time.monotonic(), text))
        del self.recent_spoken[:-self._max_recent_spoken]

    def note_playback(self, samples_16k: np.ndarray) -> None:
        """Record the exact samples handed to the speaker (16k mono float32)."""
        if samples_16k is None or samples_16k.size == 0:
            return
        ref = np.asarray(samples_16k, dtype=np.float32).reshape(-1)
        with self._playback_lock:
            buf = np.concatenate([self._playback_ref, ref])
            max_len = int(_MIC_RATE * _PLAYBACK_REF_SECS)
            self._playback_ref = buf[-max_len:]
            self._last_playback_at = time.monotonic()

    @property
    def playback_recent(self) -> bool:
        """True while speaking or within the echo-tail window after it."""
        return self.tts_active.is_set() or (
            time.monotonic() - self._last_playback_at <= _PLAYBACK_TAIL_SECS
        )

    def matches_playback(self, block: np.ndarray) -> bool:
        """True when *block* is a (possibly delayed) copy of our own playback.

        Searches the recent playback buffer for a window that correlates with
        the mic block. Correlated audio is the assistant's own voice — an echo
        the gate should ignore; a real user utterance does not correlate.
        """
        if not self.playback_recent:
            return False
        blk = np.asarray(block, dtype=np.float32).reshape(-1)
        nb = blk.size
        with self._playback_lock:
            ref = self._playback_ref
            if ref.size < nb:
                return False
            blk = blk - blk.mean()
            bnorm = np.linalg.norm(blk)
            if bnorm < 1e-6:
                return False
            for start in range(0, ref.size - nb + 1, _CORR_STRIDE):
                win = ref[start : start + nb] - ref[start : start + nb].mean()
                wnorm = np.linalg.norm(win)
                if wnorm < 1e-6:
                    continue
                corr = float(np.dot(blk, win) / (bnorm * wnorm))
                if corr >= _CORR_THRESHOLD:
                    return True
        return False

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

    def interrupt(self) -> None:
        """Cut all output immediately and release the mic gates (barge-in).

        Clearing the speaker count also makes any later ``speaking_finished``
        a no-op, so a response finishing after the barge-in can't re-block
        the mic during capture of the interrupting utterance.
        """
        self._active_speakers = 0
        self._quiet_until = 0.0
        self.tts_active.clear()

    def thinking_started(self) -> None:
        """Mark a user turn as being reasoned about (counter-balanced by
        ``thinking_finished``; gate clears only when all turns have released)."""
        self._active_thinking += 1
        self.thinking_active.set()

    def thinking_finished(self) -> None:
        """Mark a reasoning turn complete. Safe to call repeatedly; the guard
        on ``_active_thinking > 0`` prevents underflow."""
        if self._active_thinking > 0:
            self._active_thinking -= 1
        if self._active_thinking == 0:
            self.thinking_active.clear()

    async def wait_until_idle(self) -> None:
        """Block until both gates have cleared and the cooldown has elapsed."""
        while self.is_audio_input_blocked:
            remaining = self._quiet_until - time.monotonic()
            await asyncio.sleep(max(0.02, min(remaining, 0.1)))


# Global singleton used by both STT and TTS.
audio_state = AudioState()
