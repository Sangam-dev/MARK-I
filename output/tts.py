"""
Text-to-speech (TTS) module for KANCHA.

Uses edge-tts for speech synthesis and sounddevice for playback.
Implements sentence splitting with overlap optimization:

    S1: [synth]──[play]
    S2:        [synth]──[play]
    S3:               [synth]──[play]

While sentence N plays, N+1 is already synthesizing — zero gap between sentences.
"""

from __future__ import annotations
from core.audio_state import audio_state
import asyncio
import io
import re
import sys
import time

try:
    import edge_tts
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    sys.exit("Run: pip install edge-tts sounddevice soundfile")

import logging

from core.bus import EventBus
from core.events import ResponseReady

logger = logging.getLogger("kancha.output.tts")

# Module-level lock: only one TTS utterance plays at a time
_speaking_lock: asyncio.Lock | None = None


def _get_speaking_lock() -> asyncio.Lock:
    """Lazy init — must be called from async context."""
    global _speaking_lock
    if _speaking_lock is None:
        _speaking_lock = asyncio.Lock()
    return _speaking_lock


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

VOICE = "en-GB-RyanNeural"
MIN_CHUNK_LEN = 10
MAX_CHUNK_LEN = 160
BOUNDARIES = {".", "!", "?", "—", "…"}

ABBREV = re.compile(
    r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|approx|dept|est|govt|inc|ltd)\.$",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING & SENTENCE SPLITTING
# ─────────────────────────────────────────────────────────────────────────────


def _is_abbreviation(text: str, pos: int) -> bool:
    """Check if the period at `pos` is part of an abbreviation."""
    return bool(ABBREV.search(text[: pos + 1]))


def _extract_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Extract complete, speakable sentences from buffer.

    Returns:
        (list of sentences, leftover buffer)
    """
    sentences = []
    while True:
        boundary_pos = -1
        for i, char in enumerate(buffer):
            if char in BOUNDARIES and i >= MIN_CHUNK_LEN:
                if char == "." and _is_abbreviation(buffer, i):
                    continue
                if char == "." and i + 1 < len(buffer) and buffer[i + 1] == ".":
                    continue
                boundary_pos = i
                break

        if boundary_pos == -1 and len(buffer) >= MAX_CHUNK_LEN:
            sp = buffer.rfind(" ", 0, MAX_CHUNK_LEN)
            boundary_pos = sp if sp > MIN_CHUNK_LEN else MAX_CHUNK_LEN - 1

        if boundary_pos == -1:
            break

        sentence = buffer[: boundary_pos + 1].strip()
        buffer = buffer[boundary_pos + 1 :].lstrip()
        if sentence and len(sentence) > 2:
            sentences.append(sentence)

    return sentences, buffer


def _clean(text: str) -> str:
    """Strip markdown and symbols that sound weird when spoken."""
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)  # bold/italic
    text = re.sub(r"`[^`]+`", lambda m: m.group(0)[1:-1], text)  # inline code
    text = re.sub(r"#+\s*", "", text)  # headers
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links
    text = re.sub(r"—", " — ", text)  # em dash spacing
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHESIS & PLAYBACK
# ─────────────────────────────────────────────────────────────────────────────


async def _synthesize(sentence: str) -> tuple | None:
    """
    Synthesize one sentence to audio (data, samplerate).

    Returns:
        (numpy array, samplerate) or None if synthesis failed.
    """
    sentence = _clean(sentence)
    if not sentence.strip():
        return None

    audio_bytes = b""
    tts = edge_tts.Communicate(text=sentence, voice=VOICE, rate="+12%", pitch="-4Hz")
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]

    if not audio_bytes:
        return None

    data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    return data, samplerate


def _play(data, samplerate: int) -> None:
    """Play audio synchronously (blocks until done)."""
    sd.play(data, samplerate)
    sd.wait()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────


async def speak(text: str) -> None:
    """
    Synthesize and play text with overlapped synth/play.
    Acquires a lock so concurrent calls are serialized (no overlapping audio).
    """
    if not text or not text.strip():
        return

    lock = _get_speaking_lock()
    async with lock:
        logger.debug("TTS: speaking %d chars", len(text))
        print(f"\n🔊 Speaking...\n")
        start = time.perf_counter()
        loop = asyncio.get_running_loop()

        # Extract all sentences upfront
        sentences, remainder = _extract_sentences(text.strip())
        if remainder.strip() and len(remainder.strip()) > 2:
            sentences.append(remainder.strip())

        if not sentences:
            return

        audio_future = None

        for i, sentence in enumerate(sentences):
            print(f"  {i + 1}/{len(sentences)}: {sentence}")
            this_future = asyncio.create_task(_synthesize(sentence))

            # While current synthesizes, play previous
            if audio_future is not None:
                audio = await audio_future
                if audio:
                    await loop.run_in_executor(None, _play, *audio)

            audio_future = this_future

        # Play the last one
        if audio_future is not None:
            audio = await audio_future
            if audio:
                await loop.run_in_executor(None, _play, *audio)

        elapsed = time.perf_counter() - start
        print(f"\n  ✓ Done in {elapsed:.2f}s\n")


class TTSHandler:
    """Subscribes to ResponseReady events and speaks the response text."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    def register(self) -> None:
        self._bus.subscribe(ResponseReady, self.on_response_ready)

    async def on_response_ready(self, event: ResponseReady) -> None:
        """
        Handle assistant responses by speaking them.

        While speaking, notify the shared AudioState so that the
        microphone pauses and does not transcribe the assistant's own voice.
        """
        if not event.text or not event.text.strip():
            return

        logger.info("TTS: ResponseReady received, speaking...")

        audio_state.speaking_started()

        try:
            await speak(event.text)
        finally:
            audio_state.speaking_finished()
