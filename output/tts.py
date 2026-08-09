"""
Text-to-speech (TTS) module for KANCHA.

Uses edge-tts for speech synthesis and sounddevice for playback.
Implements sentence splitting with overlap optimization:

    S1: [synth]──[play]
    S2:        [synth]──[play]
    S3:               [synth]──[play]

While sentence N plays, N+1 is already synthesizing — zero gap between sentences.

The *live* output path is :class:`StreamingSpeaker` + :class:`TTSHandler`, which
start speaking complete sentences as ``PartialResponse`` events stream in from
the LLM (audio no longer waits for the full response). The standalone
``speak()`` helper implements the same pipelining for a complete text block
and is kept for backwards compatibility.
"""

from __future__ import annotations

import asyncio
import io
import re
import sys
import time

from core.audio_state import audio_state

try:
    import edge_tts
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    sys.exit("Run: pip install edge-tts sounddevice soundfile")

import logging

from core.bus import EventBus
from core.events import (
    AssistantState,
    AssistantStateChanged,
    PartialResponse,
    ResponseReady,
)

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
# Minimum characters before any boundary can fire. Lowered from 10 to 6 so
# short openings like "It is my absolute pleasure, sir" don't wait on the
# LLM stream for the rest of the sentence before TTS starts. The tail-merge
# in ``on_response_ready`` reconstructs the full sentence at playback time
# so a soft split here is never audible.
MIN_CHUNK_LEN = 6
# Maximum characters before forcing a split. Unchanged at 160 — anything
# past this would have already produced a hard or soft boundary.
MAX_CHUNK_LEN = 160
# Hard boundaries (strong sentence ends) — split as soon as one is found
# past ``MIN_CHUNK_LEN``.
HARD_BOUNDARIES = {".", "!", "?", "…"}
# Soft boundaries (commas, em-dashes) — only used as a fallback split once
# the buffer is comfortably long enough that the next clause is going to
# stand on its own anyway. Em-dash is here because the existing
# ``_clean`` already pads it with spaces, so it's already a natural pause.
SOFT_BOUNDARIES = {",", "—"}
# Minimum buffer length before a soft-boundary split is allowed. Below
# this we keep waiting for a hard boundary or the MAX_CHUNK_LEN fallback
# — splitting on a comma in a 12-char fragment would sound unnatural.
SOFT_BREAK_THRESHOLD = 24
BOUNDARIES = HARD_BOUNDARIES | SOFT_BOUNDARIES

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


# When the buffer has grown past this length, prefer splitting on the
# nearest soft boundary (comma, em-dash) over waiting for a hard
# sentence-final punctuation mark. Without this, a 250-character response
# that ends with a single period waits the entire duration of the LLM
# stream before any TTS fires — the user sees the full text on screen for
# several seconds and then hears the first word. With this, the comma at
# position 60 (long after the first clause) splits the buffer cleanly
# into a 60-char speakable fragment, TTS starts immediately on that
# fragment, and the rest of the response continues streaming into the
# next split.
SOFT_PREFERRED_THRESHOLD = 60


def _extract_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Extract complete, speakable sentences from buffer.

    Returns:
        (list of sentences, leftover buffer)

    The extractor prefers hard sentence boundaries (``.``, ``!``, ``?``,
    ``…``) past :data:`MIN_CHUNK_LEN` — but only as long as the next hard
    boundary is "close enough". Once the buffer has grown past
    :data:`SOFT_PREFERRED_THRESHOLD` characters we instead prefer the
    nearest soft boundary (``,``, ``—``) past :data:`MIN_CHUNK_LEN`. This
    is what makes long single-sentence responses — the kind Gemini likes
    to emit — start speaking at the first comma instead of waiting 4-6
    seconds for the final period. The caller is responsible for stitching
    the soft-split fragments back together at playback time
    (:meth:`TTSHandler.on_response_ready` already does this via the
    ``_visible`` / ``_buf`` merge).

    If the buffer exceeds :data:`MAX_CHUNK_LEN` with no boundary at all, the
    extractor forces a split at the nearest space past ``MIN_CHUNK_LEN``.
    """
    sentences = []
    while True:
        # First pass: hard boundary — but only if it isn't too far away.
        # If the buffer is past SOFT_PREFERRED_THRESHOLD, a hard boundary
        # 200 chars away means the user will wait 4+ seconds; prefer a
        # soft split at 60 chars instead.
        boundary_pos = -1
        for i, char in enumerate(buffer):
            if char in HARD_BOUNDARIES and i >= MIN_CHUNK_LEN:
                if char == "." and _is_abbreviation(buffer, i):
                    continue
                if char == "." and i + 1 < len(buffer) and buffer[i + 1] == ".":
                    continue
                # Mid-token period: decimal numbers (10.7), URLs (www.facebook.com),
                # version numbers (v1.2.3), domain names (python.org).
                # A sentence-ending period is always followed by whitespace or
                # end-of-buffer — anything else is not a sentence boundary.
                if char == "." and i + 1 < len(buffer) and not buffer[i + 1].isspace():
                    continue
                if (
                    len(buffer) >= SOFT_PREFERRED_THRESHOLD
                    and i > SOFT_PREFERRED_THRESHOLD
                ):
                    break
                boundary_pos = i
                break

        # Second pass: soft boundary fallback. Fires when no hard boundary
        # was found AND when we deferred a far-off hard boundary in favour
        # of an earlier soft one.
        if boundary_pos == -1 and len(buffer) >= SOFT_BREAK_THRESHOLD:
            for i, char in enumerate(buffer):
                if char in SOFT_BOUNDARIES and i >= MIN_CHUNK_LEN:
                    boundary_pos = i
                    break

        # Final fallback: force a split at the last space if the buffer
        # has exceeded MAX_CHUNK_LEN with no usable boundary at all.
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


class StreamingSpeaker:
    """Pipelined sentence TTS that spans multiple events.

    Sentences are pushed as soon as they complete during streaming. A single
    player task plays them back-to-back the moment each one's audio becomes
    ready, so the *first* sentence starts playing as soon as its synthesis
    finishes — no waiting for the full response or even for a second
    sentence. Synthesis of later sentences overlaps playback of earlier
    ones because they are queued ahead of the player.
    """

    def __init__(self) -> None:
        # Lazily created on first use (asyncio primitives bind to a loop).
        self._lock: asyncio.Lock | None = None
        self._queue: asyncio.Queue[asyncio.Task | None] | None = None
        self._player: asyncio.Task | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _ensure_player(self) -> None:
        if self._player is None or self._player.done():
            self._queue = asyncio.Queue()
            self._player = asyncio.create_task(self._player_loop())

    async def _player_loop(self) -> None:
        """Consume synthesis tasks in order; play each as it becomes ready.

        A single failure (synthesis or playback) is logged and skipped so it
        can never abort the rest of the stream. The loop only exits on the
        ``None`` sentinel pushed by :meth:`drain`.
        """
        while True:
            item = await self._queue.get()
            if item is None:
                break
            try:
                audio = await item
                if audio is not None:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, _play, *audio)
            except Exception as exc:
                logger.warning("TTS sentence failed (skipping): %s", exc)
                continue

    async def push(self, sentence: str) -> None:
        """Queue a complete sentence for synthesis+playback (non-blocking)."""
        if not sentence or not sentence.strip():
            return
        async with self._get_lock():
            self._ensure_player()
            this = asyncio.create_task(_synthesize(sentence))
            await self._queue.put(this)

    async def drain(self) -> None:
        """Stop the player and block until all queued audio has finished."""
        async with self._get_lock():
            if self._player is None or self._player.done():
                # Player only exits via the sentinel, so a done player means
                # every queued sentence already played — nothing to discard.
                self._player = None
                self._queue = None
                return
            await self._queue.put(None)
            await self._player
            self._player = None
            self._queue = None


class TTSHandler:
    """Speaks assistant responses, starting while they are still streaming.

    Subscribes to ``PartialResponse`` so complete sentences begin playing
    as soon as they are generated (audio no longer waits for the full LLM
    response), and to ``ResponseReady`` for the authoritative final text
    (task responses never stream) plus the IDLE state transition.

    While speaking, the shared AudioState is notified so the microphone
    pauses and does not transcribe the assistant's own voice.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._speaker = StreamingSpeaker()
        # Serializes a whole utterance (buffer + state) end to end so a
        # concurrent response can never clobber an in-flight one. Overlap
        # is preserved: pushes queue on the speaker's own lock, and each
        # push still starts the next synthesis before playing the previous.
        self._utterance_lock: asyncio.Lock | None = None
        self._buf = ""
        self._visible = ""
        self._session_id = "default"
        self._started = False

    def register(self) -> None:
        self._bus.subscribe(PartialResponse, self.on_partial)
        self._bus.subscribe(ResponseReady, self.on_response_ready)

    async def warmup(self) -> None:
        """Pre-establish the edge-tts connection so the first real
        synthesis doesn't pay the cold-connection cost.

        Builds a :class:`edge_tts.Communicate` with a single throwaway
        word, drains the audio stream, and discards the bytes. The
        tricky bit is that ``edge_tts`` spins up a fresh WebSocket per
        ``Communicate`` instance, so the only way to amortise the
        handshake is to *do* one and discard the result. The first real
        sentence the user actually hears will then arrive ~200–400ms
        sooner than it would otherwise.

        Safe to call at any time; idempotent. Errors are logged and
        swallowed — a failed warmup must never break pipeline startup.
        """
        try:
            logger.debug("TTS: warmup — establishing edge-tts session")
            tts = edge_tts.Communicate(
                text="ready", voice=VOICE, rate="+12%", pitch="-4Hz"
            )
            async for chunk in tts.stream():
                # We don't need the audio — just consume the stream so
                # the underlying WebSocket completes its handshake.
                if chunk.get("type") not in ("audio",):
                    continue
                # Break out as soon as the first audio chunk arrives;
                # the connection is warm, we don't need to download the
                # whole throwaway word.
                break
            logger.debug("TTS: warmup complete")
        except Exception as exc:
            logger.warning("TTS warmup failed (non-fatal): %s", exc)

    def _get_utterance_lock(self) -> asyncio.Lock:
        if self._utterance_lock is None:
            self._utterance_lock = asyncio.Lock()
        return self._utterance_lock

    # ── Streaming ───────────────────────────────────────────────────────────

    async def on_partial(self, event: PartialResponse) -> None:
        async with self._get_utterance_lock():
            if event.done:
                # Stream ended. Do NOT flush the buffered fragment here —
                # ResponseReady follows immediately with the authoritative
                # text, and it merges the fragment with the final tail (a
                # mid-word stream end would otherwise be spoken broken).
                return
            if not event.text:
                return
            self._session_id = event.session_id
            self._visible += event.text
            self._buf += event.text
            await self._speak_complete_sentences()

    async def on_response_ready(self, event: ResponseReady) -> None:
        async with self._get_utterance_lock():
            if not event.text or not event.text.strip():
                # Nothing to say — still close out an in-flight utterance so
                # the state machine can never get stuck in SPEAKING.
                if self._started:
                    await self._finish_utterance(event.session_id)
                return

            logger.info(
                "TTS: ResponseReady received (%d chars), speaking...",
                len(event.text),
            )

            self._session_id = event.session_id
            # Proactively claim the audio gate BEFORE the coordinator
            # releases the thinking gate. There is otherwise a tiny
            # window between ``ResponseReady`` (which releases the
            # thinking gate) and ``_mark_speaking`` (which sets the
            # speaking gate via the first sentence boundary) during
            # which the mic could slip in. Claiming here closes it.
            if not self._started:
                await self._mark_speaking()

            # Authoritative text: speak only what streaming hasn't covered.
            final = event.text.strip()
            visible = self._visible.strip()
            if visible and final.startswith(visible):
                # The streamed text is a prefix of the final message. The
                # buffered fragment is the unspoken tail of the stream, which
                # continues seamlessly into final — merge the two so the last
                # word is never spoken broken (e.g. "cond sentence!").
                tail = self._buf + final[len(visible) :]
                self._buf = ""
            else:
                # Divergent (rare fallback): speak only the text after the
                # longest common prefix; drop the stale buffered fragment.
                common = 0
                limit = min(len(final), len(visible))
                while common < limit and final[common] == visible[common]:
                    common += 1
                self._buf = ""
                tail = final[common:]
            if tail:
                self._buf = tail
                await self._speak_complete_sentences()
            await self._flush_remainder()
            await self._finish_utterance(event.session_id)

    # ── Internals ────────────────────────────────────────────────────────────

    async def _speak_complete_sentences(self) -> None:
        sentences, self._buf = _extract_sentences(self._buf)
        for sentence in sentences:
            await self._mark_speaking()
            await self._speaker.push(sentence)

    async def _flush_remainder(self) -> None:
        remainder = self._buf.strip()
        self._buf = ""
        if remainder and len(remainder) > 2:
            await self._mark_speaking()
            await self._speaker.push(remainder)

    async def _mark_speaking(self) -> None:
        if self._started:
            return
        self._started = True
        audio_state.speaking_started()
        self._bus.emit(
            AssistantStateChanged(
                state=AssistantState.SPEAKING, session_id=self._session_id
            )
        )

    async def _finish_utterance(self, session_id: str) -> None:
        await self._speaker.drain()
        if self._started:
            audio_state.speaking_finished()
            self._bus.emit(
                AssistantStateChanged(state=AssistantState.IDLE, session_id=session_id)
            )
        self._buf = ""
        self._visible = ""
        self._started = False
