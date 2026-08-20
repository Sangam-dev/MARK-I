"""
Text-to-speech (TTS) module for KANCHA.

Kokoro (local ONNX) synthesizes offline; sounddevice plays. Sentences are
split and pipelined so N+1 synthesizes while N plays. The live path
(StreamingSpeaker + TTSHandler) speaks complete sentences as PartialResponse
events stream in; ``speak()`` does the same for a complete text block.
"""

from __future__ import annotations

import asyncio
import ctypes
import glob
import os
import re
import sys
import threading
import time

import numpy as np

from core.aec import ensure_routing as _ensure_aec_routing
from core.audio_state import audio_state

try:
    from kokoro_onnx import Kokoro
    import onnxruntime as ort
    import sounddevice as sd
except ImportError:
    sys.exit(
        "Run: pip install kokoro-onnx sounddevice  "
        "(plus the Kokoro model files — see output/tts.py KOKORO_MODEL_DIR)"
    )

import logging

from core.bus import EventBus
from core.events import (
    AssistantState,
    AssistantStateChanged,
    PartialResponse,
    ResponseReady,
    TranscriptReady,
    UserInterrupted,
)

logger = logging.getLogger("kancha.output.tts")

_speaking_lock: asyncio.Lock | None = None


def _get_speaking_lock() -> asyncio.Lock:
    """Lazy init — must be called from async context."""
    global _speaking_lock
    if _speaking_lock is None:
        _speaking_lock = asyncio.Lock()
    return _speaking_lock


VOICE = os.getenv("KANCHA_TTS_VOICE", "bm_daniel")
TTS_SPEED = float(os.getenv("KANCHA_TTS_SPEED", "1.3"))
# Where kokoro-v1.0.onnx and voices-v1.0.bin live (gitignored via **/data/).
KOKORO_MODEL_DIR = os.getenv(
    "KANCHA_TTS_MODEL_DIR",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tts", "data"
    ),
)
# Sentence-split thresholds: hard boundaries (sentence ends) past
# MIN_CHUNK_LEN; soft boundaries (comma/em-dash) as fallback so a long
# single sentence starts speaking at its first clause.
MIN_CHUNK_LEN = 6
MAX_CHUNK_LEN = 160
HARD_BOUNDARIES = {".", "!", "?", "…"}
SOFT_BOUNDARIES = {",", "—"}
SOFT_BREAK_THRESHOLD = 24
BOUNDARIES = HARD_BOUNDARIES | SOFT_BOUNDARIES

# First-sentence early split: the opening words of a reply start speaking as
# soon as ~this many chars arrive, even without any punctuation, instead of
# waiting for the model to finish the full first sentence. Applied only to the
# first sentence of an utterance; the rest keep normal sentence granularity.
EARLY_SPLIT_MIN_LEN = int(os.getenv("KANCHA_EARLY_SPLIT_MIN_LEN", "10"))
EARLY_SPLIT_CHARS = " \t,;—"

ABBREV = re.compile(
    r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|approx|dept|est|govt|inc|ltd)\.$",
    re.IGNORECASE,
)


def _is_abbreviation(text: str, pos: int) -> bool:
    """True if the period at `pos` is part of an abbreviation."""
    return bool(ABBREV.search(text[: pos + 1]))


SOFT_PREFERRED_THRESHOLD = 60

# If no real user transcript arrives within this window after a barge-in,
# the interruption was false (noise) and the unspoken tail is resumed.
FALSE_INTERRUPTION_TIMEOUT = float(
    os.getenv("KANCHA_FALSE_INTERRUPTION_TIMEOUT", "3.0")
)


def _extract_sentences(
    buffer: str, early_first: bool = False
) -> tuple[list[str], str]:
    """
    Extract complete speakable sentences, preferring hard boundaries (., !, ?)
    past MIN_CHUNK_LEN. Once the buffer passes SOFT_PREFERRED_THRESHOLD,
    prefer the nearest soft boundary (comma/em-dash) so long single-sentence
    responses start speaking at the first clause. Falls back to the deferred
    hard boundary, then to a forced split at MAX_CHUNK_LEN.

    With ``early_first``, the very first split may also happen at a plain word
    boundary once the buffer passes EARLY_SPLIT_MIN_LEN — no punctuation
    required — so the opening words of a streaming reply start speaking
    immediately. Only the first sentence is affected; subsequent ones still
    wait for real boundaries.
    """
    sentences = []
    first = True
    while True:
        # Hard boundary pass; a far-off one past SOFT_PREFERRED_THRESHOLD is
        # deferred so the soft pass can split earlier instead.
        boundary_pos = -1
        deferred_hard = -1
        for i, char in enumerate(buffer):
            if char in HARD_BOUNDARIES and i >= MIN_CHUNK_LEN:
                if char == "." and _is_abbreviation(buffer, i):
                    continue
                if char == "." and i + 1 < len(buffer) and buffer[i + 1] == ".":
                    continue
                # A period is only a boundary when followed by whitespace/end
                # (not inside 5.9, a URL, a version, or a domain).
                if char == "." and i + 1 < len(buffer) and not buffer[i + 1].isspace():
                    continue
                if (
                    len(buffer) >= SOFT_PREFERRED_THRESHOLD
                    and i > SOFT_PREFERRED_THRESHOLD
                ):
                    deferred_hard = i
                    break
                boundary_pos = i
                break

        # Soft boundary fallback (or the deferred hard one if none found).
        if boundary_pos == -1 and len(buffer) >= SOFT_BREAK_THRESHOLD:
            soft_pos = -1
            for i, char in enumerate(buffer):
                if char in SOFT_BOUNDARIES and i >= MIN_CHUNK_LEN:
                    soft_pos = i
                    break
            if soft_pos != -1:
                boundary_pos = (
                    soft_pos
                    if deferred_hard == -1
                    else min(soft_pos, deferred_hard)
                )
            else:
                boundary_pos = deferred_hard

        # Early first split: no punctuation yet, but enough text to start.
        if (
            boundary_pos == -1
            and early_first
            and first
            and len(buffer) >= EARLY_SPLIT_MIN_LEN
        ):
            for i in range(EARLY_SPLIT_MIN_LEN, len(buffer)):
                if buffer[i] in EARLY_SPLIT_CHARS:
                    boundary_pos = i
                    break

        # Forced split at the last space past MAX_CHUNK_LEN.
        if boundary_pos == -1 and len(buffer) >= MAX_CHUNK_LEN:
            sp = buffer.rfind(" ", 0, MAX_CHUNK_LEN)
            boundary_pos = sp if sp > MIN_CHUNK_LEN else MAX_CHUNK_LEN - 1

        if boundary_pos == -1:
            break

        sentence = buffer[: boundary_pos + 1].strip()
        buffer = buffer[boundary_pos + 1 :].lstrip()
        first = False
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
    # "5.5" → "5 point 5" (digits spaced) so decimals aren't read as syllables.
    text = re.sub(
        r"\b\d+\.\d+(?:\.\d+)*\b",
        lambda m: " point ".join(" ".join(d) for d in m.group(0).split(".")),
        text,
    )
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


_kokoro: Kokoro | None = None
_kokoro_synth_lock: asyncio.Lock | None = None


def _preload_nvidia_libs() -> None:
    """Preload the pip-installed NVIDIA CUDA/cuDNN runtime libraries.

    onnxruntime-gpu's CUDA provider is dlopen'd lazily when the session is
    created, and it does NOT search ``site-packages/nvidia/*/lib`` on its
    own — without these libraries in the loader's view the provider fails
    with ``libcublasLt.so.13: cannot open shared object file`` and silently
    falls back to CPU. Loading them here with ``RTLD_GLOBAL`` (before the
    session exists) makes them visible to the provider's dlopen. Requires
    the matching nvidia wheels installed (see pyproject.toml).
    """
    import site

    site_packages = site.getsitepackages()
    lib_dirs: list[str] = []
    for sp in site_packages:
        lib_dirs.extend(
            glob.glob(os.path.join(sp, "nvidia", "cu13", "lib"))
            + glob.glob(os.path.join(sp, "nvidia", "cudnn", "lib"))
        )
    for so in sorted(
        p for d in lib_dirs for p in glob.glob(os.path.join(d, "*.so*"))
    ):
        try:
            ctypes.CDLL(so, mode=os.RTLD_GLOBAL)
        except OSError:
            # A lib whose own deps aren't loadable yet — the next dlopen
            # pass (or the session itself) will resolve it; harmless.
            pass


def _get_kokoro() -> Kokoro:
    """Load (once) the local Kokoro model; warmed up at boot via TTSHandler.warmup.

    The ONNX session is built explicitly so TTS runs on the GPU when
    onnxruntime-gpu is installed: kokoro-onnx's own detection probes for
    a module named ``onnxruntime-gpu``, which never exists (the GPU wheel
    installs the ``onnxruntime`` module), so it would silently stay on
    CPU. We prefer CUDA (then TensorRT) and fall back to CPU only when
    no GPU provider is usable.
    """
    global _kokoro
    if _kokoro is None:
        model = os.path.join(KOKORO_MODEL_DIR, "kokoro-v1.0.onnx")
        voices = os.path.join(KOKORO_MODEL_DIR, "voices-v1.0.bin")
        if not os.path.exists(model) or not os.path.exists(voices):
            raise FileNotFoundError(
                f"Kokoro models not found in {KOKORO_MODEL_DIR}. Download "
                "kokoro-v1.0.onnx and voices-v1.0.bin and set "
                "KANCHA_TTS_MODEL_DIR (default ./tts/data)."
            )
        _preload_nvidia_libs()
        available = ort.get_available_providers()
        preferred = [
            p
            for p in ("CUDAExecutionProvider", "TensorrtExecutionProvider")
            if p in available
        ]
        selected = [*preferred, "CPUExecutionProvider"] if preferred else [
            "CPUExecutionProvider"
        ]
        session = ort.InferenceSession(model, providers=selected)
        logger.info(
            "Kokoro ONNX session using providers: %s (available: %s)",
            session.get_providers(),
            available,
        )
        _kokoro = Kokoro.from_session(session, voices)
    return _kokoro


def _get_kokoro_synth_lock() -> asyncio.Lock:
    """Lazy init — must be called from async context."""
    global _kokoro_synth_lock
    if _kokoro_synth_lock is None:
        _kokoro_synth_lock = asyncio.Lock()
    return _kokoro_synth_lock


def _synth_kokoro(sentence: str) -> tuple:
    """Synchronous Kokoro synthesis (runs in a worker thread)."""
    kokoro = _get_kokoro()
    samples, samplerate = kokoro.create(sentence, voice=VOICE, speed=TTS_SPEED)
    return samples, samplerate


async def _synthesize(sentence: str) -> tuple | None:
    """Synthesize one sentence to audio via Kokoro (worker thread).

    The ONNX session isn't safe for concurrent ``create`` calls, so all
    synthesis is serialized behind a lock. Returns ``(data, samplerate)``
    or ``None`` if synthesis failed.
    """
    sentence = _clean(sentence)
    if not sentence.strip():
        return None

    try:
        async with _get_kokoro_synth_lock():
            data, samplerate = await asyncio.to_thread(_synth_kokoro, sentence)
        return data, samplerate
    except Exception as exc:
        logger.warning("Kokoro synthesis failed (skipping): %s", exc)
        return None


# Set on barge-in so the poll loop in _play cuts playback early.
_playback_interrupted = threading.Event()


def _resample_to_16k(data: np.ndarray, samplerate: int) -> np.ndarray:
    """Linear-resample a playback buffer down to the mic rate (16 kHz)."""
    if samplerate == 16000 or data.size == 0:
        return np.asarray(data, dtype=np.float32)
    n = max(1, int(len(data) * 16000 / samplerate))
    x = np.linspace(0, len(data) - 1, n)
    return np.interp(x, np.arange(len(data)), data).astype(np.float32)


# Cap playback peaks so the speaker never clips — clipping distorts
# nonlinearly, which neither the AEC filter nor the correlation gate models.
_PLAYBACK_PEAK = float(os.getenv("KANCHA_PLAYBACK_PEAK", "0.85"))


def _limit(data: np.ndarray) -> np.ndarray:
    peak = float(np.abs(data).max()) if data.size else 0.0
    return data * (_PLAYBACK_PEAK / peak) if peak > _PLAYBACK_PEAK else data


def _play(data, samplerate: int) -> None:
    """Play audio, polling so a barge-in can cut it deterministically.

    This executor thread is the ONLY one that stops/closes the playback
    stream. The event-loop thread only sets :data:`_playback_interrupted`
    (see :func:`_stop_playback`), so there is never a concurrent PortAudio
    close — the most likely cause of a mid-barge-in freeze.
    """
    _playback_interrupted.clear()
    data = _limit(np.asarray(data, dtype=np.float32))
    # Record the exact played waveform (mic rate) so STT's echo gate can
    # correlate the mic against it, and re-assert AEC routing so the
    # playback is inside the echo reference. Both are best-effort.
    audio_state.note_playback(_resample_to_16k(data, samplerate))
    _ensure_aec_routing()
    stream = None
    try:
        sd.play(data, samplerate)
        stream = sd.get_stream()
        while stream is not None and stream.active:
            if _playback_interrupted.is_set():
                break
            time.sleep(0.02)
    except Exception:  # noqa: BLE001
        pass
    finally:
        _playback_interrupted.clear()
        if stream is not None:
            try:
                stream.abort()
                stream.close()
            except Exception:  # noqa: BLE001
                pass


def _stop_playback() -> None:
    """Request an immediate cut of the current TTS playback.

    Only sets a flag (no PortAudio call on the event-loop thread); the poll
    loop in ``_play`` aborts/closes the stream within ~20ms.
    """
    _playback_interrupted.set()


async def speak(text: str) -> None:
    """Synthesize and play text with overlapped synth/play (serialized)."""
    if not text or not text.strip():
        return

    lock = _get_speaking_lock()
    async with lock:
        logger.debug("TTS: speaking %d chars", len(text))
        print("\n🔊 Speaking...\n")
        start = time.perf_counter()
        loop = asyncio.get_running_loop()

        sentences, remainder = _extract_sentences(text.strip())
        if remainder.strip() and len(remainder.strip()) > 2:
            sentences.append(remainder.strip())

        if not sentences:
            return

        audio_future: tuple[str, asyncio.Task] | None = None

        for index, sentence in enumerate(sentences):
            print(f"  {index + 1}/{len(sentences)}: {sentence}")
            this_future = (sentence, asyncio.create_task(_synthesize(sentence)))

            # While the current sentence synthesizes, play the previous one.
            if audio_future is not None:
                prev_sentence, prev_task = audio_future
                audio = await prev_task
                if audio:
                    audio_state.note_spoken(prev_sentence)
                    await loop.run_in_executor(None, _play, *audio)

            audio_future = this_future

        if audio_future is not None:
            prev_sentence, prev_task = audio_future
            audio = await prev_task
            if audio:
                audio_state.note_spoken(prev_sentence)
                await loop.run_in_executor(None, _play, *audio)

        elapsed = time.perf_counter() - start
        print(f"\n  ✓ Done in {elapsed:.2f}s\n")


class StreamingSpeaker:
    """Pipelined sentence TTS: sentences play back-to-back as each becomes
    ready, with later synthesis overlapping earlier playback."""

    def __init__(self) -> None:
        # Lazily created on first use (asyncio primitives bind to a loop).
        self._lock: asyncio.Lock | None = None
        self._queue: asyncio.Queue[tuple[str, asyncio.Task | None] | None] | None = None
        self._player: asyncio.Task | None = None
        # Monotonic time the first response token arrived; cleared once the
        # first audio of the utterance actually starts playing.
        self.first_token_at: float | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _ensure_player(self) -> None:
        if self._player is None or self._player.done():
            self._queue = asyncio.Queue()
            self._player = asyncio.create_task(self._player_loop())

    async def _player_loop(self) -> None:
        """Consume synthesis tasks in order, playing each when ready; exits
        on the ``None`` sentinel pushed by :meth:`drain`."""
        while True:
            item = await self._queue.get()
            if item is None:
                break
            sentence, task = item
            if task is None:
                continue
            try:
                audio = await task
                if audio is not None:
                    if self.first_token_at is not None:
                        dt = time.perf_counter() - self.first_token_at
                        print(f"⏱ First audio in {dt * 1000:.0f}ms after first token")
                        self.first_token_at = None
                    audio_state.note_spoken(sentence)
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
            await self._queue.put((sentence, this))

    async def drain(self) -> None:
        """Stop the player and block until all queued audio has finished."""
        async with self._get_lock():
            self.first_token_at = None
            player = self._player
            if player is None or player.done():
                self._player = None
                self._queue = None
                return
            await self._queue.put(None)
            try:
                await player
            except asyncio.CancelledError:
                # Barge-in already cut the audio — treat the drain as complete.
                pass
            self._player = None
            self._queue = None

    def interrupt(self) -> None:
        """Cut playback immediately and discard everything queued (safe from
        the event-loop thread; leaves the mic's input stream untouched)."""
        self.first_token_at = None
        _stop_playback()
        player = self._player
        self._player = None
        queue = self._queue
        self._queue = None
        if player is not None and not player.done():
            player.cancel()
        if queue is not None:
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is not None and item[1] is not None:
                    item[1].cancel()


class TTSHandler:
    """Speaks assistant responses, starting while they are still streaming.

    Subscribes to ``PartialResponse`` (speak complete sentences as they are
    generated) and ``ResponseReady`` (authoritative final text + IDLE state).
    The shared AudioState gates the mic while speaking.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._speaker = StreamingSpeaker()
        # Serializes a whole utterance end-to-end so concurrent responses
        # can't clobber each other.
        self._utterance_lock: asyncio.Lock | None = None
        self._buf = ""
        self._visible = ""
        self._session_id = "default"
        self._started = False
        # While set (after a barge-in), in-flight chunks of the interrupted
        # turn are discarded; cleared by the next TranscriptReady.
        self._discard_until_next_turn = False
        # False-interruption recovery: if no real user transcript arrives
        # within FALSE_INTERRUPTION_TIMEOUT, resume the unspoken tail.
        self._resume_text = ""
        self._resume_task: asyncio.Task | None = None

    def register(self) -> None:
        self._bus.subscribe(PartialResponse, self.on_partial)
        self._bus.subscribe(ResponseReady, self.on_response_ready)
        self._bus.subscribe(UserInterrupted, self.on_barge_in)
        self._bus.subscribe(TranscriptReady, self.on_transcript_ready)

    async def warmup(self) -> None:
        """Pre-load the local Kokoro model so the first synthesis doesn't pay
        the ~1s ONNX load on the user's critical path. Idempotent and
        non-fatal."""
        try:
            logger.debug("TTS: warmup — loading Kokoro model")
            await asyncio.to_thread(_get_kokoro)
            logger.debug("TTS: warmup complete (Kokoro loaded)")
        except Exception as exc:
            logger.warning("TTS warmup failed (non-fatal): %s", exc)

    def _get_utterance_lock(self) -> asyncio.Lock:
        if self._utterance_lock is None:
            self._utterance_lock = asyncio.Lock()
        return self._utterance_lock

    def _note_first_token(self) -> None:
        """Stamp the moment the first response token arrives (per utterance)."""
        if self._speaker.first_token_at is None:
            self._speaker.first_token_at = time.perf_counter()

    async def on_partial(self, event: PartialResponse) -> None:
        async with self._get_utterance_lock():
            if self._discard_until_next_turn:
                return  # stale chunk from the interrupted turn
            if event.done:
                # Don't flush the buffered fragment here — ResponseReady
                # follows immediately and merges it with the final tail.
                return
            if not event.text:
                return
            self._session_id = event.session_id
            self._note_first_token()
            self._visible += event.text
            self._buf += event.text
            await self._speak_complete_sentences()

    async def on_response_ready(self, event: ResponseReady) -> None:
        async with self._get_utterance_lock():
            if self._discard_until_next_turn:
                return  # authoritative tail of the interrupted turn — close out
            if not event.text or not event.text.strip():
                if self._started:
                    await self._finish_utterance(event.session_id)
                return

            logger.info(
                "TTS: ResponseReady received (%d chars), speaking...",
                len(event.text),
            )

            self._session_id = event.session_id
            # Claim the speaking gate before the coordinator releases the
            # thinking gate, so the mic can't slip in between the two.
            if not self._started:
                await self._mark_speaking()
            self._note_first_token()

            # Speak only what streaming hasn't covered. When the streamed text
            # is a prefix of the final, merge the buffered fragment with the
            # tail so the last word is never spoken broken; otherwise use the
            # longest common prefix.
            final = event.text.strip()
            visible = self._visible.strip()
            if visible and final.startswith(visible):
                tail = self._buf + final[len(visible) :]
                self._buf = ""
            else:
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

    async def on_barge_in(self, event: UserInterrupted) -> None:
        """User spoke over the assistant — cut the audio immediately.

        Deliberately does NOT take the utterance lock, so it can cancel a
        player task that ``on_response_ready`` may be blocked draining. The
        unspoken tail is kept: if no real user transcript arrives within
        FALSE_INTERRUPTION_TIMEOUT the interruption was noise and the tail
        is resumed.
        """
        resume_text = self._buf
        self._speaker.interrupt()
        self._buf = ""
        self._visible = ""
        self._started = False
        self._discard_until_next_turn = True
        if self._resume_task is not None and not self._resume_task.done():
            self._resume_task.cancel()
        self._resume_text = resume_text
        if resume_text and len(resume_text.strip()) > 2:
            self._resume_task = asyncio.create_task(
                self._resume_if_no_turn(), name="tts_false_interruption_resume"
            )
        else:
            self._resume_task = None
        audio_state.interrupt()
        self._bus.emit(
            AssistantStateChanged(state=AssistantState.IDLE, session_id=self._session_id)
        )

    async def on_transcript_ready(self, event: TranscriptReady) -> None:
        """A real user turn landed — accept responses again and cancel any
        pending false-interruption resume."""
        if self._discard_until_next_turn:
            self._discard_until_next_turn = False
            if self._resume_task is not None:
                if not self._resume_task.done():
                    self._resume_task.cancel()
                    self._speaker.interrupt()
                self._resume_task = None
            self._resume_text = ""

    async def _resume_if_no_turn(self) -> None:
        """After FALSE_INTERRUPTION_TIMEOUT with no new turn, the interruption
        was noise — speak the unspoken tail."""
        try:
            await asyncio.sleep(FALSE_INTERRUPTION_TIMEOUT)
            text = self._resume_text.strip()
            self._resume_text = ""
            if not text or len(text) <= 2:
                return
            await self._mark_speaking()
            await self._speaker.push(text)
            await self._finish_utterance(self._session_id)
        finally:
            self._resume_task = None

    async def _speak_complete_sentences(self) -> None:
        # early_first only applies to the very first sentence of an utterance:
        # once anything has been pushed (_started), normal granularity resumes.
        sentences, self._buf = _extract_sentences(
            self._buf, early_first=not self._started
        )
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
