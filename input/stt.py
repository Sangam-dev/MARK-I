import asyncio
import io
import os
import re
import time
import wave
from difflib import SequenceMatcher
from typing import Awaitable, Callable

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from core.aec import ensure_routing as _ensure_aec_routing
from core.aec import prepare as _prepare_aec
from core.aec import restore as _restore_aec
from core.audio_state import audio_state
from core.bus import EventBus
from core.events import (
    AssistantState,
    AssistantStateChanged,
    PartialTranscriptReady,
    ShutdownRequested,
    TranscriptReady,
    UserInterrupted,
    WakeWordDetected,
)

import logging as _stt_logging

_stt_logger = _stt_logging.getLogger("kancha.input.stt")

try:
    from groq import Groq
except ImportError as _groq_err:
    raise ImportError(
        "groq package is required for voice mode. "
        "Install it with: uv add groq  (or: pip install groq)"
    ) from _groq_err

load_dotenv()

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.03
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)  # now actually used as blocksize
SILENCE_THRESH = 0.01  # fallback default if calibration fails
# Silero VAD (Phase 2) model lives alongside Kokoro — gitignored via **/data/.
VAD_MODEL_DIR = os.getenv(
    "KANCHA_VAD_MODEL_DIR",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tts", "data"
    ),
)
VAD_MODEL_URL = os.getenv(
    "KANCHA_VAD_MODEL_URL",
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
)
# Trailing silence (s) that ends a recording.
SILENCE_LIMIT = float(os.getenv("KANCHA_SILENCE_LIMIT", "0.5"))
MAX_DURATION = 15.0
MIN_SPEECH_SECS = 0.5
MAX_HISTORY = 12
SPEECH_START_SECS = 0.24
LOW_CONFIDENCE_AUDIO_RMS = 0.025
LOW_CONFIDENCE_AUDIO_SECS = 1.2

# Preemptive (interim ASR) generation: a cheap Whisper races the final one,
# firing MID-speech on a snapshot so the LLM can start replying while the
# utterance is still being captured. The result is only ever reused when the
# coordinator validates it against the authoritative transcript.
PREEMPTIVE_MODEL = os.getenv("KANCHA_PREEMPTIVE_MODEL", "whisper-large-v3-turbo")
PREEMPTIVE_SNAPSHOT_SECS = float(os.getenv("KANCHA_PREEMPTIVE_SNAPSHOT_SECS", "1.2"))
PREEMPTIVE_MIN_AUDIO_SECS = float(os.getenv("KANCHA_PREEMPTIVE_MIN_AUDIO_SECS", "1.0"))
PREEMPTIVE_MIN_WORDS = int(os.getenv("KANCHA_PREEMPTIVE_MIN_WORDS", "4"))
PREEMPTIVE_RESNAPSHOT_SECS = float(os.getenv("KANCHA_PREEMPTIVE_RESNAPSHOT_SECS", "0.7"))
PREEMPTIVE_MAX_SNAPSHOTS = int(os.getenv("KANCHA_PREEMPTIVE_MAX_SNAPSHOTS", "1"))
SPECULATION_ENABLED = (
    os.getenv("SPECULATION", "1").strip().lower()
    not in ("0", "false", "off", "no")
)

# Low-RMS chunks tolerated during speech onset (unvoiced consonants dip below
# threshold for a frame or two); beyond this the onset attempt resets.
ONSET_TOLERANCE_CHUNKS = 3

# Text self-echo guard: last-resort filter for transcripts that near-duplicate
# what the assistant just said (the AEC path + correlation gate do the real work).
SELF_ECHO_WINDOW_SECS = float(os.getenv("KANCHA_SELF_ECHO_WINDOW_S", "6.0"))
SELF_ECHO_RATIO = float(os.getenv("KANCHA_SELF_ECHO_RATIO", "0.6"))
SELF_ECHO_MIN_WORDS = int(os.getenv("KANCHA_SELF_ECHO_MIN_WORDS", "4"))


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _text_similarity(a: str, b: str) -> float:
    wa, wb = _word_tokens(a), _word_tokens(b)
    if not wa or not wb:
        return 0.0
    return SequenceMatcher(None, wa, wb).ratio()


def _is_self_echo(text: str) -> bool:
    """True when *text* near-duplicates what the assistant recently spoke."""
    now = time.monotonic()
    candidates = [
        spoken
        for at, spoken in audio_state.recent_spoken
        if now - at <= SELF_ECHO_WINDOW_SECS
        and len(_word_tokens(spoken)) >= SELF_ECHO_MIN_WORDS
    ]
    if not candidates:
        return False
    ratio = max(_text_similarity(text, spoken) for spoken in candidates)
    if ratio >= SELF_ECHO_RATIO:
        _stt_logger.debug("Self-echo suspected (%.2f): %r", ratio, text)
        return True
    return False

# Calibration: noise-floor-relative threshold, clamped so a loud moment during
# the sample (fan, chime, the user talking) can't deafen the mic.
CALIBRATION_SECS = 0.3
CALIBRATION_MULTIPLIER = 5.5
_THRESHOLD_MAX = 0.20
_THRESHOLD_MIN = 0.01

# Sustained above-threshold speech required before the assistant's audio is
# cut (barge-in) — a brief blip never interrupts.
BARGE_IN_CONFIRM_SECS = float(os.getenv("KANCHA_BARGE_IN_CONFIRM_SECS", "0.4"))

SILENCE_HALLUCINATIONS = {
    "thank you",
    "thank you.",
    "thanks",
    "thanks.",
    "thanks for watching",
    "thanks for watching.",
    "bye",
    "bye.",
    "goodbye",
    "goodbye.",
}


_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        _client = Groq(api_key=api_key)
    return _client


def _audio_rms(audio: np.ndarray) -> float:
    """Return the RMS level for a float audio buffer."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


def _normalize_transcript(text: str) -> str:
    """Normalize transcript text for hallucination checks."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _is_probable_silence_hallucination(text: str, audio: np.ndarray) -> bool:
    """Whisper sometimes returns stock outro phrases for silence/noise;
    suppress them when the captured audio is short or low-energy."""
    normalized = _normalize_transcript(text)
    if normalized not in SILENCE_HALLUCINATIONS:
        return False
    duration = len(audio) / SAMPLE_RATE
    rms = _audio_rms(audio)
    return duration < LOW_CONFIDENCE_AUDIO_SECS or rms < LOW_CONFIDENCE_AUDIO_RMS


async def _calibrate_noise_floor(
    sample_rate: int = SAMPLE_RATE,
    calibration_secs: float = CALIBRATION_SECS,
    multiplier: float = CALIBRATION_MULTIPLIER,
    fallback: float = SILENCE_THRESH,
) -> float:
    """Sample ambient audio and derive a threshold relative to the noise floor."""
    try:
        frames = int(sample_rate * calibration_secs)
        rec = sd.rec(
            frames,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
        )
        sd.wait()
        noise_rms = _audio_rms(rec.flatten())

        raw_threshold = noise_rms * multiplier
        threshold = min(max(raw_threshold, _THRESHOLD_MIN), _THRESHOLD_MAX)

        if raw_threshold > _THRESHOLD_MAX:
            print(
                f"⚠ Calibration noise floor {noise_rms:.5f} too high; "
                f"clamping threshold {_THRESHOLD_MAX:.5f} (was {raw_threshold:.5f}). "
                "Try speaking louder or moving the mic closer."
            )

        print(f"🎚 Calibrated noise floor: {noise_rms:.5f} -> threshold {threshold:.5f}")
        return threshold
    except Exception as exc:
        print(f"⚠ Calibration failed ({exc}), using fallback threshold {fallback}")
        return fallback


_CALIBRATED_THRESHOLD: float | None = None


async def _get_silence_threshold(sample_rate: int, calibrate: bool) -> float:
    """Noise-floor-relative threshold; calibrates once per process (0.3s)."""
    global _CALIBRATED_THRESHOLD
    if not calibrate:
        return SILENCE_THRESH
    if _CALIBRATED_THRESHOLD is None:
        _CALIBRATED_THRESHOLD = await _calibrate_noise_floor(sample_rate)
    return _CALIBRATED_THRESHOLD


# Neural VAD (Silero): scores windows 0-1, so non-speech at any volume is
# rejected cleanly where RMS couldn't. The ONNX export needs the 64-sample
# context prepended to each 512-sample window, and a hysteresis pair keeps
# the decision stable across between-word probability dips.
class SileroVAD:
    ACTIVATION_THRESHOLD = float(os.getenv("KANCHA_VAD_ACTIVATION", "0.5"))
    DEACTIVATION_THRESHOLD = float(os.getenv("KANCHA_VAD_DEACTIVATION", "0.2"))
    WINDOW = 512
    CONTEXT = 64
    SMOOTH_WINDOWS = 5

    def __init__(self, model_path: str, sample_rate: int = SAMPLE_RATE):
        import onnxruntime as ort  # noqa: PLC0415

        self.sample_rate = sample_rate
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT), dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._recent = np.zeros(self.SMOOTH_WINDOWS, dtype=np.float32)
        self._speaking = False

    def feed(self, samples: np.ndarray) -> None:
        """Consume mic audio, running inference per full 512-sample window."""
        self._pending = np.concatenate(
            [self._pending, np.asarray(samples, dtype=np.float32).reshape(-1)]
        )
        while len(self._pending) >= self.WINDOW:
            win = self._pending[: self.WINDOW]
            self._pending = self._pending[self.WINDOW :]
            x = np.concatenate(
                [self._context, win.reshape(1, self.WINDOW)], axis=1
            ).astype(np.float32)
            out, state = self._session.run(
                ["output", "stateN"],
                {
                    "input": x,
                    "state": self._state,
                    "sr": np.array([self.sample_rate], dtype=np.int64),
                },
            )
            self._state = state
            self._context = x[:, -self.CONTEXT :]
            self._recent = np.roll(self._recent, -1)
            self._recent[-1] = float(out[0, 0])
            self._update()

    def _update(self) -> None:
        mean_prob = float(np.mean(self._recent))
        if self._speaking:
            if mean_prob < self.DEACTIVATION_THRESHOLD:
                self._speaking = False
        elif mean_prob > self.ACTIVATION_THRESHOLD:
            self._speaking = True

    def is_speech(self) -> bool:
        return self._speaking


_silero: SileroVAD | None = None
_silero_failed = False


def _download_silero_model(path: str) -> bool:
    """Fetch the Silero ONNX model on first use if it isn't present."""
    try:
        import urllib.request

        print(f"⬇ Downloading Silero VAD model to {path} …")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(VAD_MODEL_URL, path)
        return os.path.exists(path) and os.path.getsize(path) > 1_000_000
    except Exception as exc:  # noqa: BLE001
        print(f"⚠ Silero VAD download failed ({exc})")
        return False


def _get_silero_vad() -> SileroVAD | None:
    """Lazy-load the Silero VAD once; None on any failure so the RMS
    fallback keeps the pipeline alive."""
    global _silero, _silero_failed
    if _silero is not None or _silero_failed:
        return _silero
    path = os.getenv("KANCHA_VAD_MODEL", os.path.join(VAD_MODEL_DIR, "silero_vad.onnx"))
    if not os.path.exists(path):
        if not _download_silero_model(path):
            print(f"⚠ Silero VAD model not available at {path} — falling back to RMS VAD.")
            _silero_failed = True
            return None
    try:
        _silero = SileroVAD(path)
        print("🧠 Silero VAD active")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠ Silero VAD failed to load ({exc}) — falling back to RMS VAD.")
        _silero_failed = True
        _silero = None
    return _silero


async def listen(
    sample_rate: int = SAMPLE_RATE,
    chunk_duration: float = CHUNK_DURATION,
    silence_threshold: float | None = None,
    silence_limit: float = SILENCE_LIMIT,
    max_duration: float = MAX_DURATION,
    min_speech_secs: float = MIN_SPEECH_SECS,
    calibrate: bool = True,
    snapshot_cb: Callable[[np.ndarray], Awaitable[None]] | None = None,
    barge_in_cb: Callable[[], None] | None = None,
    manage_aec: bool = True,
) -> np.ndarray | None:
    """Listen until speech ends (Silero VAD, or RMS as fallback).

    ``snapshot_cb`` receives a mid-speech audio snapshot (for the interim
    ASR); ``barge_in_cb`` fires once the user starts speaking while the
    assistant still has audio in the air. ``manage_aec`` routes the AEC
    nodes around this listen (True) or trusts the caller's session routing,
    re-asserting it (False).
    """

    # Route mic + speaker through the AEC nodes before calibration, so the
    # noise floor is measured on the echo-cancelled signal. Session-managed
    # callers re-assert routing here — a mid-session revert to raw devices
    # (auto-switch, another app) must not silently break cancellation.
    if manage_aec:
        prev_aec = await asyncio.to_thread(_prepare_aec)
        aec_active = prev_aec is not None
    else:
        prev_aec = None
        aec_active = await asyncio.to_thread(_ensure_aec_routing)

    # Each listen captures one fresh utterance, so the VAD resets here.
    silero = _get_silero_vad()
    if silero is not None:
        silero.reset()

    try:
        if silence_threshold is None:
            if silero is None:
                silence_threshold = await _get_silence_threshold(sample_rate, calibrate)
            else:
                silence_threshold = SILENCE_THRESH

        print("\n🎤 Listening...")

        chunk_size = int(sample_rate * chunk_duration)

        chunks = []
        pending_speech_chunks = []

        started = False
        silent_chunks = 0
        speech_start_chunks = 0
        onset_gap_chunks = 0  # tolerated dips during speech onset
        total_chunks = 0
        snapshots_fired = 0
        snapshot_fire_base = 0
        barge_onset_chunks = 0  # barge-in debounce: brief blips can't cut TTS
        barge_in_fired = False

        max_chunks = int(max_duration / chunk_duration)
        silence_chunks_limit = int(silence_limit / chunk_duration)
        speech_start_chunks_required = max(1, int(SPEECH_START_SECS / chunk_duration))
        snapshot_chunks = max(1, int(PREEMPTIVE_SNAPSHOT_SECS / chunk_duration))
        resnapshot_chunks = max(1, int(PREEMPTIVE_RESNAPSHOT_SECS / chunk_duration))
        barge_in_chunks_required = max(
            1, int(BARGE_IN_CONFIRM_SECS / chunk_duration)
        )

        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()
        blocked_by_tts = False

        async def _run_snapshot(snapshot: np.ndarray) -> None:
            if snapshot_cb is None:
                return
            try:
                await snapshot_cb(snapshot)
            except Exception as exc:  # noqa: BLE001
                print(f"⚠ Interim ASR failed (non-fatal): {exc}")

        def _fire_snapshot(snapshot: np.ndarray) -> None:
            # PortAudio callback thread -> event loop via call_soon_threadsafe.
            asyncio.create_task(_run_snapshot(snapshot), name="interim_asr")

        def callback(indata, frames, time_info, status):
            nonlocal started
            nonlocal silent_chunks
            nonlocal speech_start_chunks
            nonlocal onset_gap_chunks
            nonlocal total_chunks
            nonlocal blocked_by_tts
            nonlocal snapshots_fired
            nonlocal snapshot_fire_base
            nonlocal barge_onset_chunks
            nonlocal barge_in_fired

            if status:
                print(status)

            if audio_state.thinking_active.is_set() and not audio_state.tts_active.is_set():
                # Reasoning but not yet speaking — stay inert (don't accumulate
                # or kill the stream); VAD resumes when the response plays or
                # the turn ends.
                return

            if not aec_active and audio_state.is_audio_input_blocked:
                # No echo cancellation: fall back to pause-while-speaking
                # rather than risking a self-transcription loop.
                blocked_by_tts = True
                chunks.clear()
                pending_speech_chunks.clear()
                loop.call_soon_threadsafe(stop_event.set)
                return

            if audio_state.matches_playback(indata[:, 0]):
                # The mic hears our own voice (echo leak or reverb tail) —
                # correlated with the exact played waveform. Ignore it; real
                # user speech doesn't correlate, so barge-in still works.
                return

            total_chunks += 1

            # Speech decision: Silero (neural) when loaded, else RMS.
            if silero is not None:
                silero.feed(indata[:, 0])
                is_speech = silero.is_speech()
            else:
                rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
                is_speech = rms > silence_threshold

            if is_speech:
                barge_onset_chunks += 1
                if not started:
                    # FIX #2: any above-threshold frame resets the onset gap
                    # counter and keeps accumulating pending speech, instead
                    # of requiring an unbroken run of loud chunks.
                    pending_speech_chunks.append(indata.copy())
                    speech_start_chunks += 1
                    onset_gap_chunks = 0

                    if speech_start_chunks < speech_start_chunks_required:
                        return

                    print("🔴 Recording...")
                    started = True
                    chunks.extend(pending_speech_chunks)
                    pending_speech_chunks.clear()
                    silent_chunks = 0
                    return

                chunks.append(indata.copy())
                silent_chunks = 0

                # Barge-in confirmation: only cut the assistant after this
                # much *sustained* above-threshold speech (0.24s onset +
                # BARGE_IN_CONFIRM_SECS), so a brief noise blip (door, people
                # outside, cough) can never interrupt the conversation.
                if (
                    not barge_in_fired
                    and audio_state.tts_active.is_set()
                    and barge_in_cb is not None
                    and barge_onset_chunks >= barge_in_chunks_required
                ):
                    barge_in_fired = True
                    loop.call_soon_threadsafe(barge_in_cb)

                # Mid-utterance snapshot for the interim ASR (skipped when
                # SPECULATION=0) — races a cheap transcription so the LLM can
                # start a speculative reply while the user is still talking.
                if SPECULATION_ENABLED and snapshots_fired < PREEMPTIVE_MAX_SNAPSHOTS and (
                    len(chunks) - snapshot_fire_base
                    >= (snapshot_chunks if snapshots_fired == 0 else resnapshot_chunks)
                ):
                    snapshot_fire_base = len(chunks)
                    snapshots_fired += 1
                    snapshot = np.concatenate(chunks, axis=0).flatten()
                    loop.call_soon_threadsafe(_fire_snapshot, snapshot)

            else:
                if not started:
                    # Tolerate brief dips during onset (unvoiced consonants);
                    # past the tolerance the onset attempt resets.
                    onset_gap_chunks += 1
                    if onset_gap_chunks > ONSET_TOLERANCE_CHUNKS:
                        speech_start_chunks = 0
                        barge_onset_chunks = 0
                        pending_speech_chunks.clear()
                        onset_gap_chunks = 0

            if started and not is_speech:
                chunks.append(indata.copy())
                silent_chunks += 1

                if silent_chunks >= silence_chunks_limit:
                    loop.call_soon_threadsafe(stop_event.set)
                    return

            if total_chunks >= max_chunks:
                loop.call_soon_threadsafe(stop_event.set)

        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
            callback=callback,
        ):
            await stop_event.wait()

        if blocked_by_tts:
            print("↺ Ignored audio while assistant was speaking")
            return None

        if not started or not chunks:
            return None

        audio = np.concatenate(chunks, axis=0).flatten()

        duration = len(audio) / sample_rate

        if duration < min_speech_secs:
            return None

        print(f"✓ Recorded {duration:.2f}s")

        return audio
    finally:
        if prev_aec is not None:
            await asyncio.to_thread(_restore_aec, prev_aec)


def _to_wav_bytes(audio: np.ndarray) -> io.BytesIO:
    """Convert numpy audio array to WAV bytes."""
    audio_int16 = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    buf.seek(0)
    return buf


async def transcribe(audio: np.ndarray) -> str:
    """Transcribe audio via Groq's Whisper API."""
    client = _get_client()
    wav_bytes = _to_wav_bytes(audio)

    print("🔄 Transcribing...")
    start = time.time()

    result = await asyncio.to_thread(
        client.audio.transcriptions.create,
        model="whisper-large-v3-turbo",
        file=("audio.wav", wav_bytes, "audio/wav"),
        language="en",
    )

    elapsed = time.time() - start
    print(f"✓ Transcribed in {elapsed:.2f}s")

    return result.text


async def transcribe_interim(audio: np.ndarray) -> str:
    """Cheap transcription for the preemptive path (a guess the coordinator
    discards unless it matches the authoritative transcript)."""
    client = _get_client()
    wav_bytes = _to_wav_bytes(audio)

    result = await asyncio.to_thread(
        client.audio.transcriptions.create,
        model=PREEMPTIVE_MODEL,
        file=("audio.wav", wav_bytes, "audio/wav"),
        language="en",
    )

    return result.text


async def listen_and_transcribe(
    sample_rate: int = SAMPLE_RATE,
    chunk_duration: float = CHUNK_DURATION,
    silence_threshold: float | None = None,
    silence_limit: float = SILENCE_LIMIT,
    max_duration: float = MAX_DURATION,
    min_speech_secs: float = MIN_SPEECH_SECS,
    partial_cb: Callable[[np.ndarray], Awaitable[None]] | None = None,
    barge_in_cb: Callable[[], None] | None = None,
    manage_aec: bool = True,
) -> str | None:
    """Listen, transcribe, and return the text (or None if no audio).

    ``partial_cb`` is forwarded as listen()'s snapshot_cb for the interim
    ASR; ``barge_in_cb`` cuts the assistant when the user speaks over it;
    ``manage_aec`` is forwarded to :func:`listen`.
    """
    audio = await listen(
        sample_rate=sample_rate,
        chunk_duration=chunk_duration,
        silence_threshold=silence_threshold,
        silence_limit=silence_limit,
        max_duration=max_duration,
        min_speech_secs=min_speech_secs,
        snapshot_cb=partial_cb,
        barge_in_cb=barge_in_cb,
        manage_aec=manage_aec,
    )

    if audio is None:
        print("⚠ No audio recorded")
        return None

    try:
        text = await transcribe(audio)
        if _is_probable_silence_hallucination(text, audio):
            print(f"↺ Ignored likely silence hallucination: {text!r}")
            return None
        return text
    except ValueError as e:
        print(f"❌ Transcription failed: {e}")
        return None


class MicrophoneListener:
    """Continuously listens, transcribes, and emits TranscriptReady events.

    wake_word_gated=True records one session per WakeWordDetected;
    False loops continuously for speech.
    """

    def __init__(
        self,
        bus: EventBus,
        session_id: str = "default",
        wake_word_gated: bool = False,
    ) -> None:
        self._bus = bus
        self._session_id = session_id
        self._wake_word_gated = wake_word_gated
        self._running = False
        self._wake_event = asyncio.Event()
        self._current_state = AssistantState.IDLE
        # Echo-cancel routing kept for the whole session so TTS playback stays
        # in the echo reference — that's what makes barge-in safe. Restored
        # on stop.
        self._session_aec: tuple[str, str] | None = None

    def register(self) -> None:
        """Subscribe to bus events."""
        if self._wake_word_gated:
            self._bus.subscribe(WakeWordDetected, self._on_wake_word)
        self._bus.subscribe(ShutdownRequested, self._on_shutdown)
        self._bus.subscribe(AssistantStateChanged, self._on_state_changed)

    async def _on_wake_word(self, event: WakeWordDetected) -> None:
        """Signal that we should start one listen-transcribe cycle."""
        self._wake_event.set()

    async def _on_shutdown(self, event: ShutdownRequested) -> None:
        self._running = False
        self._wake_event.set()  # unblock any waiting

    async def _on_state_changed(self, event: AssistantStateChanged) -> None:
        if event.session_id == self._session_id:
            self._current_state = event.state

    async def _maybe_preemptive(self, audio: np.ndarray) -> None:
        """Emit a speculative PartialTranscriptReady from a mid-speech snapshot."""
        if not SPECULATION_ENABLED:
            return
        if len(audio) / SAMPLE_RATE < PREEMPTIVE_MIN_AUDIO_SECS:
            return
        try:
            text = await transcribe_interim(audio)
        except Exception as exc:  # noqa: BLE001
            _stt_logger.debug("Interim ASR failed (non-fatal): %s", exc)
            return
        text = (text or "").strip()
        if not text or len(text.split()) < PREEMPTIVE_MIN_WORDS:
            return
        if _is_self_echo(text):
            return  # it heard the assistant's own voice — don't speculate on it
        self._bus.emit(
            PartialTranscriptReady(text=text, session_id=self._session_id)
        )

    def _on_barge_in(self) -> None:
        """The user spoke over the assistant — cut playback, cancel the turn."""
        self._bus.emit_threadsafe(
            UserInterrupted(session_id=self._session_id)
        )

    async def run(self) -> None:
        """Main loop — call this as an asyncio task."""
        self._running = True
        _stt_logger.info(
            "MicrophoneListener started (wake_word_gated=%s)", self._wake_word_gated
        )

        # Route mic + speaker through the AEC nodes for the whole session
        # (best-effort), so TTS stays in the echo reference while barge-in
        # is live. Restored in the finally below.
        try:
            self._session_aec = await asyncio.to_thread(_prepare_aec)
        except Exception as exc:  # noqa: BLE001
            _stt_logger.debug("Session AEC failed (non-fatal): %s", exc)
            self._session_aec = None

        try:
            while self._running:
                if self._session_aec is None:
                    # No echo cancellation — pause-while-speaking (barge-in
                    # needs AEC), which the correlation gate can't replace.
                    await audio_state.wait_until_idle()
                if self._wake_word_gated:
                    self._wake_event.clear()
                    await self._wake_event.wait()
                    if not self._running:
                        break

                self._bus.emit(
                    AssistantStateChanged(
                        state=AssistantState.LISTENING, session_id=self._session_id
                    )
                )

                try:
                    text = await listen_and_transcribe(
                        partial_cb=self._maybe_preemptive,
                        barge_in_cb=self._on_barge_in,
                        manage_aec=self._session_aec is None,
                    )
                except Exception as exc:
                    _stt_logger.exception("listen_and_transcribe error: %s", exc)
                    if self._current_state == AssistantState.LISTENING:
                        self._bus.emit(
                            AssistantStateChanged(
                                state=AssistantState.IDLE, session_id=self._session_id
                            )
                        )
                    await asyncio.sleep(0.5)
                    continue

                if not self._running:
                    break

                if text and text.strip():
                    if _is_self_echo(text):
                        # The assistant's own voice leaking onto the mic — drop
                        # it before it becomes a turn (the loop can never end).
                        _stt_logger.warning("Ignoring self-echo: %r", text)
                        if self._current_state == AssistantState.LISTENING:
                            self._bus.emit(
                                AssistantStateChanged(
                                    state=AssistantState.IDLE,
                                    session_id=self._session_id,
                                )
                            )
                        if not self._wake_word_gated:
                            await asyncio.sleep(0.1)
                        continue

                    _stt_logger.info("Transcript: %r", text)

                    self._bus.emit(
                        AssistantStateChanged(
                            state=AssistantState.THINKING, session_id=self._session_id
                        )
                    )

                    self._bus.emit(
                        TranscriptReady(
                            text=text.strip(),
                            session_id=self._session_id,
                        )
                    )

                    # Let TTS claim the speaking gate before the mic reopens;
                    # with AEC the mic stays live for barge-in (not waiting
                    # for TTS to finish).
                    await asyncio.sleep(0.05)
                    if self._session_aec is None:
                        await audio_state.wait_until_idle()
                else:
                    # No usable speech this cycle — back to IDLE (unless
                    # THINKING/SPEAKING is active).
                    if self._current_state == AssistantState.LISTENING:
                        self._bus.emit(
                            AssistantStateChanged(
                                state=AssistantState.IDLE, session_id=self._session_id
                            )
                        )
                    if not self._wake_word_gated:
                        await asyncio.sleep(0.1)
        finally:
            if self._session_aec is not None:
                try:
                    await asyncio.to_thread(_restore_aec, self._session_aec)
                except Exception as exc:  # noqa: BLE001
                    _stt_logger.debug("AEC restore failed (non-fatal): %s", exc)
                self._session_aec = None

        _stt_logger.info("MicrophoneListener stopped")

    def stop(self) -> None:
        """Request the listener to stop."""
        self._running = False
        self._wake_event.set()

    def unregister(self) -> None:
        """Remove all bus subscriptions added by register(). Call after stop()."""
        if self._wake_word_gated:
            self._bus.unsubscribe(WakeWordDetected, self._on_wake_word)
        self._bus.unsubscribe(ShutdownRequested, self._on_shutdown)
        self._bus.unsubscribe(AssistantStateChanged, self._on_state_changed)
