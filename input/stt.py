import asyncio
import io
import os
import re
import subprocess
import sys
import time
import wave
from typing import Awaitable, Callable

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from core.audio_state import audio_state

try:
    from groq import Groq
except ImportError as _groq_err:
    raise ImportError(
        "groq package is required for voice mode. "
        "Install it with: uv add groq  (or: pip install groq)"
    ) from _groq_err

load_dotenv()

VOICE = "en-US-AriaNeural"
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
# Trailing-silence endpoint: how long of quiet audio ends a recording.
# 0.4s trims another ~0.15s off every turn versus 0.55s. It stays
# tolerant of natural mid-speech pauses because it is the *trailing* run
# of silence that ends the turn, not the first dip. Overridable via
# KANCHA_SILENCE_LIMIT.
SILENCE_LIMIT = float(os.getenv("KANCHA_SILENCE_LIMIT", "0.4"))
MAX_DURATION = 15.0
MIN_SPEECH_SECS = 0.5
MAX_HISTORY = 12
SPEECH_START_SECS = 0.24
LOW_CONFIDENCE_AUDIO_RMS = 0.025
LOW_CONFIDENCE_AUDIO_SECS = 1.2

# ── Preemptive generation (interim ASR) ─────────────────────────────────────
# The interim pass is a cheap ASR that races ahead of the final Whisper
# transcription so the LLM can start replying while the rest of the
# utterance is still being captured and transcribed. It is fired MID-speech
# on a snapshot of the audio captured so far (~1.2s in), NOT at speech end:
# the final turbo transcription is so fast (~0.25s) that a guess started
# only after recording finished would have no runway to hide the LLM
# latency. Quality-critical output always comes from the final turbo pass,
# and the speculative reply is only *reused* when the coordinator validates
# it against the authoritative transcript.
PREEMPTIVE_MODEL = os.getenv("KANCHA_PREEMPTIVE_MODEL", "whisper-large-v3-turbo")
# Fire the snapshot once this much speech has been recorded — early enough
# that the LLM guess still has the rest of the utterance + endpoint + final
# Whisper as runway, late enough that the partial usually covers the core
# of the request.
PREEMPTIVE_SNAPSHOT_SECS = float(
    os.getenv("KANCHA_PREEMPTIVE_SNAPSHOT_SECS", "1.2")
)
# Only worth racing the final pass for turns that are long enough to have
# real transcription cost — a 0.5s "thanks" is transcribed instantly
# anyway.
PREEMPTIVE_MIN_AUDIO_SECS = float(os.getenv("KANCHA_PREEMPTIVE_MIN_AUDIO_SECS", "1.0"))
# Refuse to preempt on fragments too short to be a whole thought — a
# mid-sentence guess would fail validation and waste the generation.
PREEMPTIVE_MIN_WORDS = int(os.getenv("KANCHA_PREEMPTIVE_MIN_WORDS", "4"))
# After the first snapshot, re-fire it every time this much *additional*
# speech accumulates — each re-fire lets the coordinator supersede its
# stale guess with a partial that saw more of the utterance. The re-armed
# guess replaces the earlier one (see ``PREEMPTIVE_REARM_WORDS``).
PREEMPTIVE_RESNAPSHOT_SECS = float(
    os.getenv("KANCHA_PREEMPTIVE_RESNAPSHOT_SECS", "0.7")
)
# Cap on snapshots per utterance. Beyond this the interim ASR calls are
# pure overhead — the utterance is long enough that the final Whisper pass
# dominates anyway.
PREEMPTIVE_MAX_SNAPSHOTS = int(os.getenv("KANCHA_PREEMPTIVE_MAX_SNAPSHOTS", "3"))

# FIX #2: onset tolerance — how many low-RMS chunks we allow during the
# "is this really speech starting" window before we give up and reset.
# Unvoiced consonants (t, s, p, k, f) routinely dip below threshold for
# 1-2 frames even mid-word, let alone at onset. Without this, fast speech
# gets its first syllable eaten constantly.
ONSET_TOLERANCE_CHUNKS = 3  # ~90ms of tolerated dips during onset

# FIX #3: calibration settings for noise-floor-relative threshold instead
# of a hardcoded magic number that only works for one mic/room/gain combo.
CALIBRATION_SECS = 0.3
CALIBRATION_MULTIPLIER = 5.5

# FIXME: cap the calibrated threshold so a noisy moment (the user starts
# talking *during* the 0.3s calibration sample, a fan kicks on, a
# notification chime, etc.) can't produce a threshold above normal speech
# RMS. Quiet speech in float32 normalized audio sits at 0.05–0.20; loud
# speech peaks at 0.3–0.5. Anything above 0.20 essentially deafens the
# mic against conversational volume. Previously the threshold in the
# user log spiked to 0.56655 (noise floor 0.16 × 3.5) and the assistant
# made three consecutive "No audio recorded" cycles.
_THRESHOLD_MAX = 0.20
_THRESHOLD_MIN = 0.01  # refuse to go below this — would trigger on every breath

# Barge-in requires this much sustained above-threshold speech before the
# assistant's audio is cut. Mirrors LiveKit's interruption `min_duration`:
# a brief noise blip (door, people outside, a cough) never interrupts.
# Overridable via KANCHA_BARGE_IN_CONFIRM_SECS.
BARGE_IN_CONFIRM_SECS = float(
    os.getenv("KANCHA_BARGE_IN_CONFIRM_SECS", "0.4")
)

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
    """
    Whisper sometimes returns stock outro phrases for silence/noise. Suppress
    those when the captured audio is short or low-energy.
    """
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
    """
    Sample a short burst of ambient audio and derive a silence threshold
    relative to the actual noise floor, instead of relying on a hardcoded
    constant that only makes sense for one specific mic/gain/room.

    The returned threshold is clamped to :data:`_THRESHOLD_MAX` (≈ the
    RMS of normal conversational speech) and :data:`_THRESHOLD_MIN`.
    If the calibration sample is implausibly loud — e.g. the user started
    talking during the 0.3 s sample, a notification chime fired, or a fan
    kicked on — the raw value would have made the mic deaf to real
    speech. Capping catches that case without throwing away the
    calibration signal entirely.
    """
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
    """Return a noise-floor-relative threshold, calibrating at most once
    per process.

    Calibration samples 0.3s of ambient audio, so paying it on every
    listen adds ~0.3s to *every* turn. The noise floor of a fixed
    mic/room/gain barely drifts within a session, so the first listen
    calibrates and caches; subsequent listens reuse the value.
    """
    global _CALIBRATED_THRESHOLD
    if not calibrate:
        return SILENCE_THRESH
    if _CALIBRATED_THRESHOLD is None:
        _CALIBRATED_THRESHOLD = await _calibrate_noise_floor(sample_rate)
    return _CALIBRATED_THRESHOLD


# ── Silero VAD (Phase 2) ─────────────────────────────────────────────────────
# Replaces the RMS energy gate with a neural speech detector. RMS cannot tell
# "loud noise" from "speech": a door slam, TV, or people outside easily exceed
# the calibrated threshold, while whispered speech can sit below it. Silero
# scores each window 0-1, so non-speech at any volume is rejected cleanly.
#
# The ONNX export differs from the torch model: every inference step takes a
# 576-sample input = a 64-sample context (carried across windows) prepended to
# a 512-sample window. Feeding bare 512-sample windows — as the torch model
# expects — silently returns ~0 probabilities on real speech. The context is
# what the original 2020 export embeds in its LSTM state.
#
# A hysteresis pair (0.5 to start / 0.35 to stop, LiveKit's defaults) keeps
# the decision stable, and the decision uses the mean over the last few
# windows so between-word probability dips can't flap the flag.
class SileroVAD:
    ACTIVATION_THRESHOLD = float(os.getenv("KANCHA_VAD_ACTIVATION", "0.5"))
    DEACTIVATION_THRESHOLD = float(os.getenv("KANCHA_VAD_DEACTIVATION", "0.35"))
    WINDOW = 512
    CONTEXT = 64
    # Number of inference windows averaged into the decision.
    SMOOTH_WINDOWS = 5

    def __init__(self, model_path: str, sample_rate: int = SAMPLE_RATE):
        # Lazy import so the RMS fallback still works if onnxruntime is broken.
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
        """Consume mic audio, running inference on each full 512-sample
        window (with the 64-sample context prepended). Between inferences the
        previous decision stands."""
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


_ECHO_CANCEL_SOURCE = "echo-cancel-source"
_ECHO_CANCEL_SINK = "echo-cancel-sink"


def _pactl(*args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            ["pactl", *args], capture_output=True, text=True, timeout=15
        )
    except Exception:
        return None


def _default_source_name() -> str:
    r = _pactl("get-default-source")
    return r.stdout.strip() if r and r.returncode == 0 else ""


def _default_sink_name() -> str:
    r = _pactl("get-default-sink")
    return r.stdout.strip() if r and r.returncode == 0 else ""


def _set_default_source(name: str) -> None:
    _pactl("set-default-source", name)


def _set_default_sink(name: str) -> None:
    _pactl("set-default-sink", name)


def _echo_cancel_nodes_exist() -> bool:
    sources = _pactl("list", "short", "sources")
    sinks = _pactl("list", "short", "sinks")
    return bool(
        sources
        and sinks
        and sources.returncode == 0
        and sinks.returncode == 0
        and any(_ECHO_CANCEL_SOURCE in line for line in sources.stdout.splitlines())
        and any(_ECHO_CANCEL_SINK in line for line in sinks.stdout.splitlines())
    )


def _load_echo_cancel_module() -> bool:
    """Load module-echo-cancel once; idempotent. True when both AEC nodes exist."""
    if _echo_cancel_nodes_exist():
        return True
    args = ["load-module", "module-echo-cancel", "aec_method=webrtc"]
    source = _default_source_name()
    sink = _default_sink_name()
    if source:
        args.append(f"source_master={source}")
    if sink:
        args.append(f"sink_master={sink}")
    _pactl(*args)
    return _echo_cancel_nodes_exist()


def _prepare_aec() -> tuple[str, str] | None:
    """Route default mic + speaker through the AEC nodes.

    Returns the previous (source, sink) defaults to restore afterward, or
    None if AEC is unavailable — in which case raw devices are used unchanged.
    """
    if not _load_echo_cancel_module():
        return None
    prev_source = _default_source_name()
    prev_sink = _default_sink_name()
    if prev_source != _ECHO_CANCEL_SOURCE:
        _set_default_source(_ECHO_CANCEL_SOURCE)
    if prev_sink != _ECHO_CANCEL_SINK:
        _set_default_sink(_ECHO_CANCEL_SINK)
    return prev_source, prev_sink


def _restore_aec(prev: tuple[str, str] | None) -> None:
    if not prev:
        return
    prev_source, prev_sink = prev
    if prev_source and prev_source != _ECHO_CANCEL_SOURCE:
        _set_default_source(prev_source)
    if prev_sink and prev_sink != _ECHO_CANCEL_SINK:
        _set_default_sink(prev_sink)


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
) -> np.ndarray | None:
    """
    Listen until speech ends using simple RMS-based VAD.

    If silence_threshold is None and calibrate=True, the threshold is
    derived from a short ambient-noise sample instead of using a fixed
    magic number (FIX #3).

    When ``snapshot_cb`` is given, a snapshot of the audio captured so
    far is handed to it (as a background task) once speech has been
    recording for :data:`PREEMPTIVE_SNAPSHOT_SECS` — still mid-utterance.
    The callback typically runs a fast interim ASR and emits
    ``PartialTranscriptReady`` so the coordinator can start generating a
    speculative reply while the rest of the utterance is captured and the
    final transcription runs. A slow or failing snapshot never delays the
    recording.

    When ``barge_in_cb`` is given and the AEC nodes are active, speech
    onset while the assistant is still speaking does NOT abort the listen
    — the assistant's own voice is echo-cancelled, so the recording (and
    the callback) treat the user's interjection as a fresh utterance. The
    callback fires once onset is confirmed so the caller can cut the
    assistant's playback.
    """

    # A1: route the default mic AND speaker through the AEC nodes (before
    # calibration, so the noise floor is measured on the echo-cancelled
    # signal too). Without the sink routing, the assistant's own voice is
    # not in the echo reference and would be picked up as mic input while
    # music or TTS is playing.
    prev_aec = await asyncio.to_thread(_prepare_aec)

    # Phase 2: neural VAD replaces the RMS energy gate. Resetting per listen
    # is correct — each listen() captures one fresh utterance.
    silero = _get_silero_vad()
    if silero is not None:
        silero.reset()

    try:
        if silence_threshold is None:
            if silero is None:
                silence_threshold = await _get_silence_threshold(sample_rate, calibrate)
            else:
                # Silero needs no RMS threshold; keep a sane value in case
                # the session is torn down mid-listen.
                silence_threshold = SILENCE_THRESH

        print("\n🎤 Listening...")

        chunk_size = int(sample_rate * chunk_duration)

        chunks = []
        pending_speech_chunks = []

        started = False
        silent_chunks = 0
        speech_start_chunks = 0
        onset_gap_chunks = 0  # FIX #2: tracks tolerated dips during onset
        total_chunks = 0
        snapshots_fired = 0
        snapshot_fire_base = 0
        # Barge-in debounce: counts above-threshold chunks since the onset
        # window began, so a brief noise blip can't cut the assistant off.
        barge_onset_chunks = 0
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
        # Barge-in (recording through the assistant's voice) is only safe
        # when the echo-cancel path is active — otherwise the VAD cannot
        # tell the assistant's own audio from the user's.
        aec_active = prev_aec is not None

        async def _run_snapshot(snapshot: np.ndarray) -> None:
            if snapshot_cb is None:
                return
            try:
                await snapshot_cb(snapshot)
            except Exception as exc:  # noqa: BLE001
                print(f"⚠ Interim ASR failed (non-fatal): {exc}")

        def _fire_snapshot(snapshot: np.ndarray) -> None:
            # Called from the PortAudio callback thread via
            # call_soon_threadsafe; creates the task on the loop thread.
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
                # A turn is being reasoned about but nothing is being said
                # yet — stay inert. Do NOT accumulate and do NOT kill the
                # stream: the VAD resumes the moment the response starts
                # playing (barge-in) or the turn ends.
                return

            if not aec_active and audio_state.is_audio_input_blocked:
                # No echo cancellation: we can't separate the assistant's
                # own voice from the user's, so keep the classic
                # pause-while-speaking behavior instead of risking a
                # self-transcription loop.
                blocked_by_tts = True
                chunks.clear()
                pending_speech_chunks.clear()
                loop.call_soon_threadsafe(stop_event.set)
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

                # Mid-utterance snapshots for the interim ASR. Fire the
                # first once PREEMPTIVE_SNAPSHOT_SECS of speech is in, then
                # re-fire every PREEMPTIVE_RESNAPSHOT_SECS of *additional*
                # speech (capped). Each re-fire is a fuller partial, letting
                # the coordinator supersede its stale guess mid-utterance.
                if snapshots_fired < PREEMPTIVE_MAX_SNAPSHOTS and (
                    len(chunks) - snapshot_fire_base
                    >= (snapshot_chunks if snapshots_fired == 0 else resnapshot_chunks)
                ):
                    snapshot_fire_base = len(chunks)
                    snapshots_fired += 1
                    snapshot = np.concatenate(chunks, axis=0).flatten()
                    loop.call_soon_threadsafe(_fire_snapshot, snapshot)

            else:
                if not started:
                    # FIX #2: tolerate brief dips (unvoiced consonants, etc.)
                    # during onset instead of nuking the buffer on frame one.
                    onset_gap_chunks += 1
                    if onset_gap_chunks > ONSET_TOLERANCE_CHUNKS:
                        speech_start_chunks = 0
                        barge_onset_chunks = 0
                        pending_speech_chunks.clear()
                        onset_gap_chunks = 0
                    # else: keep pending_speech_chunks and speech_start_chunks
                    # as-is, give the next chunk a chance to recover.

            if started and not is_speech:
                chunks.append(indata.copy())
                silent_chunks += 1

                if silent_chunks >= silence_chunks_limit:
                    loop.call_soon_threadsafe(stop_event.set)
                    return

            # Max recording time
            if total_chunks >= max_chunks:
                loop.call_soon_threadsafe(stop_event.set)

        # FIX #1: blocksize was computed (CHUNK_SIZE) but never passed to the
        # stream, so PortAudio picked its own buffer size and every duration-based
        # threshold in this function (onset, silence, max) was operating on a
        # false assumption about how much audio each callback represented.
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
    """
    Transcribe audio using Groq's Whisper API.

    Args:
        audio: Numpy array of audio samples (int16).

    Returns:
        Transcribed text.

    Raises:
        ValueError: If API key is not configured.
    """
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
    """
    Fast, cheap transcription for the preemptive path.

    Same shape as :func:`transcribe` but routed through the lightweight
    :data:`PREEMPTIVE_MODEL`, which returns in a fraction of the time of
    the final turbo pass. The result is only ever a *guess*: the
    coordinator discards it unless the authoritative transcript matches.
    """
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
) -> str | None:
    """
    Listen to mic input, detect silence, and transcribe.

    When ``partial_cb`` is given, a snapshot of the audio captured so far
    is handed to it mid-utterance (via :func:`listen`'s ``snapshot_cb``)
    as a background task — *while* the user is still speaking. The
    callback typically runs a fast interim ASR and emits
    ``PartialTranscriptReady`` so the coordinator can begin generating
    while the rest of the utterance is captured and the final
    transcription runs.

    ``barge_in_cb`` (see :func:`listen`) fires when the user starts
    speaking while the assistant still has audio in the air.

    Returns the transcribed text, or None if no audio was captured.
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


import logging as _stt_logging

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

_stt_logger = _stt_logging.getLogger("kancha.input.stt")


class MicrophoneListener:
    """
    Continuously listens for speech, transcribes it, and emits TranscriptReady events.

    Can operate in two modes:
    - wake_word_gated=True: starts a single recording session each time WakeWordDetected fires
    - wake_word_gated=False: loops continuously listening for speech
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
        self._aec_warmed = False
        # Held while the listener is running when echo cancellation is
        # available: the default mic AND speaker stay routed through the
        # AEC nodes for the whole session, so TTS playback is always in the
        # echo reference. That is what makes barge-in (recording while the
        # assistant speaks) safe. Restored on stop.
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

    async def _warm_aec(self) -> None:
        """Pre-load the echo-cancel module and nodes before first use.

        The AEC path is otherwise set up lazily inside the first
        :func:`listen` call, adding pactl module-load latency to the very
        turn the user is already waiting on. Routing through the nodes
        once at boot (then restoring) warms everything up so subsequent
        :func:`listen` calls only re-point the defaults — a few fast
        pactl calls instead of a module load. Best-effort: AEC stays
        optional, so a failure here is swallowed.
        """
        try:
            prev = await asyncio.to_thread(_prepare_aec)
            if prev is not None:
                await asyncio.sleep(0.05)
                await asyncio.to_thread(_restore_aec, prev)
        except Exception as exc:  # noqa: BLE001
            _stt_logger.debug("AEC warm-up failed (non-fatal): %s", exc)

    async def _maybe_preemptive(self, audio: np.ndarray) -> None:
        """Race a cheap ASR ahead of the authoritative transcription.

        Receives a *mid-speech snapshot* of the audio (fired by
        :func:`listen` once :data:`PREEMPTIVE_SNAPSHOT_SECS` of speech has
        been recorded). If the interim text looks like a whole thought it
        is emitted as :class:`PartialTranscriptReady` so the coordinator
        can start generating a speculative reply while the user is still
        talking and the final Whisper call runs. Everything here is a
        guess — the coordinator validates and may discard it.
        """
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
        self._bus.emit(
            PartialTranscriptReady(text=text, session_id=self._session_id)
        )

    def _on_barge_in(self) -> None:
        """The user started speaking while the assistant had audio in the
        air. Emit ``UserInterrupted`` so TTS cuts playback and the
        coordinator cancels the interrupted generation."""
        self._bus.emit_threadsafe(
            UserInterrupted(session_id=self._session_id)
        )

    async def run(self) -> None:
        """Main loop — call this as an asyncio task."""
        self._running = True
        _stt_logger.info(
            "MicrophoneListener started (wake_word_gated=%s)", self._wake_word_gated
        )
        if not self._aec_warmed:
            self._aec_warmed = True
            asyncio.create_task(self._warm_aec(), name="aec_warmup")

        # Route default mic + speaker through the AEC nodes for the whole
        # session (best-effort). Kept routed while TTS plays so the
        # assistant's own voice is in the echo reference — the precondition
        # for barge-in. Restored in the finally below.
        try:
            self._session_aec = await asyncio.to_thread(_prepare_aec)
        except Exception as exc:  # noqa: BLE001
            _stt_logger.debug("Session AEC failed (non-fatal): %s", exc)
            self._session_aec = None

        try:
            while self._running:
                if self._session_aec is None:
                    # No echo cancellation — fall back to the classic
                    # pause-while-speaking behavior (barge-in needs AEC).
                    await audio_state.wait_until_idle()
                if self._wake_word_gated:
                    # Wait for wake word before recording
                    self._wake_event.clear()
                    await self._wake_event.wait()
                    if not self._running:
                        break

                # UI state: the mic is open and actively recording/listening.
                self._bus.emit(
                    AssistantStateChanged(
                        state=AssistantState.LISTENING, session_id=self._session_id
                    )
                )

                try:
                    text = await listen_and_transcribe(
                        partial_cb=self._maybe_preemptive,
                        barge_in_cb=self._on_barge_in,
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
                    _stt_logger.info("Transcript: %r", text)

                    # UI state: recording finished, a transcript is ready — the
                    # rest of the pipeline (NLU -> Reasoning -> LLM) is now
                    # "thinking". TTSHandler will move this to SPEAKING/IDLE once
                    # a response is ready (see output/tts.py).
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

                    # Give the TTS handler a chance to begin speaking before
                    # opening the microphone again. (We do NOT wait for TTS to
                    # finish: with AEC the mic stays live for barge-in.)
                    await asyncio.sleep(0.05)
                    if self._session_aec is None:
                        # No AEC — stay paused until TTS fully finishes.
                        await audio_state.wait_until_idle()
                else:
                    # No usable speech captured this cycle — back to idle rather
                    # than leaving the UI stuck showing "listening".
                    # But only if the state is currently LISTENING (to avoid overriding THINKING or SPEAKING).
                    if self._current_state == AssistantState.LISTENING:
                        self._bus.emit(
                            AssistantStateChanged(
                                state=AssistantState.IDLE, session_id=self._session_id
                            )
                        )
                    if not self._wake_word_gated:
                        # Brief pause before re-listening in continuous mode
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
