import asyncio
import io
import os
import re
import sys
import time
import wave
from core.audio_state import audio_state
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

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
SILENCE_LIMIT = 1.5
MAX_DURATION = 15.0
MIN_SPEECH_SECS = 0.5
MAX_HISTORY = 12
SPEECH_START_SECS = 0.24
LOW_CONFIDENCE_AUDIO_RMS = 0.025
LOW_CONFIDENCE_AUDIO_SECS = 1.2

# FIX #2: onset tolerance — how many low-RMS chunks we allow during the
# "is this really speech starting" window before we give up and reset.
# Unvoiced consonants (t, s, p, k, f) routinely dip below threshold for
# 1-2 frames even mid-word, let alone at onset. Without this, fast speech
# gets its first syllable eaten constantly.
ONSET_TOLERANCE_CHUNKS = 3  # ~90ms of tolerated dips during onset

# FIX #3: calibration settings for noise-floor-relative threshold instead
# of a hardcoded magic number that only works for one mic/room/gain combo.
CALIBRATION_SECS = 0.3
CALIBRATION_MULTIPLIER = 3.5

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

        threshold = max(noise_rms * multiplier, fallback * 0.5)
        print(f"🎚 Calibrated noise floor: {noise_rms:.5f} -> threshold {threshold:.5f}")
        return threshold
    except Exception as exc:
        print(f"⚠ Calibration failed ({exc}), using fallback threshold {fallback}")
        return fallback


async def listen(
    sample_rate: int = SAMPLE_RATE,
    chunk_duration: float = CHUNK_DURATION,
    silence_threshold: float | None = None,
    silence_limit: float = SILENCE_LIMIT,
    max_duration: float = MAX_DURATION,
    min_speech_secs: float = MIN_SPEECH_SECS,
    calibrate: bool = True,
) -> np.ndarray | None:
    """
    Listen until speech ends using simple RMS-based VAD.

    If silence_threshold is None and calibrate=True, the threshold is
    derived from a short ambient-noise sample instead of using a fixed
    magic number (FIX #3).
    """

    if silence_threshold is None:
        silence_threshold = (
            await _calibrate_noise_floor(sample_rate)
            if calibrate
            else SILENCE_THRESH
        )

    print("\n🎤 Listening...")

    chunk_size = int(sample_rate * chunk_duration)

    chunks = []
    pending_speech_chunks = []

    started = False
    silent_chunks = 0
    speech_start_chunks = 0
    onset_gap_chunks = 0  # FIX #2: tracks tolerated dips during onset
    total_chunks = 0

    max_chunks = int(max_duration / chunk_duration)
    silence_chunks_limit = int(silence_limit / chunk_duration)
    speech_start_chunks_required = max(1, int(SPEECH_START_SECS / chunk_duration))

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    blocked_by_tts = False

    def callback(indata, frames, time_info, status):
        nonlocal started
        nonlocal silent_chunks
        nonlocal speech_start_chunks
        nonlocal onset_gap_chunks
        nonlocal total_chunks
        nonlocal blocked_by_tts

        if status:
            print(status)

        if audio_state.is_audio_input_blocked:
            blocked_by_tts = True
            chunks.clear()
            pending_speech_chunks.clear()
            loop.call_soon_threadsafe(stop_event.set)
            return

        total_chunks += 1

        # RMS VAD
        rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))

        if rms > silence_threshold:
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

        else:
            if not started:
                # FIX #2: tolerate brief dips (unvoiced consonants, etc.)
                # during onset instead of nuking the buffer on frame one.
                onset_gap_chunks += 1
                if onset_gap_chunks > ONSET_TOLERANCE_CHUNKS:
                    speech_start_chunks = 0
                    pending_speech_chunks.clear()
                    onset_gap_chunks = 0
                # else: keep pending_speech_chunks and speech_start_chunks
                # as-is, give the next chunk a chance to recover.

        if started and rms <= silence_threshold:
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

    if blocked_by_tts or audio_state.is_audio_input_blocked:
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
        model="whisper-large-v3",
        file=("audio.wav", wav_bytes, "audio/wav"),
        language="en",
    )

    elapsed = time.time() - start
    print(f"✓ Transcribed in {elapsed:.2f}s")

    return result.text


async def listen_and_transcribe(
    sample_rate: int = SAMPLE_RATE,
    chunk_duration: float = CHUNK_DURATION,
    silence_threshold: float | None = None,
    silence_limit: float = SILENCE_LIMIT,
    max_duration: float = MAX_DURATION,
    min_speech_secs: float = MIN_SPEECH_SECS,
) -> str | None:
    """
    Listen to mic input, detect silence, and transcribe.

    Returns the transcribed text, or None if no audio was captured.
    """
    await audio_state.wait_until_idle()

    audio = await listen(
        sample_rate=sample_rate,
        chunk_duration=chunk_duration,
        silence_threshold=silence_threshold,
        silence_limit=silence_limit,
        max_duration=max_duration,
        min_speech_secs=min_speech_secs,
    )

    if audio is None:
        print("⚠ No audio recorded")
        return None

    if audio_state.is_audio_input_blocked:
        print("↺ Skipping transcription during assistant audio")
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
from core.events import ShutdownRequested, TranscriptReady, WakeWordDetected

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

    def register(self) -> None:
        """Subscribe to bus events."""
        if self._wake_word_gated:
            self._bus.subscribe(WakeWordDetected, self._on_wake_word)
        self._bus.subscribe(ShutdownRequested, self._on_shutdown)

    async def _on_wake_word(self, event: WakeWordDetected) -> None:
        """Signal that we should start one listen-transcribe cycle."""
        self._wake_event.set()

    async def _on_shutdown(self, event: ShutdownRequested) -> None:
        self._running = False
        self._wake_event.set()  # unblock any waiting

    async def run(self) -> None:
        """Main loop — call this as an asyncio task."""
        self._running = True
        _stt_logger.info(
            "MicrophoneListener started (wake_word_gated=%s)", self._wake_word_gated
        )

        while self._running:
            await audio_state.wait_until_idle()
            if self._wake_word_gated:
                # Wait for wake word before recording
                self._wake_event.clear()
                await self._wake_event.wait()
                if not self._running:
                    break

            try:
                text = await listen_and_transcribe()
            except Exception as exc:
                _stt_logger.exception("listen_and_transcribe error: %s", exc)
                await asyncio.sleep(0.5)
                continue

            if not self._running:
                break

            if text and text.strip():
                _stt_logger.info("Transcript: %r", text)

                self._bus.emit(
                    TranscriptReady(
                        text=text.strip(),
                        session_id=self._session_id,
                    )
                )

                #
                # Give the TTS handler a chance to begin speaking before
                # opening the microphone again.
                #
                await asyncio.sleep(0.05)

                #
                # Stay paused until TTS playback has completely finished.
                #
                await audio_state.wait_until_idle()
            else:
                if not self._wake_word_gated:
                    # Brief pause before re-listening in continuous mode
                    await asyncio.sleep(0.1)

        _stt_logger.info("MicrophoneListener stopped")

    def stop(self) -> None:
        """Request the listener to stop."""
        self._running = False
        self._wake_event.set()