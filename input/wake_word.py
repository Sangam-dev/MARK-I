"""TFLite + openwakeword wake-word detector for the JARVIS assistant.

Uses openwakeword's BUILT-IN ``hey_jarvis_v0.1.onnx`` model which is shipped
as part of the ``openwakeword`` package (no separate model files needed).

NOTE: The custom ``.tflite`` models in ``wakeword tenserflow/models/`` are not
used here — the installed openwakeword (>=0.4) only accepts ONNX format for
``wakeword_model_paths``, but ships with a high-quality ``hey_jarvis`` ONNX
model already, so we use that directly.

The public API is a single coroutine:

    detector = WakeWordDetector()
    stop = threading.Event()
    detected = await detector.listen_for_wake_word(stop)
    # True  → "hey jarvis" heard
    # False → stop.set() was called externally before detection

Runs inside ``asyncio.to_thread()`` so it never blocks the event loop. Opens
a PyAudio InputStream and feeds 1280-sample chunks to openwakeword. Runs
ONLY during the SLEEPING phase — the sounddevice STT stream (input/stt.py)
is inactive during this time, so there is no mic-device conflict.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger("kancha.input.wake_word")

_CHUNK_SIZE = 1280  # 80 ms at 16 kHz (openwakeword default)
_SAMPLE_RATE = 16000
_DEFAULT_THRESHOLD = 0.5  # score above this → wake word triggered
# override via KANCHA_WAKEWORD_THRESHOLD in .env


def _find_builtin_jarvis_model() -> str | None:
    """Return the filesystem path to openwakeword's built-in hey_jarvis ONNX model."""
    try:
        import openwakeword as _oww

        pkg_dir = Path(_oww.__file__).parent
        candidate = pkg_dir / "resources" / "models" / "hey_jarvis_v0.1.onnx"
        if candidate.exists():
            return str(candidate)
        # Fallback: search the package directory
        for f in pkg_dir.rglob("*jarvis*.onnx"):
            return str(f)
    except Exception:
        pass
    return None


def _run_detection_loop(
    stop_event: threading.Event,
    threshold: float,
    model_path: str | None,
) -> bool:
    """Blocking detection loop — run inside asyncio.to_thread().

    Returns True when "hey jarvis" is detected, False when stop_event fires.
    """
    try:
        import numpy as np
        import pyaudio
        from openwakeword.model import Model
    except ImportError as exc:
        logger.error(
            "Wake word detection unavailable — missing dependency: %s. "
            "Install with: uv add openwakeword pyaudio",
            exc,
        )
        return False

    # Load ONLY the hey_jarvis model (faster startup, no false positives
    # from unrelated wake words like "alexa", "hey mycroft" etc.)
    try:
        if model_path:
            oww = Model(wakeword_model_paths=[model_path])
            logger.info("Loaded custom hey_jarvis model: %s", model_path)
        else:
            # No path found — load ALL built-in models and filter to jarvis
            oww = Model()
    except Exception as exc:
        logger.error("Failed to load openwakeword model: %s", exc)
        return False

    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=_SAMPLE_RATE,
            input=True,
            frames_per_buffer=_CHUNK_SIZE,
        )
    except Exception as exc:
        logger.error("Could not open microphone for wake word detection: %s", exc)
        pa.terminate()
        return False

    logger.info("Wake word detector active — listening for 'hey jarvis'")

    try:
        while not stop_event.is_set():
            try:
                raw = stream.read(_CHUNK_SIZE, exception_on_overflow=False)
            except Exception:
                break

            audio = np.frombuffer(raw, dtype=np.int16)
            oww.predict(audio)

            for model_name, scores in oww.prediction_buffer.items():
                # Accept any model whose name contains "jarvis"
                if "jarvis" in model_name.lower() and scores[-1] > threshold:
                    logger.info(
                        "Wake word detected: '%s' (score=%.3f)",
                        model_name,
                        scores[-1],
                    )
                    try:
                        oww.reset()
                    except Exception:
                        pass
                    return True
    except Exception as exc:
        logger.exception("Wake word detection loop error: %s", exc)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        logger.debug("Wake word detector microphone closed")

    return False  # stop_event was set


class WakeWordDetector:
    """Async-friendly wrapper around the blocking openwakeword detection loop."""

    def __init__(self, threshold: float | None = None) -> None:
        import os

        self._threshold = (
            threshold
            if threshold is not None
            else float(os.getenv("KANCHA_WAKEWORD_THRESHOLD", str(_DEFAULT_THRESHOLD)))
        )
        self._model_path = _find_builtin_jarvis_model()
        if self._model_path:
            logger.info("Will use hey_jarvis model: %s", self._model_path)
        else:
            logger.warning(
                "hey_jarvis model not found in openwakeword package; "
                "will load all built-in models and filter by name."
            )

    async def listen_for_wake_word(self, stop_event: threading.Event) -> bool:
        """Await detection in a worker thread.

        Returns True when "hey jarvis" is heard, False if ``stop_event``
        is set before detection (e.g. server shutting down).
        """
        import asyncio

        return await asyncio.to_thread(
            _run_detection_loop,
            stop_event,
            self._threshold,
            self._model_path,
        )
