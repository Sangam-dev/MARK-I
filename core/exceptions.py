"""Custom exception hierarchy for KANCHA.

Every public module raises a subclass of ``KanchaError`` so callers can
distinguish KANCHA-domain failures from unexpected Python errors.  The
optional ``context`` dict lets handlers attach structured metadata
(e.g. the offending config key, the LLM HTTP status code) without
relying on string-parsing of the message.

Design decisions
----------------
* Thin hierarchy — each exception maps 1-to-1 with a subsystem, making
  ``except`` clauses readable and precise.
* ``MemoryError`` intentionally shadows the Python builtin of the same
  name within this package.  All KANCHA code imports from here rather
  than relying on the builtin, so there is no ambiguity in practice.
* ``context`` is stored as an instance attribute so logging handlers
  can forward it to structured log sinks without extra parsing.
"""

from __future__ import annotations

from typing import Any

# ── Base ─────────────────────────────────────────────────────────────────────


class KanchaError(Exception):
    """Base class for all KANCHA-domain errors.

    Parameters
    ----------
    message:
        Human-readable description of what went wrong.
    context:
        Optional mapping of structured key/value pairs providing extra
        diagnostic information (e.g. ``{"config_key": "GEMINI_API_KEY"}``).
    """

    def __init__(
        self,
        message: str = "",
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = context or {}

    def __repr__(self) -> str:
        ctx = f", context={self.context!r}" if self.context else ""
        return f"{type(self).__name__}({str(self)!r}{ctx})"


# ── Configuration ─────────────────────────────────────────────────────────────


class ConfigurationError(KanchaError):
    """Raised when required configuration is absent or invalid.

    Typical causes: missing environment variable, unreadable config
    file, or a value that fails validation (e.g. a non-numeric port).

    Example::

        raise ConfigurationError(
            "GEMINI_API_KEY is not set",
            context={"env_var": "GEMINI_API_KEY"},
        )
    """


# ── Memory ────────────────────────────────────────────────────────────────────


class MemoryError(KanchaError):
    """Raised when a memory-layer operation fails.

    Covers SQLite (structured), ChromaDB (vector), and the in-process
    short-term buffer.  Use ``context`` to record the affected layer
    and operation so the MemoryManager can decide whether to retry or
    degrade gracefully.

    Note: this intentionally shadows the Python builtin ``MemoryError``.
    KANCHA modules should import this class explicitly from
    ``core.exceptions`` rather than relying on the builtin.

    Example::

        raise MemoryError(
            "ChromaDB collection unreachable",
            context={"layer": "vector", "collection": "kancha_main"},
        )
    """


# ── LLM ──────────────────────────────────────────────────────────────────────


class LLMError(KanchaError):
    """Raised when a call to the Gemini API fails.

    Use the more specific subclasses where possible so callers can
    apply appropriate back-off or fallback strategies.

    Example::

        raise LLMError(
            "Gemini returned status 500",
            context={"http_status": 500, "model": "gemini-1.5-flash"},
        )
    """


class LLMRateLimitError(LLMError):
    """Raised when the Gemini API returns a rate-limit (429) response.

    The caller should implement exponential back-off before retrying.

    Example::

        raise LLMRateLimitError(
            "Gemini rate limit exceeded",
            context={"retry_after_seconds": 30},
        )
    """


class LLMTimeoutError(LLMError):
    """Raised when a Gemini API call exceeds the configured timeout.

    Distinct from ``LLMRateLimitError`` — the request may be retried
    immediately (with a shorter prompt or lower token budget) rather
    than after a back-off delay.

    Example::

        raise LLMTimeoutError(
            "Gemini call timed out after 10 s",
            context={"timeout_seconds": 10},
        )
    """


# ── NLU ──────────────────────────────────────────────────────────────────────


class NLUError(KanchaError):
    """Raised when intent classification or entity extraction fails.

    This covers both structural failures (the LLM returned
    unparseable JSON) and semantic failures (confidence below
    threshold).

    Example::

        raise NLUError(
            "Intent classifier returned invalid JSON",
            context={"raw_response": response_text},
        )
    """


# ── Speech ───────────────────────────────────────────────────────────────────


class STTError(KanchaError):
    """Raised when faster-whisper transcription fails.

    Covers model-load failures, audio-format mismatches, and
    runtime errors during transcription.

    Example::

        raise STTError(
            "faster-whisper failed to load model",
            context={"model_size": "tiny", "audio_path": str(path)},
        )
    """


class TTSError(KanchaError):
    """Raised when Piper TTS synthesis fails.

    Covers voice-model load failures, synthesis errors, and audio
    write failures.

    Example::

        raise TTSError(
            "Piper failed to synthesise audio",
            context={"voice_model": "en_US-amy-medium"},
        )
    """


# ── Tasks ─────────────────────────────────────────────────────────────────────


class TaskError(KanchaError):
    """Raised when a task execution fails at runtime.

    Use ``TaskNotAllowedError`` for registry/allowlist failures;
    reserve this class for unexpected runtime errors that occur while
    the task is actually executing.

    Example::

        raise TaskError(
            "Reminder scheduler failed to persist task",
            context={"task_name": "set_reminder", "parameters": params},
        )
    """


class TaskNotAllowedError(TaskError):
    """Raised when a requested task type is not in the task registry.

    This is an allowlist enforcement error, not a runtime failure.
    The executor should emit a ``SystemError`` event and return a
    safe response to the user rather than crashing.

    Example::

        raise TaskNotAllowedError(
            "Task type 'shell_exec' is not registered",
            context={"requested_task": "shell_exec"},
        )
    """
