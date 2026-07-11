"""Structured logging setup for KANCHA.

All modules obtain their logger via ``get_logger(__name__)`` — never by
calling ``logging.getLogger`` directly.  This ensures every logger sits
under the ``kancha`` namespace, making it trivial to adjust verbosity
for the whole project from one place.

Design decisions
----------------
* Two handlers are configured: a human-readable ``ConsoleHandler`` for
  interactive use and an optional ``RotatingFileHandler`` when
  ``KANCHA_LOG_FILE`` is set.
* Log level is driven by the ``KANCHA_LOG_LEVEL`` env var (default
  ``"INFO"``).  Set it to ``"DEBUG"`` during development to see every
  bus emit and handler call.
* ``setup_logging()`` is idempotent — calling it more than once (e.g.
  from tests) is safe.
* The JSON formatter attaches ``exc_info`` and the ``context`` dict
  from ``KanchaError`` instances automatically so structured log sinks
  (e.g. a future ELK stack) receive machine-readable error details.
* Third-party loggers (``httpx``, ``chromadb``, ``asyncio``) are
  clamped to WARNING to avoid log noise.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Constants ─────────────────────────────────────────────────────────────────

_ROOT_LOGGER_NAME = "kancha"
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "chromadb",
    "asyncio",
    "urllib3",
)

# Guard so setup_logging() is idempotent.
_configured: bool = False


# ── Formatters ────────────────────────────────────────────────────────────────


class _ConsoleFormatter(logging.Formatter):
    """Human-readable coloured formatter for terminal output.

    Format::

        2024-11-01 12:34:56.789 [INFO ] kancha.bus — Event emitted: TextInputReceived
    """

    _LEVEL_COLOURS = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[35m",  # magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        colour = self._LEVEL_COLOURS.get(record.levelno, "")
        level = f"{colour}{record.levelname:<8}{self._RESET}"
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]  # trim microseconds to milliseconds
        msg = super().format(record)
        # Strip the default timestamp/level already embedded by Formatter.
        # We rebuild the whole line ourselves.
        base = f"{ts} [{level}] {record.name} — {record.getMessage()}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


class _JSONFormatter(logging.Formatter):
    """Machine-readable JSON formatter for file / structured-sink output.

    Each log record is serialised as a single-line JSON object with the
    fields:  ``ts``, ``level``, ``logger``, ``message``, ``module``,
    ``line``, and optionally ``exc`` and ``context`` (from
    ``KanchaError.context``).
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        if record.exc_info:
            payload["exc"] = traceback.format_exception(*record.exc_info)

        # Attach structured context from KanchaError if present.
        exc_val = record.exc_info[1] if record.exc_info else None
        if hasattr(exc_val, "context") and exc_val.context:
            payload["context"] = exc_val.context

        return json.dumps(payload, default=str)


# ── Public API ────────────────────────────────────────────────────────────────


def setup_logging() -> None:
    """Configure the KANCHA logging stack.

    Must be called once at application startup (``main.py``) before any
    module calls ``get_logger``.  Safe to call multiple times — only the
    first call has any effect.

    Environment variables
    ---------------------
    KANCHA_LOG_LEVEL
        Logging level for the ``kancha`` root logger.  Defaults to
        ``"INFO"``.  Accepted values: ``DEBUG``, ``INFO``, ``WARNING``,
        ``ERROR``, ``CRITICAL``.
    KANCHA_LOG_FILE
        Optional path to a log file.  When set, a rotating file handler
        (10 MB × 5 backups) is added alongside the console handler,
        using the JSON formatter.
    """
    global _configured  # noqa: PLW0603
    if _configured:
        return

    raw_level = os.environ.get("KANCHA_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, raw_level, logging.INFO)

    # Root Python logger — set to DEBUG so child loggers can go lower;
    # the kancha logger itself is the real gating level.
    logging.root.setLevel(logging.DEBUG)

    # ── kancha namespace logger ───────────────────────────────────────────
    kancha_logger = logging.getLogger(_ROOT_LOGGER_NAME)
    kancha_logger.setLevel(level)
    kancha_logger.propagate = False  # don't double-log via root

    # Console handler (human-readable).
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(_ConsoleFormatter())
    kancha_logger.addHandler(console_handler)

    # Optional file handler (JSON, rotating).
    log_file_env = os.environ.get("KANCHA_LOG_FILE", "")
    if log_file_env:
        log_path = Path(log_file_env).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # always capture everything to file
        file_handler.setFormatter(_JSONFormatter())
        kancha_logger.addHandler(file_handler)

    # ── Quieten noisy third-party loggers ────────────────────────────────
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True
    kancha_logger.debug(
        "Logging initialised (level=%s, file=%s)",
        raw_level,
        log_file_env or "none",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a ``logging.Logger`` scoped under the ``kancha`` namespace.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.  If the name
        already starts with ``"kancha"`` it is used as-is; otherwise it
        is prefixed with ``"kancha."`` so all project loggers form a
        coherent hierarchy.

    Example::

        from core.project_logging import get_logger

        logger = get_logger(__name__)
        logger.info("Bus initialised with %d handlers", count)
    """
    if not name.startswith(_ROOT_LOGGER_NAME):
        name = f"{_ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)
