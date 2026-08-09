#!/usr/bin/env python3
"""
JARVIS — Event-Driven Voice AI Assistant
=========================================

Entry point for the complete assistant pipeline.

Usage::

    python main.py              # continuous voice mode (default)
    python main.py --text       # keyboard text input mode
    python main.py --voice      # explicit continuous voice mode
    python main.py --log-level DEBUG
    python main.py --no-tts     # disable TTS (text responses only)
    python main.py --session my_session  # custom session ID

Environment variables (see .env):
    GEMINI_API_KEY          Gemini API key (or GEMINI_API_KEY_1…_9 for rotation)
    GROQ_API_KEY            Groq API key for Whisper STT (voice mode only)
    JARVIS_LOG_LEVEL        Log level env var (default: INFO)
    JARVIS_LOG_FILE         Optional path to a log file

Full event pipeline::

    [Voice]  MicrophoneListener → TranscriptReady  ┐
    [Text]   TextInputHandler   → TextInputReceived ┘
                                        ↓
                    NLUClassifier → IntentIdentified
                                        ↓
                     Intent → ReasoningRequested
                                        ↓
                      ReasoningCoordinator
                        ├── Task  → TaskExecutionRequested → TaskExecutor
                        │              → TaskCompleted → LLM → ResponseReady
                        └── Chat  → LLM → ResponseReady
                                        ↓
                       TTSHandler (speaks) + ResponseFormatter (prints)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before anything else so API keys are available at import time.
load_dotenv()

# ── Core ──────────────────────────────────────────────────────────────────────
from core.events import ShutdownRequested
from core.pipeline import build_pipeline, shutdown_pipeline
from core.project_logging import setup_logging

# ── Input: text (always available) ────────────────────────────────────────────
from input.text_input import TextInputHandler

# ── Input: STT/voice (requires groq package) ──────────────────────────────────
# Imported lazily inside _run() when --voice mode is selected so that
# missing `groq` never prevents --text mode from starting.
#
# NOTE: all other module wiring (EventBus, Memory, LLM, NLU, Reasoning,
# TaskExecutor, output handlers) now lives in core/pipeline.py, shared with
# api/server.py. See answers/guide.md for how the two entry points relate.

# ── Constants ─────────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent / "memory" / "data"
_DEFAULT_SESSION = "default"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JARVIS — Event-Driven Voice AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --text         # keyboard mode\n"
            "  python main.py --voice        # microphone mode\n"
            "  python main.py --text --no-tts --log-level DEBUG\n"
        ),
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--text",
        action="store_true",
        help="Keyboard text input mode (no microphone or Groq API key required)",
    )
    mode_group.add_argument(
        "--voice",
        action="store_true",
        help="Continuous microphone input mode (requires GROQ_API_KEY)",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("JARVIS_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar="LEVEL",
        help="Logging verbosity: DEBUG | INFO | WARNING | ERROR (default: INFO)",
    )
    parser.add_argument(
        "--session",
        default=_DEFAULT_SESSION,
        metavar="ID",
        help="Session identifier for memory isolation (default: default)",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable text-to-speech; assistant responses are printed only",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=int(os.getenv("MAX_PROMPT_TOKENS", "4000")),
        metavar="N",
        help="Max tokens passed in LLM context window (default: 4000)",
    )
    parser.add_argument(
        "--script",
        metavar="FILE",
        help="Execute a scripted demonstration from a JSON file",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Async main
# ─────────────────────────────────────────────────────────────────────────────


async def _run(args: argparse.Namespace) -> None:
    """Initialise every component and run until shutdown."""
    logger = logging.getLogger("JARVIS.main")
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("JARVIS starting  (session=%s)", args.session)
    logger.info("=" * 60)

    # ── 1–10. Shared pipeline (EventBus, Memory, LLM, NLU, Reasoning, Tasks, Output) ──
    # All of this used to be inline here. It now lives in core/pipeline.py so
    # api/server.py can build the exact same pipeline for the WebSocket/REST
    # front door without duplicating the wiring. See answers/guide.md.
    pipeline = await build_pipeline(
        session_id=args.session,
        data_dir=_DATA_DIR,
        enable_tts=not args.no_tts,
        enable_console_formatter=True,
    )
    bus = pipeline.bus
    logger.debug("Pipeline built: %r", pipeline)

    # Shutdown coordination ── any module emits ShutdownRequested to exit.
    _shutdown_event = asyncio.Event()

    async def _handle_shutdown(event: ShutdownRequested) -> None:
        logger.info("Shutdown requested: %s ── stopping…", event.reason)
        _shutdown_event.set()

    bus.subscribe(ShutdownRequested, _handle_shutdown)

    key_count = len(pipeline.llm.pool._entries) if pipeline.llm.pool else 0
    logger.info("GeminiClient initialised  (%d API key(s) in pool)", key_count)
    logger.info("Memory initialised  (structured facts=SQLite, vector/RAG=disabled)")
    logger.info("NLUClassifier registered  (regex fast-path + LLM fallback)")
    logger.info("Reasoning bridge registered  (facts only, RAG disabled)")
    logger.info("ReasoningCoordinator registered")
    logger.info(
        "TaskExecutor registered  "
        "(tasks: open_app, set_alarm, list_alarms, cancel_alarms, "
        "get_weather, sleep, shutdown, restart, file_operation)"
    )
    if pipeline.tts is not None:
        logger.info("TTSHandler registered  (edge-tts + sounddevice, mutex-serialised)")
    else:
        logger.info("TTS disabled  (--no-tts flag set)")

    # ── 11. Input mode ────────────────────────────────────────────────────────
    input_tasks: list[asyncio.Task] = []

    if args.text:
        # Text mode: stdin → TextInputReceived
        logger.info("Input mode: TEXT  (type + Enter; 'exit' or 'quit' to stop)")
        print("\n" + "=" * 60)
        print("  JARVIS — Text Mode")
        print("  Type your message and press Enter.")
        print("  Type  exit  or  quit  to shut down.")
        print("=" * 60 + "\n")

        text_handler = TextInputHandler(
            bus=bus, session_id=args.session, script_file=args.script
        )
        input_tasks.append(asyncio.create_task(text_handler.run(), name="text_input"))

    else:
        # Voice mode: microphone → VAD → Groq Whisper → TranscriptReady
        logger.info("Input mode: VOICE  (speak → pause → transcribe → respond)")

        try:
            from input.stt import MicrophoneListener
        except ImportError as exc:
            logger.critical(
                "Voice mode requires the groq package: %s\n"
                "Install with: uv add groq  (or: pip install groq)\n"
                "Or run in text mode: python main.py --text",
                exc,
            )
            sys.exit(1)

        print("\n" + "=" * 60)
        print("  JARVIS — Voice Mode")
        print("  Speak clearly after the 🎤 prompt appears.")
        print("  Pause to end your utterance.  Ctrl+C to quit.")
        print("=" * 60 + "\n")

        mic = MicrophoneListener(
            bus=bus,
            session_id=args.session,
            wake_word_gated=False,  # continuous; no wake-word required
        )
        mic.register()
        input_tasks.append(asyncio.create_task(mic.run(), name="microphone"))

    # ── 12. Run until a shutdown signal ───────────────────────────────────────
    shutdown_sentinel = asyncio.create_task(
        _shutdown_event.wait(), name="shutdown_sentinel"
    )

    try:
        done, pending = await asyncio.wait(
            [shutdown_sentinel, *input_tasks],
            return_when=asyncio.FIRST_COMPLETED,
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("KeyboardInterrupt — initiating shutdown…")

    # ── 13. Graceful cleanup ───────────────────────────────────────────────────
    logger.info("Shutting down — cleaning up…")

    for task in input_tasks:
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass

    # Drain in-flight event handlers, close the bus, and flush memory backends.
    await shutdown_pipeline(pipeline)

    logger.info("JARVIS stopped. Goodbye.")
    print("\nJARVIS: Goodbye.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Sync entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Parse CLI args, configure logging, then run the async pipeline."""
    args = _parse_args()

    # Let setup_logging() pick up the CLI-chosen level via the env var it reads.
    os.environ["JARVIS_LOG_LEVEL"] = args.log_level
    setup_logging()

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass  # already handled inside _run
    except Exception as exc:
        logging.getLogger("JARVIS.main").critical(
            "Fatal unhandled error: %s", exc, exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
