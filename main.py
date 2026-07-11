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
from core.bus import EventBus
from core.events import (
    IntentIdentified,
    MemoryRetrieved,
    ReasoningRequested,
    ShutdownRequested,
    SystemError,
)
from core.project_logging import setup_logging

# ── Input: text (always available) ────────────────────────────────────────────
from input.text_input import TextInputHandler

# ── Memory ────────────────────────────────────────────────────────────────────
from memory.manager import MemoryManager

# ── NLU ───────────────────────────────────────────────────────────────────────
from nlu.classifier import NLUClassifier
from output.response_formatter import ResponseFormatter

# ── Output ────────────────────────────────────────────────────────────────────
from output.tts import TTSHandler
from reasoning.coordinator import ReasoningCoordinator

# ── Reasoning ─────────────────────────────────────────────────────────────────
from reasoning.llm_client import GeminiClient
# RAG is intentionally disabled for now.
# from reasoning.rag import RAGPipeline

# ── Tasks ─────────────────────────────────────────────────────────────────────
from tasks.executor import TaskExecutor

# ── Input: STT/voice (requires groq package) ──────────────────────────────────
# Imported lazily inside _run() when --voice mode is selected so that
# missing `groq` never prevents --text mode from starting.

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

    # ── 1. EventBus ───────────────────────────────────────────────────────────
    bus = EventBus()
    logger.debug("EventBus created: %r", bus)

    # Shutdown coordination — any module emits ShutdownRequested to exit.
    _shutdown_event = asyncio.Event()

    async def _handle_shutdown(event: ShutdownRequested) -> None:
        logger.info("Shutdown requested: %s — stopping…", event.reason)
        _shutdown_event.set()

    bus.subscribe(ShutdownRequested, _handle_shutdown)

    # Central system-error logger so nothing is silently swallowed.
    async def _handle_system_error(event: SystemError) -> None:
        lvl = logging.WARNING if event.recoverable else logging.ERROR
        logger.log(
            lvl, "SystemError [%s]: %s", event.source_module, event.error_message
        )

    bus.subscribe(SystemError, _handle_system_error)

    # ── 2. Memory ─────────────────────────────────────────────────────────────
    memory = MemoryManager(bus=bus, data_dir=_DATA_DIR, session_id=args.session)
    try:
        await memory.initialize()
        logger.info("Memory initialised  (structured facts=SQLite, vector/RAG=disabled)")
    except Exception as exc:
        logger.warning(
            "Memory initialisation failure: %s — continuing without persistent memory.",
            exc,
        )

    # Wire MemoryManager's @subscribe-decorated handlers into the bus.
    bus.register_handlers(memory)
    logger.debug("MemoryManager event handlers registered via @subscribe decorators")

    # ── 3. LLM Client ─────────────────────────────────────────────────────────
    llm = GeminiClient()
    await llm.initialize()
    key_count = len(llm.pool._entries) if llm.pool else 0
    logger.info("GeminiClient initialised  (%d API key(s) in pool)", key_count)

    # ── 5. NLU Classifier ─────────────────────────────────────────────────────
    # Subscribes to: TextInputReceived, TranscriptReady
    # Emits:         IntentIdentified
    nlu = NLUClassifier(llm_client=llm, bus=bus)
    nlu.register()
    logger.info("NLUClassifier registered  (regex fast-path + LLM fallback)")

    # ── 6. Reasoning request bridge ──────────────────────────────────────────
    # RAG is intentionally disabled. This bridge forwards intents to reasoning
    # with only durable user facts from SQLite, and no vector/episodic context.
    async def _handle_intent_identified(event: IntentIdentified) -> None:
        facts = await memory.get_all_facts()
        memory_event = MemoryRetrieved(
            session_id=event.session_id,
            query=event.raw_input,
            structured_context=facts,
            episodic_context=[],
        )
        bus.emit(
            ReasoningRequested(
                session_id=event.session_id,
                intent_event=event,
                memory_events=[memory_event],
            )
        )

    bus.subscribe(IntentIdentified, _handle_intent_identified)
    logger.info("Reasoning bridge registered  (facts only, RAG disabled)")

    # ── 7. Reasoning Coordinator ──────────────────────────────────────────────
    # Subscribes to: ReasoningRequested, TaskCompleted
    # Emits:         TaskExecutionRequested  (for task intents)
    #                ResponseReady            (for conversational intents + post-task)
    #                MemoryUpdateNeeded       (to persist turns)
    coordinator = ReasoningCoordinator(
        bus=bus,
        gemini_client=llm,
        memory_manager=memory,
    )
    coordinator.register()
    logger.info("ReasoningCoordinator registered")

    # ── 8. Task Executor ──────────────────────────────────────────────────────
    # Subscribes to: TaskExecutionRequested
    # Emits:         TaskCompleted
    task_executor = TaskExecutor(bus=bus)
    task_executor.register()
    logger.info(
        "TaskExecutor registered  "
        "(tasks: open_app, set_alarm, list_alarms, cancel_alarms, "
        "get_weather, sleep, shutdown, restart, file_operation)"
    )

    # ── 9. Output: console formatter (always active) ──────────────────────────
    # Subscribes to: ResponseReady
    formatter = ResponseFormatter(bus=bus)
    formatter.register()
    logger.debug("ResponseFormatter registered")

    # ── 10. Output: TTS (optional) ────────────────────────────────────────────
    # Subscribes to: ResponseReady
    if not args.no_tts:
        tts_handler = TTSHandler(bus=bus)
        tts_handler.register()
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

        text_handler = TextInputHandler(bus=bus, session_id=args.session)
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

    # Drain any still-running event handlers (give them up to 3 s).
    try:
        await asyncio.wait_for(bus.drain(), timeout=3.0)
    except asyncio.TimeoutError:
        logger.warning("Bus drain timed out — some handlers may not have completed")

    # Close the bus (cancels stray tasks, clears handler registry).
    await bus.close()

    # Close memory backends (flush SQLite WAL).
    try:
        await memory.close()
        logger.debug("Memory backends closed")
    except Exception as exc:
        logger.warning("Memory close error (non-fatal): %s", exc)

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
