"""Reusable KANCHA pipeline construction.

Both the CLI entry point (``main.py``) and the FastAPI/WebSocket server
(``api/server.py``) need the exact same EventBus wiring: EventBus -> Memory ->
LLM -> NLU -> Reasoning bridge -> ReasoningCoordinator -> TaskExecutor ->
output handlers. Before this module existed, that ~150 lines of wiring lived
inline inside ``main.py:_run()``, which meant the API server would have had
to duplicate it (and the two copies would inevitably drift).

``build_pipeline()`` is now the single source of truth. Anything that needs a
running KANCHA pipeline (CLI, API server, tests, a future gRPC front door,
whatever) calls this instead of re-wiring the bus by hand.

See answers/guide.md for how to extend this when adding new modules/actions.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from core.bus import EventBus
from core.events import (
    IntentIdentified,
    MemoryRetrieved,
    ReasoningRequested,
    SystemError,
)
from memory.manager import MemoryManager
from nlu.classifier import NLUClassifier
from output.response_formatter import ResponseFormatter
from output.tts import TTSHandler
from reasoning.coordinator import ReasoningCoordinator
from reasoning.llm_client import GeminiClient
from tasks.executor import TaskExecutor

logger = logging.getLogger("kancha.pipeline")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "memory" / "data"


@dataclass
class Pipeline:
    """Handle bundle for a fully-wired KANCHA event pipeline.

    Consumers (CLI, API server) hold on to this to reach the bus, add their
    own input-mode subscriptions (stdin, microphone, WebSocket), and to shut
    everything down cleanly via ``shutdown_pipeline()``.
    """

    bus: EventBus
    memory: MemoryManager
    llm: GeminiClient
    nlu: NLUClassifier
    coordinator: ReasoningCoordinator
    task_executor: TaskExecutor
    formatter: ResponseFormatter
    tts: TTSHandler | None
    session_id: str
    tts_enabled: bool


async def build_pipeline(
    session_id: str = "default",
    data_dir: Path | None = None,
    enable_tts: bool = True,
    enable_console_formatter: bool = True,
) -> Pipeline:
    """Construct and wire every KANCHA module onto a fresh EventBus.

    Parameters
    ----------
    session_id:
        Memory/session isolation key. One pipeline == one session today
        (see answers/integration_plan.md, "single session assumption").
    data_dir:
        Directory for SQLite/structured memory. Defaults to
        ``kancha/memory/data``.
    enable_tts:
        Register :class:`~output.tts.TTSHandler` (speaks responses aloud on
        this machine's speakers via edge-tts + sounddevice).
    enable_console_formatter:
        Register :class:`~output.response_formatter.ResponseFormatter`
        (prints responses to stdout). Handy to leave on even for the API
        server since it doubles as free server-side logging of replies.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    bus = EventBus()

    # Central system-error logger so nothing is silently swallowed.
    async def _handle_system_error(event: SystemError) -> None:
        lvl = logging.WARNING if event.recoverable else logging.ERROR
        logger.log(
            lvl, "SystemError [%s]: %s", event.source_module, event.error_message
        )

    bus.subscribe(SystemError, _handle_system_error)

    # ── Memory ────────────────────────────────────────────────────────────
    memory = MemoryManager(bus=bus, data_dir=data_dir, session_id=session_id)
    try:
        await memory.initialize()
        logger.info("Memory initialised (session=%s)", session_id)
    except Exception as exc:
        logger.warning(
            "Memory initialisation failure: %s — continuing without persistent memory.",
            exc,
        )
    bus.register_handlers(memory)

    # ── LLM ───────────────────────────────────────────────────────────────
    llm = GeminiClient()
    await llm.initialize()

    # ── NLU ───────────────────────────────────────────────────────────────
    nlu = NLUClassifier(llm_client=llm, bus=bus)
    nlu.register()

    # ── Reasoning request bridge ──────────────────────────────────────────
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

    # ── Reasoning Coordinator ─────────────────────────────────────────────
    coordinator = ReasoningCoordinator(
        bus=bus,
        gemini_client=llm,
        memory_manager=memory,
    )
    coordinator.register()

    # ── Task Executor ─────────────────────────────────────────────────────
    task_executor = TaskExecutor(bus=bus)
    task_executor.register()

    # ── Output: console formatter ────────────────────────────────────────
    formatter = ResponseFormatter(bus=bus)
    if enable_console_formatter:
        formatter.register()

    # ── Output: TTS (optional) ───────────────────────────────────────────
    tts_handler: TTSHandler | None = None
    if enable_tts:
        tts_handler = TTSHandler(bus=bus)
        tts_handler.register()

    logger.info(
        "Pipeline built (session=%s, tts=%s, console_formatter=%s)",
        session_id,
        enable_tts,
        enable_console_formatter,
    )

    return Pipeline(
        bus=bus,
        memory=memory,
        llm=llm,
        nlu=nlu,
        coordinator=coordinator,
        task_executor=task_executor,
        formatter=formatter,
        tts=tts_handler,
        session_id=session_id,
        tts_enabled=enable_tts,
    )


async def shutdown_pipeline(pipeline: Pipeline, drain_timeout: float = 3.0) -> None:
    """Drain in-flight event handlers and close memory backends cleanly."""
    try:
        await asyncio.wait_for(pipeline.bus.drain(), timeout=drain_timeout)
    except asyncio.TimeoutError:
        logger.warning("Bus drain timed out — some handlers may not have completed")

    await pipeline.bus.close()

    try:
        await pipeline.memory.close()
    except Exception as exc:
        logger.warning("Memory close error (non-fatal): %s", exc)

    logger.info("Pipeline shut down (session=%s)", pipeline.session_id)
