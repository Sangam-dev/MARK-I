"""Reusable KANCHA pipeline construction.

Both the CLI entry point (``main.py``) and the FastAPI/WebSocket server
(``api/server.py``) need the exact same EventBus wiring: EventBus -> Memory ->
LLM -> NLU -> Reasoning bridge -> ReasoningCoordinator -> Planner ->
PlanScheduler -> TaskExecutor -> output handlers. Before this module existed,
that ~150 lines of wiring lived inline inside ``main.py:_run()``, which meant
the API server would have had to duplicate it (and the two copies would
inevitably drift).

``build_pipeline()`` is now the single source of truth. Anything that needs a
running KANCHA pipeline (CLI, API server, tests, a future gRPC front door,
whatever) calls this instead of re-wiring the bus by hand.

See answers/guide.md for how to extend this when adding new modules/actions.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from core.bus import EventBus
from core.events import (
    IntentIdentified,
    MemoryRetrieved,
    ReasoningRequested,
    ResponseReady,
    SystemError,
    SystemMonitorAlert,
)
from core.system_monitor_loop import SystemMonitorLoop
from memory.manager import MemoryManager
from memory.token_log import TokenLog, jsonl_sink
from nlu.classifier import NLUClassifier
from output.response_formatter import ResponseFormatter
from output.tts import TTSHandler
from planning.planner import Planner
from planning.scheduler import PlanScheduler
from reasoning.coordinator import ReasoningCoordinator
from reasoning.llm_client import GeminiClient
from tasks.executor import (
    TaskExecutor,
    attach_monitor_loop,
    detach_monitor_loop,
    get_shared_monitor,
)

logger = logging.getLogger("kancha.pipeline")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "memory" / "data"

# Default polling interval for the background SystemMonitorLoop. Overridable via
# the KANCHA_MONITOR_INTERVAL env var (seconds). 30s is intentionally light —
# psutil.cpu_percent(interval=None) is already cheap, and the 300s cooldown
# inside actions/system_monitor.py throttles actual alerts.
DEFAULT_MONITOR_INTERVAL = 30


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
    planner: Planner
    plan_scheduler: PlanScheduler
    formatter: ResponseFormatter
    tts: TTSHandler | None
    monitor_loop: SystemMonitorLoop | None
    session_id: str
    tts_enabled: bool
    monitor_enabled: bool


async def build_pipeline(
    session_id: str = "default",
    data_dir: Path | None = None,
    enable_tts: bool = True,
    enable_console_formatter: bool = True,
    enable_system_monitor: bool = True,
    monitor_interval_s: float | None = None,
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
    enable_system_monitor:
        Spawn the background :class:`~core.system_monitor_loop.SystemMonitorLoop`
        that polls CPU/RAM/temp/GPU and emits ``SystemMonitorAlert`` when a
        threshold is crossed. Disabled callers (e.g. tests) skip the loop and
        the on-demand ``system_monitor`` task no-ops the enable/disable action.
    monitor_interval_s:
        Override the polling interval (seconds). Defaults to
        ``DEFAULT_MONITOR_INTERVAL`` (30s) or the ``KANCHA_MONITOR_INTERVAL``
        env var if set.
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

    # ── Background system monitor ───────────────────────────────────────
    # Forward SystemMonitorAlert to the existing ResponseReady pipeline so
    # TTS and the console formatter speak/print the alert without us
    # duplicating fan-out logic.
    monitor_loop: SystemMonitorLoop | None = None
    if enable_system_monitor:
        async def _announce_monitor_alert(event: SystemMonitorAlert) -> None:
            if not event.text:
                return
            bus.emit(ResponseReady(text=event.text))

        bus.subscribe(SystemMonitorAlert, _announce_monitor_alert)

        if monitor_interval_s is None:
            env_value = os.getenv("KANCHA_MONITOR_INTERVAL")
            try:
                monitor_interval_s = float(env_value) if env_value else DEFAULT_MONITOR_INTERVAL
            except ValueError:
                monitor_interval_s = DEFAULT_MONITOR_INTERVAL
        monitor_interval_s = max(1.0, monitor_interval_s)

        monitor = get_shared_monitor()
        monitor_loop = SystemMonitorLoop(
            bus=bus,
            monitor=monitor,
            interval_s=monitor_interval_s,
            enabled=True,
        )
        attach_monitor_loop(monitor_loop)
        monitor_loop.start()
        logger.info(
            "SystemMonitorLoop started  (interval=%.1fs, thresholds=%s)",
            monitor_interval_s,
            monitor.thresholds,
        )

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

    # ── LLM + token logging ─────────────────────────────────────────────
    token_log = TokenLog(
        session_id=session_id,
        sink=jsonl_sink(data_dir / "token_log.jsonl"),
    )
    llm = GeminiClient(token_log=token_log)
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
        token_log=token_log,
    )
    coordinator.register()

    # ── Task Executor ─────────────────────────────────────────────────────
    task_executor = TaskExecutor(bus=bus)
    task_executor.register()

    # ── Planner + Plan Scheduler ──────────────────────────────────────────
    # These run on top of the existing task executor. The Planner decomposes
    # multi-step requests; the PlanScheduler walks the dependency graph and
    # dispatches individual tasks via the existing TaskExecutor (re-using
    # TaskExecutionRequested).
    planner = Planner(
        bus=bus,
        llm=llm,
        memory=memory,
        token_log=token_log,
    )
    planner.register()

    plan_scheduler = PlanScheduler(bus=bus)
    plan_scheduler.register()
    plan_scheduler.executor.register()

    # ── Output: console formatter ────────────────────────────────────────
    formatter = ResponseFormatter(bus=bus)
    if enable_console_formatter:
        formatter.register()

    # ── Output: TTS (optional) ───────────────────────────────────────────
    tts_handler: TTSHandler | None = None
    if enable_tts:
        tts_handler = TTSHandler(bus=bus)
        tts_handler.register()
        # Fire-and-forget warmup so the first user utterance doesn't
        # pay the edge-tts cold-connection cost (~200–400ms of TLS +
        # WebSocket handshake). The task is intentionally not awaited —
        # pipeline startup must not block on TTS readiness.
        asyncio.create_task(tts_handler.warmup(), name="tts_warmup")

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
        planner=planner,
        plan_scheduler=plan_scheduler,
        formatter=formatter,
        tts=tts_handler,
        monitor_loop=monitor_loop,
        session_id=session_id,
        tts_enabled=enable_tts,
        monitor_enabled=enable_system_monitor,
    )


async def shutdown_pipeline(pipeline: Pipeline, drain_timeout: float = 3.0) -> None:
    """Drain in-flight event handlers and close memory backends cleanly."""
    # Stop the background monitor loop FIRST so it can't emit a new
    # SystemMonitorAlert after the bus drain begins.
    if pipeline.monitor_loop is not None:
        try:
            await pipeline.monitor_loop.stop()
        except Exception as exc:
            logger.warning("SystemMonitorLoop stop error (non-fatal): %s", exc)
        detach_monitor_loop()

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
