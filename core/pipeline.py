"""Reusable KANCHA pipeline construction.

Both the CLI entry point (``main.py``) and the FastAPI/WebSocket server
(``api/server.py``) need the exact same EventBus wiring. Before this module
existed, that ~150 lines of wiring lived inline inside ``main.py:_run()``,
which meant the API server would have had to duplicate it (and the two
copies would inevitably drift).

The control flow it wires up is a hierarchy, not a fan-out::

    input (mic / text / WebSocket)
        -> ReasoningCoordinator          Conversation LLM — the controller
             |- ResponseReady            -> ResponseFormatter, TTS
             `- TaskRequested            explicit delegation
                  -> Planner             Task LLM — the executor
                       -> PlanScheduler -> PlanExecutor -> TaskExecutor
                       -> TaskResultReady
                            -> ReasoningCoordinator -> ResponseReady

The Conversation LLM is the only subscriber to user input; the Task LLM is
the only consumer of ``TaskRequested``. See ``reasoning/coordinator.py``
and ``planning/planner.py`` for the two halves.

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
    ResponseReady,
    SystemError,
    SystemMonitorAlert,
    TextInputReceived,
    TranscriptReady,
)
from core.system_monitor_loop import SystemMonitorLoop
from memory.manager import MemoryManager
from memory.rag import (
    IngestService,
    MemoryRouter,
    RAGConfig,
    RAGManager,
    RagFileSync,
    build_router,
)
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


async def _warm_and_sync_rag(
    rag_manager: RAGManager,
    rag_config: RAGConfig,
    session_id: str,
) -> None:
    """Background boot task: warm the index, then replay ``memory/rag.txt``.

    Runs detached from ``build_pipeline`` so neither step delays the
    assistant becoming responsive. Both halves are individually guarded —
    a failure here costs recall, never availability.
    """
    if rag_config.warm_on_boot:
        try:
            await rag_manager.warm()
        except Exception as exc:  # noqa: BLE001
            logger.debug("RAG warm-up failed (non-fatal): %s", exc)

    if not rag_config.sync_on_boot:
        return

    try:
        report = await RagFileSync(rag_config, rag_manager).run(session_id=session_id)
        if report.entries_indexed:
            # Newly-indexed rows invalidated the similarity matrix; rebuild
            # it now so the next user turn still gets a warm index.
            await rag_manager.warm()
        logger.info("RAG boot sync complete — %s", report.summary())
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG boot sync failed (non-fatal): %s", exc)


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
    # ── RAG (long-term semantic memory) ───────────────────────────────
    # All three are None when RAG is disabled (KANCHA_RAG_ENABLED=0) or
    # when its startup failed; every consumer already null-checks, so a
    # broken vector store degrades the assistant instead of killing it.
    rag: RAGManager | None = None
    rag_router: MemoryRouter | None = None
    rag_ingest: IngestService | None = None
    rag_enabled: bool = False


async def build_pipeline(
    session_id: str = "default",
    data_dir: Path | None = None,
    enable_tts: bool = True,
    enable_console_formatter: bool = True,
    enable_system_monitor: bool = True,
    monitor_interval_s: float | None = None,
    enable_rag: bool | None = None,
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
    enable_rag:
        Build the RAG subsystem (long-term semantic memory + document
        upload index). ``None`` defers to ``KANCHA_RAG_ENABLED`` (default
        on). Pass ``False`` in tests that must not touch the vector store.
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

    # ── RAG: long-term semantic memory ──────────────────────────────────
    # SQLite (above) keeps structured facts; this keeps meaning. The two
    # are deliberately separate stores with separate write paths — see
    # memory/rag/__init__.py for the split.
    #
    # Startup is best-effort: a missing embedding key or an unreadable
    # vector store must degrade the assistant to "no long-term recall",
    # never prevent it from booting.
    rag_config = RAGConfig.from_env(data_dir=data_dir)
    rag_manager: RAGManager | None = None
    rag_router: MemoryRouter | None = None
    rag_ingest: IngestService | None = None

    rag_wanted = rag_config.enabled if enable_rag is None else enable_rag
    if rag_wanted:
        try:
            rag_manager = RAGManager(rag_config)
            await rag_manager.initialize()
            rag_router = build_router(rag_config)
            rag_ingest = IngestService(rag_config, rag_manager)
            logger.info(
                "RAG enabled — router=%s, %s", rag_router.name, rag_config.describe()
            )

            # Warm the index and replay memory/rag.txt in the background.
            #
            # Deliberately NOT awaited. The first boot after a long
            # conversation history has to embed every entry in the audit
            # log, which can take minutes — blocking startup on that would
            # leave the user staring at a dead assistant. Warm-up runs
            # first (fast, and it makes the very first query cheap), then
            # the replay converges the index a few seconds later.
            asyncio.create_task(
                _warm_and_sync_rag(rag_manager, rag_config, session_id),
                name="rag_boot_sync",
            )
        except Exception as exc:
            logger.warning(
                "RAG initialisation failed: %s — continuing without semantic memory.",
                exc,
            )
            rag_manager = None
            rag_router = None
            rag_ingest = None
    else:
        logger.info("RAG disabled (KANCHA_RAG_ENABLED=0 or enable_rag=False)")

    # ── LLM + token logging ─────────────────────────────────────────────
    token_log = TokenLog(
        session_id=session_id,
        sink=jsonl_sink(data_dir / "token_log.jsonl"),
    )
    llm = GeminiClient(token_log=token_log)
    await llm.initialize()

    # ── NLU ───────────────────────────────────────────────────────────────
    # Built but deliberately NOT subscribed to the bus. User input goes to
    # exactly one LLM — the ReasoningCoordinator (Conversation LLM) — which
    # is the single authority on whether a turn needs a task. Registering
    # the classifier here would put a second, independent intent decision
    # on every utterance, which is the fan-out this pipeline used to have.
    #
    # Its deterministic matcher (`nlu.classifier.classify_tool_request`)
    # still earns its keep: the Planner calls it to pick a tool for work
    # that has *already* been delegated, which keeps simple actions free of
    # a second LLM round-trip. The object below is kept on the Pipeline
    # handle for callers that want to classify text on demand.
    nlu = NLUClassifier(llm_client=llm, bus=bus)

    # ── RAG query prefetch ────────────────────────────────────────────────
    # Latency, not correctness. Embedding the query is the only slow step
    # in retrieval (a network round-trip); the search itself is a warm
    # numpy dot product. Starting the embedding the moment the transcript
    # lands lets it finish inside the window the NLU classifier already
    # spends, so by the time the coordinator asks for context it is
    # cached and the retrieval costs effectively nothing.
    #
    # The router gate is applied here too, so "hi" and "open chrome" never
    # trigger an embedding call at all.
    if rag_manager is not None and rag_router is not None:

        async def _prefetch_rag_query(event: TranscriptReady | TextInputReceived) -> None:
            text = (getattr(event, "text", "") or "").strip()
            if not text:
                return
            try:
                decision = await rag_router.decide(text)
            except Exception:  # noqa: BLE001
                return
            if decision.retrieve:
                rag_manager.prefetch(decision.query or text)

        bus.subscribe(TranscriptReady, _prefetch_rag_query)
        bus.subscribe(TextInputReceived, _prefetch_rag_query)

    # ── Conversation LLM (controller) ─────────────────────────────────────
    # The single entry point for user input. Decides whether a turn is
    # conversation or work, resolves "do it"/"that one" against the session
    # history, and delegates via TaskRequested.
    coordinator = ReasoningCoordinator(
        bus=bus,
        gemini_client=llm,
        memory_manager=memory,
        token_log=token_log,
        rag_manager=rag_manager,
        rag_router=rag_router,
    )
    coordinator.register()

    # ── Task Executor ─────────────────────────────────────────────────────
    task_executor = TaskExecutor(bus=bus)
    task_executor.register()

    # ── Task LLM (Planner) + Plan Scheduler ───────────────────────────────
    # These run on top of the existing task executor. The Planner subscribes
    # to TaskRequested — the Conversation LLM's delegation, never raw user
    # input — and decomposes it; the PlanScheduler walks the dependency graph
    # and dispatches individual tasks via the existing TaskExecutor (re-using
    # TaskExecutionRequested). The Planner then reports the outcome back to
    # the Conversation LLM as TaskResultReady.
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
        "Pipeline built (session=%s, tts=%s, console_formatter=%s, rag=%s)",
        session_id,
        enable_tts,
        enable_console_formatter,
        rag_manager is not None,
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
        rag=rag_manager,
        rag_router=rag_router,
        rag_ingest=rag_ingest,
        rag_enabled=rag_manager is not None,
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

    if pipeline.rag is not None:
        try:
            await pipeline.rag.close()
        except Exception as exc:
            logger.warning("RAG close error (non-fatal): %s", exc)

    logger.info("Pipeline shut down (session=%s)", pipeline.session_id)
