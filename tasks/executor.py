from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

from actions.alarms import cancel_alarms, list_alarms, set_alarm
from actions.apps import open_app
from actions.desktop_control import desktop_control
from actions.file_controller import file_controller, resolve_fuzzy_name
from actions.gmail_tool import get_shared_gmail_tool
from actions.system_commands import SystemCommandExecutor
from actions.system_monitor import (
    SystemMonitor,
    format_status_text,
    get_system_status,
)
from actions.system_tool import get_shared_system_tool
from actions.weather import get_weather
from actions.web_search import web_search
from actions.youtube_video import youtube_video
from agent.tool import get_shared_opencode_tool
from core.bus import EventBus
from core.events import TaskCompleted, TaskExecutionRequested
from tasks.registry import TASK_REGISTRY, validate_task

logger = logging.getLogger("kancha.tasks.executor")

# file_operation has NO 'open' action (planning/prompts.py rule 4d), but
# the Task LLM still occasionally emits one — route it to the `system`
# tool instead of failing with "Unknown action: 'open_path'".
_OPEN_ALIASES = frozenset({"open", "open_path", "open_file", "open_folder"})


def _resolve_open_target(
    params: dict[str, Any],
) -> tuple[str | None, list[tuple[str, str]] | None]:
    """Best-effort fuzzy resolution for an ``open_path`` target.

    Returns ``(path, None)`` when confident, ``(None, candidates)`` when
    several files tie (the caller replies with the list instead of
    guessing), and ``(None, None)`` to leave the request untouched — the
    tool then answers with its normal not-found message.
    """
    target = str(params.get("target") or params.get("path") or "").strip()
    if not target or "://" in target or target.startswith("www."):
        return None, None  # nothing to resolve, or a URL rather than a name
    path, candidates = resolve_fuzzy_name(target)
    if path:
        return path, None
    if candidates:
        return None, candidates
    return None, None

_FILE_FAILURE_RE = re.compile(
    r"^(?:access denied|path not found|not a directory|permission denied|"
    r"could not|not found|source not found|no destination specified|"
    r"file not found|not a file|search path not found|search error|"
    r"error|unknown action|file controller error|protected directory|"
    r"no\s+[^\n]+?\s+found\b)",  # "No sss.pdf found in Desktop" is a negative result
    re.IGNORECASE,
)

# Desktop control results are plain strings; the same failure-prefix
# regex family covers "Could not", "No open window", "Neither", "Install",
# "Required tool not found", "Wallpaper query not supported", etc.
# Desktop_control returns longer multi-line strings for list-style
# actions (list_windows, list_workspaces, list_desktop, stats) so we
# only treat short single-line strings as failures.
_DESKTOP_FAILURE_RE = re.compile(
    r"^(?:could not|no open window|no action|no app name|neither|"
    r"required tool not found|please describe|unknown action|"
    r"unsupported (?:format|image format|operating system)|"
    r"install|wallpaper query not supported|no image path|no url provided|"
    r"this action cannot|execution error|desktop control error|"
    r"task cancelled|image not found)",
    re.IGNORECASE,
)

# system_monitor action surface — kept in one place so on-call triage can
# audit it without grepping the dispatch body. Failure prefixes for
# `set_threshold`/`enable`/`disable` mirror the desktop_control style.
_SYSTEM_MONITOR_FAILURE_RE = re.compile(
    r"^(?:unknown action|unknown metric|invalid threshold|"
    r"metric required|threshold required|enabled required|"
    r"value out of range|could not|system monitor error)",
    re.IGNORECASE,
)

_VALID_METRICS = ("cpu", "ram", "temp", "gpu")

# A single, process-wide SystemMonitor instance keeps the cooldown timers
# alive across turns. The background loop in core/pipeline.py reads from
# this same instance via ``core.system_monitor_loop.SystemMonitorLoop``.
_monitor_instance: SystemMonitor | None = None


def get_shared_monitor() -> SystemMonitor:
    """Lazily construct (and cache) the singleton SystemMonitor.

    Cooldown state — ``_last_alert`` and ``_cpu_streak`` — survives
    across turns and across the on-demand / background dispatch paths
    so the 300s cooldown isn't reset every time the user asks "any alerts?".
    """
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = SystemMonitor()
    return _monitor_instance


def reset_shared_monitor() -> None:
    """Drop the singleton (used by tests)."""
    global _monitor_instance
    _monitor_instance = None


# The background loop attaches itself here so ``enable``/``disable`` actions
# can toggle it without the executor importing core.pipeline (which would
# create a circular import — pipeline imports the executor).
_attached_monitor_loop = None


def _get_attached_monitor_loop():
    return _attached_monitor_loop


def attach_monitor_loop(loop) -> None:
    """Called by ``core.pipeline.build_pipeline`` after constructing the loop."""
    global _attached_monitor_loop
    _attached_monitor_loop = loop


def detach_monitor_loop() -> None:
    """Called by ``core.pipeline.shutdown_pipeline``."""
    global _attached_monitor_loop
    _attached_monitor_loop = None


@dataclass(slots=True)
class TaskExecutionResult:
    success: bool
    message: str
    #: Structured failure class (see :mod:`core.failures`). None on success
    #: or when the failure is transient/unknown (retryable).
    error_type: str | None = None


def classify_failure(message: str) -> str | None:
    """Map a failure message to a permanent failure class, or ``None`` when
    the failure looks transient/unknown (and therefore may be retried).

    Conservative by design: a false *negative* (permanent failure treated as
    retryable) is bounded by the scheduler's retry cap and the planner's
    replan guards; a false *positive* (transient failure treated as terminal)
    would skip a legitimate retry, so only clearly-permanent wording is
    classified.
    """
    text = (message or "").strip()
    if not text:
        return None
    low = text.lower()

    if any(
        m in low
        for m in (
            "does not exist",
            "file not found",
            "path not found",
            "search path not found",
            "source not found",
            "not a directory",
            "not a file",
            "no such file",
            "no open window",
            "no windows found",
            "no files found",
        )
    ):
        return "not_found"
    # "No sss.pdf found in Desktop" — the search-miss shape.
    if re.search(r"\bno\s+[\w ._+-]+?\s+found\b", low):
        return "not_found"

    if any(
        m in low
        for m in (
            "is required",
            "must be one of",
            "must be a ",
            "must be between",
            "use one of",
            "unknown action",
            "unknown system action",
            "unknown metric",
            "unknown protocol",
            "numeric threshold",
            "invalid threshold",
            "missing required param",
            "not in the registry",
            "no handler found for task",
        )
    ):
        return "invalid_argument"

    if any(
        m in low
        for m in (
            "not supported",
            "unsupported",
            "none is installed",
            "is not installed",
            "this action cannot",
        )
    ):
        return "unsupported"

    if any(
        m in low
        for m in (
            "access denied",
            "permission denied",
            "protected directory",
            "requires confirmation",
            "rejected the credentials",
        )
    ):
        return "permission"

    return None


class TaskExecutor:
    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    def register(self) -> None:
        self._bus.subscribe(TaskExecutionRequested, self.on_task_requested)

    async def on_task_requested(self, event: TaskExecutionRequested) -> None:
        start = asyncio.get_running_loop().time()

        is_valid, reason = validate_task(event.task_name, event.parameters)
        if not is_valid:
            await self._bus.emit_and_wait(
                TaskCompleted(
                    task_name=event.task_name,
                    success=False,
                    error=reason,
                    # A plan-level validation failure is a bad argument,
                    # not a transient condition — never retried.
                    error_type="invalid_argument",
                    session_id=event.session_id,
                )
            )
            return

        spec = TASK_REGISTRY[event.task_name]
        if spec.requires_confirmation:
            await self._bus.emit_and_wait(
                TaskCompleted(
                    task_name=event.task_name,
                    success=False,
                    error=(
                        f"Action '{event.task_name}' requires confirmation. "
                        f"Please say 'confirm {event.task_name}' to proceed."
                    ),
                    error_type="permission",
                    session_id=event.session_id,
                )
            )
            return

        try:
            result = await self._dispatch(event.task_name, dict(event.parameters))
        except Exception as exc:
            logger.exception("Task %s failed with exception", event.task_name)
            await self._bus.emit_and_wait(
                TaskCompleted(
                    task_name=event.task_name,
                    success=False,
                    error=str(exc),
                    error_type=classify_failure(str(exc)),
                    session_id=event.session_id,
                )
            )
            return

        elapsed = asyncio.get_running_loop().time() - start
        logger.info("Task %s completed in %.3fs", event.task_name, elapsed)
        await self._bus.emit_and_wait(
            TaskCompleted(
                task_name=event.task_name,
                success=result.success,
                result=result.message if result.success else "",
                error="" if result.success else result.message,
                error_type=(
                    None if result.success else classify_failure(result.message)
                ),
                session_id=event.session_id,
            )
        )

    async def _dispatch(
        self, task_name: str, params: dict[str, Any]
    ) -> TaskExecutionResult:
        """Route task to the appropriate action function (all blocking calls run in thread)."""

        if task_name == "open_app":
            # Pass the whole param dict, not just the name: `target` is
            # what makes "open kancha in vscode" open the project rather
            # than an empty editor window.
            params = dict(params)
            params = self._resolve_project_target(params)
            result = await asyncio.to_thread(open_app, params)
            return TaskExecutionResult(result.success, result.message)

        if task_name == "set_alarm":
            description = params.get("description", "")
            delay_seconds = int(params.get("delay_seconds", 60))
            result = await asyncio.to_thread(set_alarm, description, delay_seconds)
            return TaskExecutionResult(result.success, result.message)

        if task_name == "list_alarms":
            result = await asyncio.to_thread(list_alarms)
            return TaskExecutionResult(result.success, result.message)

        if task_name == "cancel_alarms":
            result = await asyncio.to_thread(cancel_alarms)
            return TaskExecutionResult(result.success, result.message)

        if task_name == "get_weather":
            city = params.get("city", "")
            date = params.get("date")
            units = params.get("units")
            result = await asyncio.to_thread(get_weather, city, date, units)
            return TaskExecutionResult(result.success, result.message)

        # No sleep / shutdown / restart branch. Power-state control was
        # removed from the assistant entirely — see the note in
        # tasks/registry.py. A request naming one falls through to the
        # "no handler" result at the bottom of this method.

        if task_name == "file_operation":
            action = str(params.get("action") or "").strip().lower()
            if action in _OPEN_ALIASES:
                # Rule 4d: opens belong to `system`, but the Task LLM
                # still occasionally emits file_operation — route it so
                # the request succeeds instead of dying on an unknown
                # action.
                logger.info(
                    "file_operation action=%r is an open — routing to system/open_path",
                    action,
                )
                open_params: dict[str, Any] = {"action": "open_path"}
                for key in ("target", "path"):
                    if params.get(key):
                        open_params[key] = str(params[key])
                if not open_params.get("target"):
                    open_params["target"] = str(params.get("name") or "")
                return await self._dispatch("system", open_params)
            result_text = await asyncio.to_thread(file_controller, params)
            return TaskExecutionResult(
                not _FILE_FAILURE_RE.match(result_text.strip()),
                result_text,
            )

        if task_name == "desktop_control":
            # desktop_control is blocking (subprocess, file IO, possibly
            # an LLM round-trip for the `task` action) — run on a worker
            # thread so the event loop stays free.
            result_text = await asyncio.to_thread(desktop_control, params)
            stripped = result_text.strip()
            # Multi-line results are list/listing replies (list_windows,
            # list_workspaces, list_desktop, stats) — those are always
            # successes as long as we got output. Single-line strings
            # that look like error messages are treated as failures so
            # the naturalize / replanner path can handle them.
            is_failure = "\n" not in stripped and bool(
                _DESKTOP_FAILURE_RE.match(stripped)
            )
            return TaskExecutionResult(not is_failure, result_text)

        if task_name == "execute_protocol":
            protocol_name = params.get("protocol_name", "")
            original_request = params.get("original_request", "")

            # Map protocol names to script paths
            protocol_scripts = {
                "genesis": "tests/scripts.py",
            }

            script_path = protocol_scripts.get(protocol_name)
            if not script_path:
                return TaskExecutionResult(
                    False,
                    f"Unknown protocol: {protocol_name}. Available: {list(protocol_scripts.keys())}",
                )

            # Run the protocol script using SystemCommandExecutor
            executor = SystemCommandExecutor()
            # Add "uv" to allowed commands since it's not in the default allowlist
            executor.add_allowed_command("uv")
            command = f"uv run main.py --text --script scripts/{protocol_name}.json"
            result = await executor.execute(command)

            if result.success:
                return TaskExecutionResult(
                    True, f"Protocol '{protocol_name}' executed successfully"
                )
            else:
                return TaskExecutionResult(
                    False, f"Protocol '{protocol_name}' failed: {result.stderr}"
                )

        if task_name == "web_search":
            query = params.get("query", "")
            result = await asyncio.to_thread(web_search, query)
            return TaskExecutionResult(result.success, result.message)

        if task_name == "system":
            # SystemTool is already async (asyncio.create_subprocess_exec),
            # so no to_thread hop here. It returns a structured result;
            # this layer flattens it into the executor's success/message.
            params = dict(params)
            # A spoken name that is not a real path yet ("open the study
            # plan file") gets resolved to an absolute path before the
            # tool refuses it — SystemTool only opens existing paths by
            # design.
            if str(params.get("action") or "").strip().lower() == "open_path":
                path, candidates = _resolve_open_target(params)
                if path:
                    params["target"] = path
                elif candidates:
                    listed = "; ".join(
                        f"{i + 1}) {name} ({location})"
                        for i, (name, location) in enumerate(candidates)
                    )
                    return TaskExecutionResult(
                        True,
                        f"I found a few matches for that: {listed}. Which one should I open?",
                    )
            result = await get_shared_system_tool().execute(params)
            if result.success:
                return TaskExecutionResult(True, result.output or "Done.")
            return TaskExecutionResult(False, result.error or "System action failed.")

        if task_name == "youtube_video":
            # yt-dlp's search is blocking, so it goes to a thread like the
            # other synchronous actions.
            result = await asyncio.to_thread(
                youtube_video, params.get("request", "")
            )
            if result.get("success"):
                channel = result.get("channel") or ""
                by = f" by {channel}" if channel else ""
                return TaskExecutionResult(
                    True, f"Playing '{result['title']}'{by} on YouTube."
                )
            return TaskExecutionResult(
                False, result.get("error") or "Could not play that on YouTube."
            )

        if task_name == "gmail":
            # GmailTool is already async (imaplib/smtplib run via
            # to_thread inside the client), so no to_thread hop here.
            # The shared instance is required, not incidental: armed
            # confirmations live on it and must survive between turns.
            result = await get_shared_gmail_tool().execute(params)
            if result.success:
                return TaskExecutionResult(True, result.output or "Done.")
            return TaskExecutionResult(False, result.error or "Gmail action failed.")

        if task_name == "agent_task":
            # The OpenCode layer is already async and owns its own
            # transport, so no to_thread hop. The shared instance is
            # required, not incidental: the session map lives on it, and
            # a follow-up needs the session its predecessor opened.
            #
            # Delegated work runs for minutes, not milliseconds — that is
            # the point of it — so `delegate` starts it in the background
            # and returns; the run is watched via OpenCode's event stream
            # and reported through the `progress` action.
            prepared = self._with_resolved_directory(params)
            prepared, project_name, summary = self._enrich_with_project_context(prepared)
            if project_name and summary:
                await self._index_activity_memory(
                    summary, str(prepared.get("session_id") or "default")
                )
            result = await get_shared_opencode_tool().execute(prepared)
            if result.success:
                return TaskExecutionResult(True, result.output or "Done.")
            return TaskExecutionResult(False, result.error or "Delegation failed.")

        if task_name == "system_monitor":
            return await self._dispatch_system_monitor(params)

        return TaskExecutionResult(False, f"No handler found for task: {task_name}")

    # ── agent_task helpers ──────────────────────────────────────────────────

    @staticmethod
    def _resolve_project_target(params: dict[str, Any]) -> dict[str, Any]:
        """"Open our previous project in vscode" → target=its root.

        Only kicks in when the target is not already a real path (the
        file controller would resolve those) and the project registry
        confidently recognises the name. Unresolved targets pass through
        untouched, so the existing "no such folder" failure still
        surfaces.
        """
        target = str(params.get("target") or "").strip()
        if not target:
            return params
        try:
            from actions.file_controller import _resolve_path  # noqa: PLC0415

            if _resolve_path(target).exists():
                return params
        except Exception:  # noqa: BLE001
            return params
        try:
            from memory.projects import get_shared_project_store  # noqa: PLC0415

            store = get_shared_project_store()
            project = store.resolve_project(target)
        except Exception:  # noqa: BLE001
            return params
        if project is None:
            return params
        updated = dict(params)
        updated["target"] = project.root
        logger.info("open_app target resolved to project '%s' at %s", project.name, project.root)
        return updated

    @staticmethod
    def _enrich_with_project_context(
        params: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None, str | None]:
        """Give a delegated agent the context of the project it is asked
        to continue.

        "continue working on our previous project" reaches OpenCode as a
        bare sentence and a blank workspace. When the instruction names
        or implies a project ("previous project", "study plan", a
        follow-up to the thing we just built), this resolves it, points
        the agent at its root (unless a directory was already named) and
        prepends what the project is and what was last done in it — the
        difference between resuming real work and starting from nothing.

        Returns ``(params, project_name, summary)``; when nothing
        resolves, ``params`` is returned untouched and the other two are
        ``None``. The summary is the condensed record for semantic
        memory. Best-effort: any failure leaves the delegation unchanged.
        """
        if str(params.get("action") or "").strip() not in ("delegate", "follow_up"):
            return params, None, None
        task = str(params.get("task") or params.get("instruction") or "").strip()
        if not task:
            return params, None, None
        try:
            from memory.projects import get_shared_project_store  # noqa: PLC0415

            store = get_shared_project_store()
            project = store.resolve_project(task)
            if project is None:
                return params, None, None
        except Exception:  # noqa: BLE001
            logger.exception("Project-context enrichment failed — delegating unchanged")
            return params, None, None

        updated = dict(params)
        # Point a *new* delegation at the project root. A follow_up must
        # keep its session's own working directory — moving it would
        # detach the continuation from the work already in flight.
        if (
            str(params.get("action") or "").strip() == "delegate"
            and not str(updated.get("directory") or "").strip()
        ):
            updated["directory"] = project.root
        updated["task"] = f"{task}\n\n{store.project_context(project)}"
        try:
            store.record_activity(
                str(params.get("session_id") or "default"),
                task,
                project=project.name,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Could not record project activity")
        summary = store.activity_summary(project, task)
        logger.info(
            "agent_task enriched with project context: '%s' at %s",
            project.name,
            project.root,
        )
        return updated, project.name, summary

    @staticmethod
    def _with_resolved_directory(params: dict[str, Any]) -> dict[str, Any]:
        """Turn a spoken working directory into an absolute path.

        "Build me a site in the projects folder" has to reach OpenCode as
        a real directory, because OpenCode resolves every relative path
        in the task against its own working directory — get it wrong and
        the work lands in the default workspace, which is exactly the
        "it created kancha-workspace and nothing else" failure.

        Resolution happens *here* rather than in ``agent/`` on purpose:
        that package is deliberately isolated from ``actions/`` (see
        agent/tool.py), and the resolver lives in the file controller.
        """
        raw = str(
            params.get("directory") or params.get("workspace") or params.get("path") or ""
        ).strip()
        if not raw:
            return params

        from actions.file_controller import _resolve_path  # noqa: PLC0415

        resolved = _resolve_path(raw)
        if not resolved.is_dir():
            # Creating it is the right call: "build me a site in
            # ~/projects/jarvis_frontend" names where it should go, not
            # somewhere that must already exist.
            try:
                resolved.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("Could not create agent directory %s: %s", resolved, exc)
                return params

        updated = dict(params)
        updated["directory"] = str(resolved)
        logger.info("agent_task will work in %s (from %r)", resolved, raw)
        return updated

    async def _index_activity_memory(self, summary: str, session: str) -> None:
        """Best-effort write of a condensed activity record into the
        isolated semantic store. Never raises, never delays the
        delegation: a store that is not ready (or fails) just skips."""
        try:
            from memory.activity_memory import get_shared_activity_memory  # noqa: PLC0415

            memory = get_shared_activity_memory()
            if not memory.ready:
                return
            await memory.index(summary, session=session)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Project activity index failed (non-fatal): %s", exc)

    # ── system_monitor handler ──────────────────────────────────────────────

    async def _dispatch_system_monitor(
        self, params: dict[str, Any]
    ) -> TaskExecutionResult:
        """Route the ``system_monitor`` task to its action surface.

        All four actions are blocking (psutil + thread + maybe an LLM
        round-trip would be the worst case — we don't do that here) so
        each sub-call is wrapped in ``asyncio.to_thread``.
        """
        action = (params.get("action") or "").lower().strip()

        def _run_sync() -> str:
            monitor = get_shared_monitor()

            if action == "status":
                snapshot = get_system_status()
                return format_status_text(snapshot)

            if action == "check_alerts":
                text = monitor.check()
                return text if text else "No system alerts right now."

            if action == "set_threshold":
                metric = (params.get("metric") or "").lower().strip()
                if metric not in _VALID_METRICS:
                    return (
                        f"Unknown metric '{metric}'. "
                        f"Valid: {', '.join(_VALID_METRICS)}."
                    )
                raw_threshold = params.get("threshold")
                if raw_threshold is None:
                    return (
                        f"Provide a numeric threshold for '{metric}' "
                        "(e.g. threshold=75)."
                    )
                try:
                    threshold = float(raw_threshold)
                except (TypeError, ValueError):
                    return f"Invalid threshold '{raw_threshold}'. Use a number."
                if not 0 < threshold < 100:
                    return "Threshold must be between 1 and 99."
                monitor.thresholds[metric] = threshold
                return f"{metric.upper()} threshold set to {threshold:.0f}%."

            if action in {"enable", "disable"}:
                enabled_flag = bool(params.get("enabled", action == "enable"))
                monitor_loop = _get_attached_monitor_loop()
                if monitor_loop is not None:
                    monitor_loop.set_enabled(enabled_flag)
                state = "enabled" if enabled_flag else "disabled"
                return f"System monitoring {state}."

            return (
                f"Unknown system_monitor action '{action}'. "
                "Use one of: status, check_alerts, set_threshold, enable, disable."
            )

        result_text = await asyncio.to_thread(_run_sync)
        stripped = result_text.strip()
        is_failure = bool(_SYSTEM_MONITOR_FAILURE_RE.match(stripped))
        return TaskExecutionResult(not is_failure, result_text)
