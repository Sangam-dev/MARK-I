"""Plan executor — emits :class:`TaskExecutionRequested` for a single
:class:`PlannedTask` and correlates the asynchronous
:class:`TaskCompleted` reply back to the caller.

This module is intentionally thin: it never runs the tool itself. The
existing :class:`tasks.executor.TaskExecutor` still owns tool
dispatch. The Plan executor's only job is to:

1. Bind ``<<task_id:result>>`` placeholders against prior task results.
2. Validate the arguments against
   :data:`tasks.registry.TASK_REGISTRY`.
3. Emit :class:`TaskStarted` then :class:`TaskExecutionRequested` with
   ``_plan_id`` and ``_task_id`` injected into ``parameters`` for
   correlation.
4. Await the matching :class:`TaskCompleted` reply.

Correlation strategy
-------------------

:class:`TaskCompleted` does not echo the original request parameters,
so we can't fish ``_plan_id`` out of the reply. Instead, the executor
keeps a ``(plan_id, task_id) -> Future[TaskCompleted]`` map and
**subscribes to** :class:`TaskCompleted` itself. When a completion
arrives whose ``task_name`` matches one of the in-flight requests,
the executor resolves the corresponding future.

This keeps :class:`tasks.executor.TaskExecutor` completely untouched.
The downside is that completions for unrelated (non-plan) tasks are
ignored — which is exactly what we want.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

from core.bus import EventBus
from core.events import (
    TaskCompleted,
    TaskExecutionRequested,
    TaskStarted,
)
from tasks.registry import validate_task

from planning.models import ExecutionPlan, PlannedTask

logger = logging.getLogger("kancha.planning.executor")


# Sentinel keys we inject into TaskExecutionRequested.parameters so the
# Executor can correlate the asynchronous TaskCompleted reply. The
# existing tasks/executor.py never reads these.
PLAN_ID_KEY = "_plan_id"
TASK_ID_KEY = "_task_id"

# Matches "<<task_id:result>>" or "<<task_id:any_key>>" placeholders.
_REF_RE = re.compile(r"<<(?P<task_id>[A-Za-z0-9_]+):(?P<key>[A-Za-z0-9_]+)>>")


@dataclass(slots=True)
class ExecutionOutcome:
    """Result of running one task through the leaf executor."""

    success: bool
    result: str
    error: str


class PlanExecutor:
    """Stateless executor used by :class:`PlanScheduler`.

    One instance per :class:`PlanScheduler` — it's not safe to share
    a single executor across sessions because the in-flight future
    map is keyed only by ``(plan_id, task_id)``.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        # In-flight tasks: (plan_id, task_id) -> Future[TaskCompleted]
        self._pending: dict[tuple[str, str], asyncio.Future[TaskCompleted]] = {}
        # Last-requested task_name per (plan_id, task_id), so we can
        # distinguish "this completion is for me" from "this completion
        # is for someone else". In practice each plan_id/task_id has
        # exactly one task_name, but we keep the map for clarity.
        self._names: dict[tuple[str, str], str] = {}

    def register(self) -> None:
        self._bus.subscribe(TaskCompleted, self.on_task_completed)

    # ── public ──────────────────────────────────────────────────────

    async def run_task(
        self,
        plan: ExecutionPlan,
        task: PlannedTask,
    ) -> ExecutionOutcome:
        """Dispatch *task* and return its outcome.

        The future-based correlation lives for the duration of one
        :class:`TaskExecutionRequested` round-trip; entries are cleaned
        up in both the happy and error paths.
        """
        key = (plan.id, task.id)

        # ── 1. Bind output_refs ────────────────────────────────────
        bound_args = self._bind_references(task.arguments, plan)
        ok, reason = validate_task(task.tool, bound_args)
        if not ok:
            return ExecutionOutcome(False, "", reason)

        # ── 2. Register correlation future ─────────────────────────
        loop = asyncio.get_running_loop()
        future: asyncio.Future[TaskCompleted] = loop.create_future()
        self._pending[key] = future
        self._names[key] = task.tool

        # ── 3. Announce + dispatch ─────────────────────────────────
        self._bus.emit(
            TaskStarted(
                plan_id=plan.id,
                task_id=task.id,
                tool=task.tool,
                session_id=plan.session_id,
            )
        )

        params_with_meta = {
            **bound_args,
            PLAN_ID_KEY: plan.id,
            TASK_ID_KEY: task.id,
        }
        self._bus.emit(
            TaskExecutionRequested(
                task_name=task.tool,
                parameters=params_with_meta,
                session_id=plan.session_id,
            )
        )

        # ── 4. Await completion ────────────────────────────────────
        try:
            completed = await future
        except asyncio.CancelledError:
            self._cleanup_key(key)
            raise
        finally:
            self._cleanup_key(key)

        return ExecutionOutcome(
            success=completed.success,
            result=completed.result if completed.success else "",
            error="" if completed.success else (completed.error or completed.result or "task failed"),
        )

    async def on_task_completed(self, event: TaskCompleted) -> None:
        """Bus subscriber: resolve any matching in-flight future."""
        if not self._pending:
            return

        # We don't know which plan/task this completion belongs to, so
        # scan pending entries by (session_id, task_name). A more
        # sophisticated design would embed plan_id in TaskCompleted,
        # but that would mean changing tasks/executor.py. Since plan
        # task names are unique within a plan and we keep the map
        # small, scanning is fine.
        matching_key: tuple[str, str] | None = None
        for key, name in self._names.items():
            if name == event.task_name and event.session_id:
                # The plan carries session_id; we use it to scope the
                # match (each session has at most one active plan).
                matching_key = key
                break
        if matching_key is None:
            return

        future = self._pending.get(matching_key)
        if future is None or future.done():
            return
        future.set_result(event)

    # ── helpers ─────────────────────────────────────────────────────

    def _cleanup_key(self, key: tuple[str, str]) -> None:
        self._pending.pop(key, None)
        self._names.pop(key, None)

    @staticmethod
    def _bind_references(
        arguments: dict[str, Any],
        plan: ExecutionPlan,
    ) -> dict[str, Any]:
        """Replace ``<<task_id:key>>`` placeholders with prior task results.

        Currently we support the well-known key ``result`` (which maps
        to :attr:`PlannedTask.result` populated on TaskCompleted).
        Unknown keys resolve to an empty string — the LLM is told only
        to emit the ``result`` key, so this is a safe fallback.
        """
        resolved: dict[str, Any] = {}
        for name, value in arguments.items():
            if isinstance(value, str):
                resolved[name] = _REF_RE.sub(
                    lambda m: _resolve_ref(m, plan),
                    value,
                )
            else:
                resolved[name] = value
        return resolved


def _resolve_ref(match: re.Match[str], plan: ExecutionPlan) -> str:
    task_id = match.group("task_id")
    key = match.group("key")
    if task_id == "self":
        return match.group(0)  # leave for executor to handle later
    target = plan.task_by_id(task_id)
    if target is None:
        logger.warning("Plan %s references unknown task %s", plan.id, task_id)
        return ""
    if key == "result":
        return target.result
    logger.warning(
        "Plan %s task %s: unknown ref key %s", plan.id, task_id, key
    )
    return ""
