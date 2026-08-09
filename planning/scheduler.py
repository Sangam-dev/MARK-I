"""Scheduler — runs an :class:`ExecutionPlan` to completion.

Responsibilities:

* Track each plan in memory (``self._plans``).
* Detect tasks whose dependencies are all :attr:`TaskStatus.COMPLETED`
  and dispatch them concurrently (bounded by ``max_concurrent``).
* On :class:`TaskCompleted`, decrement dependents' dep counters,
  re-dispatch newly-ready tasks.
* On failure: retry once if the task is ``retryable`` and
  ``max_retries`` allows; otherwise emit
  :class:`PlanReplanRequested` so the Planner can rebuild the
  remaining steps.
* Emit :class:`PlanCompleted` once every task reaches a terminal
  state.

Concurrency model: the Scheduler runs a single
:func:`asyncio.create_task` per plan; tasks are dispatched inside that
task via ``asyncio.gather`` so the plan's overall state machine stays
synchronous from the Scheduler's perspective even though individual
tool calls run in parallel.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from core.bus import EventBus
from core.events import (
    PlanCancelled,
    PlanCompleted,
    PlanCreated,
    PlanReplanRequested,
)
from planning.executor import PlanExecutor
from planning.models import ExecutionPlan, PlannedTask, PlanStatus, TaskStatus

logger = logging.getLogger("kancha.planning.scheduler")


def _natural_summary(plan: ExecutionPlan) -> str:
    """Build a user-facing summary of a finished plan.

    The machine-style "0/1 tasks completed, 1 failed" is internal
    bookkeeping. The user-facing reply should be the actual tool
    results, concatenated when more than one task ran. This keeps
    "what's the weather in London?" producing the weather text rather
    than a status counter.
    """
    completed = [t for t in plan.tasks if t.status == TaskStatus.COMPLETED]
    failed = [t for t in plan.tasks if t.status == TaskStatus.FAILED]
    skipped = [t for t in plan.tasks if t.status == TaskStatus.SKIPPED]

    parts: list[str] = []

    # 1. Successful task results — the meat of the response.
    for t in completed:
        text = (t.result or "").strip()
        if text:
            parts.append(text)
        else:
            parts.append(f"Done: {t.description}")

    # 2. Failures — short, factual, no decoration.
    for t in failed:
        err = (t.error or "").strip() or "unknown error"
        parts.append(f"Couldn't {t.description}: {err}")

    # 3. Skipped — only mention if any.
    for t in skipped:
        parts.append(f"Skipped {t.description} (dependency failed)")

    if not parts:
        return "Nothing to do."
    return "\n".join(parts)


class PlanScheduler:
    """In-memory async scheduler for :class:`ExecutionPlan`."""

    def __init__(
        self,
        bus: EventBus,
        max_concurrent: int = 4,
    ) -> None:
        self._bus = bus
        self._executor = PlanExecutor(bus)
        self._max_concurrent = max_concurrent
        self._plans: dict[str, ExecutionPlan] = {}
        # plan_id -> the asyncio.Task driving the plan
        self._plan_tasks: dict[str, asyncio.Task] = {}
        # session_id -> currently-active plan_id (one plan per session)
        self._active_by_session: dict[str, str] = {}

    # ── registration ──────────────────────────────────────────────

    def register(self) -> None:
        self._bus.subscribe(PlanCreated, self.on_plan_created)
        self._bus.subscribe(PlanCancelled, self.on_plan_cancelled)
        self._bus.subscribe(PlanReplanRequested, self.on_replan_requested)

    # ── public handlers ───────────────────────────────────────────

    async def on_plan_created(self, event: PlanCreated) -> None:
        plan = self._deserialize_plan(event.plan)
        if plan is None:
            logger.error("PlanCreated event had unparseable plan: %r", event.plan)
            return

        session = plan.session_id
        active = self._active_by_session.get(session)
        if active is not None:
            logger.warning(
                "Plan %s arrived but session %s already has active plan %s — "
                "queueing the new plan until the active one finishes.",
                plan.id,
                session,
                active,
            )
            # Wait for the active plan to finish, then run this one.
            existing = self._plan_tasks.get(active)
            if existing is not None:
                try:
                    await existing
                except Exception:  # noqa: BLE001
                    pass

        self._plans[plan.id] = plan
        self._active_by_session[session] = plan.id
        plan.status = PlanStatus.RUNNING
        logger.info(
            "PlanScheduler: starting plan %s (%d tasks)",
            plan.id,
            len(plan.tasks),
        )

        task = asyncio.create_task(
            self._run_plan(plan),
            name=f"plan:{plan.id}",
        )
        self._plan_tasks[plan.id] = task

    async def on_plan_cancelled(self, event: PlanCancelled) -> None:
        plan = self._plans.get(event.plan_id)
        if plan is None:
            return
        plan.status = PlanStatus.CANCELLED
        for task in plan.tasks:
            if task.status in {TaskStatus.PENDING, TaskStatus.READY, TaskStatus.RUNNING}:
                task.status = TaskStatus.CANCELLED
        runner = self._plan_tasks.get(event.plan_id)
        if runner is not None and not runner.done():
            runner.cancel()
        logger.info("Plan %s cancelled: %s", plan.id, event.reason)

    async def on_replan_requested(self, event: PlanReplanRequested) -> None:
        # The Planner is responsible for emitting a new PlanCreated.
        # We just mark the failed task as terminal-failed so the plan's
        # bookkeeping reflects reality.
        plan = self._plans.get(event.plan_id)
        if plan is None:
            return
        task = plan.task_by_id(event.failed_task_id)
        if task is not None and task.status != TaskStatus.COMPLETED:
            task.status = TaskStatus.FAILED
            task.error = event.reason or task.error
        logger.info(
            "Plan %s replan requested: task %s failed (%s)",
            event.plan_id,
            event.failed_task_id,
            event.reason,
        )

    # ── private: state machine ────────────────────────────────────

    async def _run_plan(self, plan: ExecutionPlan) -> None:
        """Drive a single plan to completion."""
        try:
            # Validate the graph up-front so we fail fast on bad deps.
            bad = self._validate_graph(plan)
            if bad:
                plan.status = PlanStatus.FAILED
                self._emit_plan_completed(
                    plan,
                    status="failed",
                    summary=f"plan graph invalid: {bad}",
                )
                return

            # Round-robin until every task is terminal.
            while not plan.all_terminal():
                ready = self._ready_tasks(plan)
                if not ready:
                    # Nothing ready and not all terminal: either
                    # everything running or a dep is unsatisfiable.
                    # Sleep briefly and re-check.
                    await asyncio.sleep(0.05)
                    if plan.all_terminal():
                        break
                    if not any(
                        t.status in {TaskStatus.RUNNING, TaskStatus.READY}
                        for t in plan.tasks
                    ):
                        # Nothing in flight and nothing to run => stuck.
                        self._cascade_skip_dependents(plan)
                        break
                    continue

                await self._dispatch_batch(plan, ready)

            self._finalize(plan)

        except asyncio.CancelledError:
            plan.status = PlanStatus.CANCELLED
            for task in plan.tasks:
                if task.status in {TaskStatus.PENDING, TaskStatus.READY, TaskStatus.RUNNING}:
                    task.status = TaskStatus.CANCELLED
            self._emit_plan_completed(
                plan,
                status="cancelled",
                summary=_natural_summary(plan),
            )
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Plan %s crashed: %s", plan.id, exc)
            plan.status = PlanStatus.FAILED
            self._emit_plan_completed(
                plan,
                status="failed",
                summary=f"plan crashed: {exc}",
            )
        finally:
            self._cleanup(plan)

    def _ready_tasks(self, plan: ExecutionPlan) -> list[PlannedTask]:
        out: list[PlannedTask] = []
        for task in plan.tasks:
            if task.status != TaskStatus.PENDING:
                continue
            if plan.pending_deps_for(task) == 0:
                task.status = TaskStatus.READY
                out.append(task)
        return out

    async def _dispatch_batch(
        self, plan: ExecutionPlan, tasks: list[PlannedTask]
    ) -> None:
        """Dispatch a batch of ready tasks concurrently, bounded by
        ``max_concurrent``."""
        sem = asyncio.Semaphore(self._max_concurrent)

        async def _one(t: PlannedTask) -> None:
            async with sem:
                await self._run_one_task(plan, t)

        await asyncio.gather(*[_one(t) for t in tasks], return_exceptions=True)

    async def _run_one_task(self, plan: ExecutionPlan, task: PlannedTask) -> None:
        """Run one task with retry + replan logic."""
        task.status = TaskStatus.RUNNING
        max_attempts = 1 + (task.max_retries if task.retryable else 0)

        while task.attempts < max_attempts:
            task.attempts += 1
            try:
                outcome = await self._executor.run_task(plan, task)
            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
                raise
            except Exception as exc:  # noqa: BLE001
                task.error = f"dispatcher crashed: {exc}"
                logger.exception(
                    "Plan %s task %s dispatcher crashed",
                    plan.id,
                    task.id,
                )
                outcome = None  # signal "exception path"

            if outcome is not None and outcome.success:
                task.status = TaskStatus.COMPLETED
                task.result = outcome.result
                task.error = ""
                return

            err = (
                outcome.error
                if outcome is not None
                else task.error or "task failed"
            )
            task.error = err
            logger.warning(
                "Plan %s task %s failed (attempt %d/%d): %s",
                plan.id,
                task.id,
                task.attempts,
                max_attempts,
                err,
            )

        # Out of attempts. Mark failed and ask the Planner to replan.
        task.status = TaskStatus.FAILED
        self._cascade_skip_dependents(plan)
        self._bus.emit(
            PlanReplanRequested(
                plan_id=plan.id,
                failed_task_id=task.id,
                reason=task.error,
                session_id=plan.session_id,
            )
        )

    def _cascade_skip_dependents(self, plan: ExecutionPlan) -> None:
        """Mark every task that (transitively) depends on a failed task
        as :attr:`TaskStatus.SKIPPED`."""
        failed_ids = {
            t.id for t in plan.tasks if t.status == TaskStatus.FAILED
        }
        changed = True
        while changed:
            changed = False
            for t in plan.tasks:
                if t.status != TaskStatus.PENDING:
                    continue
                if any(dep in failed_ids for dep in t.depends_on):
                    t.status = TaskStatus.SKIPPED
                    failed_ids.add(t.id)
                    changed = True

    def _finalize(self, plan: ExecutionPlan) -> None:
        plan.completed_at = datetime.utcnow()
        if plan.any_failed():
            plan.status = PlanStatus.PARTIAL
            self._emit_plan_completed(
                plan,
                status="partial",
                summary=_natural_summary(plan),
            )
        else:
            plan.status = PlanStatus.COMPLETED
            self._emit_plan_completed(
                plan,
                status="completed",
                summary=_natural_summary(plan),
            )

    def _emit_plan_completed(
        self, plan: ExecutionPlan, status: str, summary: str
    ) -> None:
        task_results = [
            {
                "tool": t.tool,
                "result": (t.result or t.error or "").strip(),
                "arguments": dict(t.arguments or {}),
            }
            for t in plan.tasks
            if t.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED}
            if (t.result or t.error)
        ]
        self._bus.emit(
            PlanCompleted(
                plan_id=plan.id,
                status=status,
                summary=summary,
                task_results=task_results,
                user_request=plan.user_request,
                session_id=plan.session_id,
            )
        )

    def _cleanup(self, plan: ExecutionPlan) -> None:
        self._plan_tasks.pop(plan.id, None)
        if self._active_by_session.get(plan.session_id) == plan.id:
            self._active_by_session.pop(plan.session_id, None)
        # Drop the plan immediately in this context. Late TaskCompleted
        # events are correlated by the PlanExecutor (not by the
        # Scheduler), so we no longer need a deferred drop.
        self._plans.pop(plan.id, None)

    # ── helpers ───────────────────────────────────────────────────

    @staticmethod
    def _validate_graph(plan: ExecutionPlan) -> str:
        """Return a non-empty error string if the plan graph is invalid."""
        ids = {t.id for t in plan.tasks}
        for t in plan.tasks:
            for dep in t.depends_on:
                if dep not in ids:
                    return f"task {t.id} depends on unknown task {dep}"
        # Cycle detection (Kahn-style)
        in_degree: dict[str, int] = {t.id: 0 for t in plan.tasks}
        for t in plan.tasks:
            in_degree[t.id] = sum(
                1 for d in t.depends_on if d in in_degree
            )
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        visited = 0
        while queue:
            current = queue.pop()
            visited += 1
            for t in plan.tasks:
                if current in t.depends_on:
                    in_degree[t.id] -= 1
                    if in_degree[t.id] == 0:
                        queue.append(t.id)
        if visited != len(plan.tasks):
            return "dependency cycle detected"
        return ""

    @staticmethod
    def _deserialize_plan(payload: dict[str, Any]) -> ExecutionPlan | None:
        """Rebuild an :class:`ExecutionPlan` from a dict (event payload)."""
        try:
            tasks: list[PlannedTask] = []
            for raw in payload.get("tasks", []):
                tasks.append(
                    PlannedTask(
                        id=raw["id"],
                        description=raw.get("description", ""),
                        tool=raw["tool"],
                        arguments=dict(raw.get("arguments", {})),
                        depends_on=tuple(raw.get("depends_on", ())),
                        retryable=raw.get("retryable", True),
                        max_retries=raw.get("max_retries", 1),
                        status=TaskStatus(raw.get("status", TaskStatus.PENDING.value)),
                        output_refs=dict(raw.get("output_refs", {})),
                    )
                )
            return ExecutionPlan(
                id=payload["id"],
                user_request=payload.get("user_request", ""),
                tasks=tasks,
                status=PlanStatus(payload.get("status", PlanStatus.CREATED.value)),
                session_id=payload.get("session_id", "default"),
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Failed to deserialize plan: %s", exc)
            return None

    # Public hook for tests to inject a deterministic executor.
    @property
    def executor(self) -> PlanExecutor:
        return self._executor
