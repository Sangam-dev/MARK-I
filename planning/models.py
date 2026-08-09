"""Data models for the planning layer.

The classes here are intentionally dataclass-based (not Pydantic) so
they can be mutated by the in-memory :class:`PlanScheduler` after a
``PlanCreated`` event is deserialized (``BaseEvent`` is frozen, but the
plan lives on after the event is gone).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """Lifecycle of a single :class:`PlannedTask`."""

    PENDING = "pending"          # waiting for dependencies
    READY = "ready"              # deps satisfied, not yet dispatched
    RUNNING = "running"          # dispatched, awaiting TaskCompleted
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"          # a dependency failed; cascade-skipped
    CANCELLED = "cancelled"


class PlanStatus(str, Enum):
    """Lifecycle of an :class:`ExecutionPlan`."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"          # some tasks succeeded, some failed
    CANCELLED = "cancelled"


@dataclass(slots=True)
class PlannedTask:
    """A single atomic tool call inside an :class:`ExecutionPlan`.

    Attributes
    ----------
    id:
        Stable identifier inside a plan (``"t1"``, ``"t2"`` …). Referenced
        by :attr:`depends_on`.
    description:
        Human-readable summary; useful in logs and in the user-facing
        summary emitted on :class:`PlanCompleted`.
    tool:
        Tool name. Must exist in
        :data:`tasks.registry.TASK_REGISTRY`. The Planner validates this
        before emitting :class:`PlanCreated`.
    arguments:
        Parameters passed to the tool. The Executor binds any
        ``output_refs`` placeholders against prior task results before
        dispatching.
    depends_on:
        Tuple of task ids that must reach ``COMPLETED`` before this
        task becomes ``READY``. Empty tuple = root task.
    retryable:
        If ``False``, a failure short-circuits to replan on the first
        attempt. If ``True``, up to ``max_retries`` additional attempts
        are made before replanning.
    max_retries:
        Maximum number of *additional* attempts after the first failure.
        ``1`` (the default) means "retry once, then replan".
    status:
        Updated by the Scheduler as the task progresses.
    result:
        Populated from :class:`TaskCompleted.result` on success.
    error:
        Populated from :class:`TaskCompleted.error` on failure.
    attempts:
        Number of times the Executor has dispatched this task. Bounded
        by ``1 + max_retries``.
    output_refs:
        Mapping of ``arg_name -> "task_id:output_key"`` for arguments
        that must be resolved against a prior task's result. Set by the
        :class:`ReferenceResolver` and the LLM.
    """

    id: str
    description: str
    tool: str
    arguments: dict[str, Any]
    depends_on: tuple[str, ...] = ()
    retryable: bool = True
    max_retries: int = 1
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    error: str = ""
    attempts: int = 0
    output_refs: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionPlan:
    """A directed acyclic graph of :class:`PlannedTask` items.

    The graph is implicit: each :class:`PlannedTask` carries its
    ``depends_on`` list. The Scheduler walks it as tasks complete.
    """

    id: str
    user_request: str
    tasks: list[PlannedTask]
    status: PlanStatus = PlanStatus.CREATED
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    references_resolved: bool = False
    session_id: str = "default"

    # ── convenience helpers used by the Scheduler ──────────────────────

    def task_by_id(self, task_id: str) -> PlannedTask | None:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    def pending_deps_for(self, task: PlannedTask) -> int:
        """Count dependencies of *task* that are not yet COMPLETED."""
        if not task.depends_on:
            return 0
        unresolved = 0
        for dep_id in task.depends_on:
            dep = self.task_by_id(dep_id)
            if dep is None:
                # unknown dependency — treat as unresolved; Scheduler will
                # fail-fast on it rather than silently skipping.
                unresolved += 1
                continue
            if dep.status not in {TaskStatus.COMPLETED}:
                unresolved += 1
        return unresolved

    def all_terminal(self) -> bool:
        """True if every task has reached a terminal status."""
        return all(
            t.status in {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.SKIPPED,
                TaskStatus.CANCELLED,
            }
            for t in self.tasks
        )

    def any_failed(self) -> bool:
        return any(
            t.status in {TaskStatus.FAILED, TaskStatus.SKIPPED}
            for t in self.tasks
        )

    def summary(self) -> str:
        """One-line human summary used in PlanCompleted and ResponseReady."""
        done = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)
        skipped = sum(1 for t in self.tasks if t.status == TaskStatus.SKIPPED)
        total = len(self.tasks)
        parts = [f"{done}/{total} tasks completed"]
        if failed:
            parts.append(f"{failed} failed")
        if skipped:
            parts.append(f"{skipped} skipped")
        return ", ".join(parts)
