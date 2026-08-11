"""Planning layer for MARK-I.

Decomposes a single user request into an ``ExecutionPlan`` of atomic
``PlannedTask`` items, schedules them according to explicit dependencies,
and dispatches them through the existing ``TaskExecutionRequested`` bus
event so the leaf tool layer (``tasks/executor.py``) stays untouched.

Work only ever enters this layer as a ``TaskRequested`` event emitted by
the Conversation LLM (``reasoning/coordinator.py``) — never as raw user
input. The flow is::

    TaskRequested  (from the Conversation LLM)
        -> Planner  (LLM decomposition + reference resolution)
        -> emits PlanCreated
        -> PlanScheduler (asyncio graph runner, dependency tracking,
                          retries, replan)
        -> PlanExecutor (validates + emits TaskExecutionRequested
                          with _plan_id/_task_id correlation)
        -> existing TaskExecutor runs the tool
        -> TaskCompleted returned to PlanScheduler
        -> PlanCompleted (or PlanReplanRequested on permanent failure)
        -> Planner correlates it back to the originating TaskRequested
        -> TaskResultReady (returned to the Conversation LLM, which
                            writes the user-facing response)
"""

from planning.models import (
    ExecutionPlan,
    PlanStatus,
    PlannedTask,
    TaskStatus,
)

__all__ = [
    "ExecutionPlan",
    "PlanStatus",
    "PlannedTask",
    "TaskStatus",
]
