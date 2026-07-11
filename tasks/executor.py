from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

from actions.alarms import cancel_alarms, list_alarms, set_alarm
from actions.apps import open_app
from actions.file_controller import file_controller
from actions.power import restart, shutdown, sleep
from actions.weather import get_weather
from core.bus import EventBus
from core.events import TaskCompleted, TaskExecutionRequested
from tasks.registry import TASK_REGISTRY, validate_task

logger = logging.getLogger("kancha.tasks.executor")

_FILE_FAILURE_RE = re.compile(
    r"^(?:access denied|path not found|not a directory|permission denied|"
    r"could not|not found|source not found|no destination specified|"
    r"file not found|not a file|search path not found|search error|"
    r"error|unknown action|file controller error|protected directory)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class TaskExecutionResult:
    success: bool
    message: str


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
                session_id=event.session_id,
            )
        )

    async def _dispatch(
        self, task_name: str, params: dict[str, Any]
    ) -> TaskExecutionResult:
        """Route task to the appropriate action function (all blocking calls run in thread)."""

        if task_name == "open_app":
            app_name = params.get("app_name", "")
            result = await asyncio.to_thread(open_app, app_name)
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

        if task_name == "sleep":
            result = await asyncio.to_thread(sleep)
            return TaskExecutionResult(result.success, result.message)

        if task_name == "shutdown":
            result = await asyncio.to_thread(shutdown)
            return TaskExecutionResult(result.success, result.message)

        if task_name == "restart":
            result = await asyncio.to_thread(restart)
            return TaskExecutionResult(result.success, result.message)

        if task_name == "file_operation":
            result_text = await asyncio.to_thread(file_controller, params)
            return TaskExecutionResult(
                not _FILE_FAILURE_RE.match(result_text.strip()),
                result_text,
            )

        return TaskExecutionResult(False, f"No handler found for task: {task_name}")
