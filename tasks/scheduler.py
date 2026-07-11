from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from core.bus import EventBus
from core.events import ResponseReady
from memory.structured import StructuredMemory

logger = logging.getLogger("kancha.tasks.scheduler")


class TaskScheduler:
    """asyncio-based reminder and scheduled task system."""

    def __init__(self, bus: EventBus, structured_memory: StructuredMemory) -> None:
        self.bus = bus
        self.memory = structured_memory
        self._pending_tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Load pending tasks from SQLite and reschedule them."""
        await self._reschedule_on_startup()

    async def schedule_reminder(
        self,
        description: str,
        delay_seconds: float,
        session_id: str,
        task_id: str | None = None,
    ) -> str:
        """Schedule a new reminder and persist it to database."""
        async with self._lock:
            tid = task_id or str(uuid.uuid4())
            due_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
            due_at_str = due_at.isoformat()

            # Store in SQLite with status='pending'
            await self.memory.store_task(
                description=description,
                due_at=due_at_str,
                session_id=session_id,
                task_id=tid,
                metadata={"type": "reminder"},
            )

            # Schedule the task
            task = asyncio.create_task(
                self._fire_reminder(tid, description, delay_seconds, session_id)
            )
            self._pending_tasks[tid] = task
            logger.info("Scheduled reminder %s in %.1fs", tid, delay_seconds)
            return tid

    async def _fire_reminder(
        self, task_id: str, description: str, delay_seconds: float, session_id: str
    ) -> None:
        """Sleep for the specified delay, then fire the reminder."""
        try:
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

            # Update status in SQLite
            await self.memory.update_task_status(task_id, "fired")

            # Emit ResponseReady
            logger.info("Firing reminder %s: %s", task_id, description)
            self.bus.emit(
                ResponseReady(
                    text=f"⏰ Reminder: {description}",
                    session_id=session_id,
                )
            )
        except asyncio.CancelledError:
            logger.info("Reminder %s was cancelled", task_id)
            raise
        except Exception as e:
            logger.exception("Error firing reminder %s: %s", task_id, e)
        finally:
            async with self._lock:
                self._pending_tasks.pop(task_id, None)

    async def cancel_reminder(self, task_id: str) -> bool:
        """Cancel a pending reminder."""
        async with self._lock:
            task = self._pending_tasks.pop(task_id, None)
            if task:
                task.cancel()

            # Update SQLite status to cancelled
            updated = await self.memory.update_task_status(task_id, "cancelled")
            logger.info("Cancelled reminder %s (updated in DB: %s)", task_id, updated)
            return updated or (task is not None)

    async def get_pending_reminders(self, session_id: str) -> list[dict[str, Any]]:
        """Get list of pending reminders/tasks for a session."""
        return await self.memory.get_pending_tasks(session_id)

    async def _reschedule_on_startup(self) -> None:
        """Load pending tasks from DB, fire those still in future, or fire immediately if passed."""
        async with self._lock:
            pending = await self.memory.get_all_pending_tasks()
            logger.info("Found %d pending tasks to reschedule", len(pending))
            for t in pending:
                tid = t["id"]
                description = t["description"]
                due_at_str = t["due_at"]
                session_id = t["session_id"]

                if not due_at_str:
                    continue

                try:
                    due_at = datetime.fromisoformat(due_at_str)
                    now = datetime.utcnow()
                    delay = (due_at - now).total_seconds()
                    if delay < 0:
                        delay = 0.0

                    task = asyncio.create_task(
                        self._fire_reminder(tid, description, delay, session_id)
                    )
                    self._pending_tasks[tid] = task
                    logger.info("Rescheduled reminder %s with delay %.1fs", tid, delay)
                except Exception as e:
                    logger.exception("Failed to reschedule task %s: %s", tid, e)
