"""Background loop that drives ``actions.system_monitor.SystemMonitor``.

Owns a single asyncio task that wakes every ``interval_s`` seconds, calls
``monitor.check()`` on a worker thread (``psutil`` calls are blocking), and
emits ``SystemMonitorAlert`` on the EventBus whenever the check returns a
non-empty ``[SYSTEM_ALERT] …`` string.

Subscribes to **nothing** — the loop is purely an emitter. The hop from
``SystemMonitorAlert`` to ``ResponseReady`` (which TTS and the console
formatter already subscribe to) is wired inside ``build_pipeline()`` as a
one-line handler so this module never grows bus-wiring responsibilities.

Lifecycle:
- Created and started in :func:`core.pipeline.build_pipeline`.
- Stopped in :func:`core.pipeline.shutdown_pipeline` via the ``stop()``
  coroutine, which cancels the running task and waits briefly for clean
  shutdown. Mirrors the ``input_tasks`` cleanup pattern in ``main.py``.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from core.events import SystemMonitorAlert

if TYPE_CHECKING:                               # pragma: no cover
    from core.bus import EventBus
    from actions.system_monitor import SystemMonitor

logger = logging.getLogger("kancha.core.system_monitor_loop")


class SystemMonitorLoop:
    """Periodically run a SystemMonitor and surface alerts on the bus."""

    def __init__(
        self,
        bus: "EventBus",
        monitor: "SystemMonitor",
        interval_s: float = 30.0,
        enabled: bool = True,
    ) -> None:
        self._bus = bus
        self._monitor = monitor
        self._interval_s = max(1.0, float(interval_s))
        self._enabled = bool(enabled)
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Toggle the loop at runtime. Does not start/stop the task itself —
        the loop checks ``self._enabled`` each tick, so callers don't need
        to restart anything.
        """
        self._enabled = bool(enabled)
        logger.info("SystemMonitorLoop enabled=%s", self._enabled)

    def start(self) -> None:
        """Spawn the background asyncio task. Idempotent."""
        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self.run(), name="system_monitor_loop")
        logger.info(
            "SystemMonitorLoop started  (interval=%.1fs, enabled=%s)",
            self._interval_s,
            self._enabled,
        )

    async def stop(self) -> None:
        """Cancel the background task and wait up to 2s for clean shutdown."""
        if self._task is None:
            return
        self._stop_event.set()
        if not self._task.done():
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                # Cancellation is the normal path; timeout means the tick is
                # blocked on psutil — drop it, the process is exiting anyway.
                pass
        self._task = None
        logger.info("SystemMonitorLoop stopped")

    async def run(self) -> None:
        """Tick forever until ``stop()`` is called.

        Each iteration:
        1. Sleep ``interval_s`` (or until ``_stop_event`` is set).
        2. If disabled, skip the check.
        3. Run ``monitor.check()`` on a worker thread.
        4. Emit ``SystemMonitorAlert`` when it returns a non-empty string.
        """
        try:
            while not self._stop_event.is_set():
                # Wait for the next tick OR for shutdown — whichever first.
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self._interval_s
                    )
                    # If wait_for returned without timeout, the stop event fired.
                    break
                except asyncio.TimeoutError:
                    pass  # normal tick boundary

                if not self._enabled:
                    continue

                try:
                    text = await asyncio.to_thread(self._monitor.check)
                except Exception as exc:
                    logger.exception("SystemMonitor check crashed: %s", exc)
                    continue

                if not text:
                    continue

                logger.info("System monitor alert: %s", text)
                try:
                    self._bus.emit(SystemMonitorAlert(text=text))
                except Exception as exc:
                    logger.exception("Failed to emit SystemMonitorAlert: %s", exc)
        except asyncio.CancelledError:
            # Raised by self._task.cancel() — propagate so the task transitions
            # to done and ``stop()`` can finish.
            raise
        except Exception as exc:
            logger.exception("SystemMonitorLoop crashed: %s", exc)
