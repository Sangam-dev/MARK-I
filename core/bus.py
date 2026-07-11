from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine, Type, TypeVar

from .events import BaseEvent, SystemError

logger = logging.getLogger("kancha.bus")

E = TypeVar("E", bound=BaseEvent)
Handler = Callable[[Any], Coroutine[Any, Any, None]]

def subscribe(event_type: Type[E]) -> Callable[[Handler], Handler]:
    """Decorator to mark a method as a subscriber for a specific event type."""
    def decorator(func: Handler) -> Handler:
        if not hasattr(func, "_subscribed_events"):
            func._subscribed_events = []
        func._subscribed_events.append(event_type)
        return func
    return decorator

class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[type, list[Handler]] = defaultdict(list)
        self._tasks: set[asyncio.Task] = set()
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def subscribe(self, event_type: Type[E], handler: Handler) -> None:
        if not asyncio.iscoroutinefunction(handler):
            raise TypeError(
                f"{handler.__name__} must be async. Got: {type(handler)}"
            )
        self._handlers[event_type].append(handler)
        logger.debug("Subscribed %s → %s", event_type.__name__, handler.__name__)

    def unsubscribe(self, event_type: Type[E], handler: Handler) -> None:
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def handler_count(self, event_type: type) -> int:
        return len(self._handlers.get(event_type, []))

    async def _run_handler(self, handler: Handler, event: BaseEvent) -> None:
        try:
            await handler(event)
        except Exception as exc:
            logger.exception(
                "Handler %s crashed on %s: %s",
                handler.__name__, type(event).__name__, exc
            )
            if SystemError in self._handlers:
                self.emit(SystemError(
                    source_module=handler.__name__,
                    error_message=str(exc),
                    recoverable=True,
                ))

    def emit(self, event: BaseEvent) -> None:
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

        event_type = type(event)
        handlers = self._handlers.get(event_type, [])

        if not handlers:
            logger.debug("No handlers for %s", event_type.__name__)
            return

        logger.debug(
            "Emitting %s [%s] → %d handler(s)",
            event_type.__name__, event.event_id[:8], len(handlers)
        )

        for handler in handlers:
            task = asyncio.create_task(
                self._run_handler(handler, event),
                name=f"{handler.__name__}:{event.event_id[:8]}"
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    def emit_threadsafe(self, event: BaseEvent) -> None:
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                raise RuntimeError("EventBus loop not initialized and no running loop found in thread.")
        self._loop.call_soon_threadsafe(self.emit, event)

    async def emit_and_wait(self, event: BaseEvent) -> None:
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
        handlers = self._handlers.get(type(event), [])
        if not handlers:
            return
        await asyncio.gather(
            *[self._run_handler(h, event) for h in handlers],
            return_exceptions=True
        )

    async def drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def close(self) -> None:
        for task in list(self._tasks):
            if not task.done():
                task.cancel()
        await self.drain()
        self._handlers.clear()

    def register_handlers(self, obj: Any) -> None:
        for attr_name in dir(obj):
            try:
                attr = getattr(obj, attr_name)
            except AttributeError:
                continue
            if hasattr(attr, "_subscribed_events"):
                for event_type in attr._subscribed_events:
                    self.subscribe(event_type, attr)

    def __repr__(self) -> str:
        total = sum(len(v) for v in self._handlers.values())
        return f"EventBus(types={len(self._handlers)}, handlers={total})"