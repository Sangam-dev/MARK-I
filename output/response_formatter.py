from __future__ import annotations

import logging

from core.bus import EventBus
from core.events import ResponseReady

logger = logging.getLogger("kancha.output.formatter")


class ResponseFormatter:
    """Prints assistant responses to the console.

    Only subscribes to ResponseReady — TaskCompleted is handled exclusively
    by the ReasoningCoordinator which formats it via LLM before emitting
    ResponseReady. Subscribing here too would cause double output.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    def register(self) -> None:
        self._bus.subscribe(ResponseReady, self.on_response_ready)

    async def on_response_ready(self, event: ResponseReady) -> None:
        if event.text:
            print(f"\nKANCHA: {event.text}\n")
            logger.debug("Response printed (%d chars)", len(event.text))
