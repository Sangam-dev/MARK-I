from __future__ import annotations

import logging
from typing import Any

from core.bus import EventBus
from core.events import IntentIdentified, MemoryRetrieved, ReasoningRequested
from memory.manager import MemoryManager

logger = logging.getLogger("kancha.reasoning.rag")


class RAGPipeline:
    """
    Disabled compatibility shim.

    RAG/vector retrieval is intentionally turned off. If older code still
    registers this pipeline, it forwards intents to reasoning with only
    structured user facts and no episodic/vector context.
    """

    def __init__(
        self, memory_manager: MemoryManager, bus: EventBus, config: Any = None
    ) -> None:
        self.memory_manager = memory_manager
        self.bus = bus
        self.config = config

    def register(self) -> None:
        """Subscribe to IntentIdentified events without enabling RAG."""
        self.bus.subscribe(IntentIdentified, self.on_intent_identified)
        logger.info("RAGPipeline is disabled; forwarding structured facts only")

    async def retrieve(self, query: str, session_id: str) -> MemoryRetrieved:
        """Return durable facts only; vector/episodic context stays empty."""
        facts = await self.memory_manager.get_all_facts()
        return MemoryRetrieved(
            session_id=session_id,
            query=query,
            structured_context=facts,
            episodic_context=[],
        )

    async def on_intent_identified(self, event: IntentIdentified) -> None:
        """Handle IntentIdentified and emit ReasoningRequested without RAG."""
        retrieved = await self.retrieve(event.raw_input, event.session_id)
        self.bus.emit(
            ReasoningRequested(
                session_id=event.session_id,
                intent_event=event,
                memory_events=[retrieved],
            )
        )
