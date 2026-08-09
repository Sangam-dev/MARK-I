"""Tracks connected WebSocket clients and broadcasts JSON messages to them.

v1 assumes a single Electron window (one client), but broadcasting to a set
is trivially correct for that case and needs no changes to support more
connections later (e.g. a companion mobile app).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger("kancha.api.ws_manager")


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        logger.info("WebSocket connected (%d total)", len(self._connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)
        logger.info("WebSocket disconnected (%d total)", len(self._connections))

    async def broadcast(
        self, message_type: str, payload: dict[str, Any], session_id: str = "default"
    ) -> None:
        """Send `{type, payload, session_id}` to every connected client."""
        envelope = {"type": message_type, "payload": payload, "session_id": session_id}

        async with self._lock:
            connections = list(self._connections)

        dead: list[WebSocket] = []
        for ws in connections:
            try:
                await ws.send_json(envelope)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.discard(ws)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


# Module-level singleton — one process, one manager, shared by server.py and
# bridge.py without needing to thread it through FastAPI dependency injection.
manager = ConnectionManager()
