"""GET /api/health — used by the Electron main process to know the backend is
ready before loading the renderer, and by the DEVELOPER panel in the UI.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from api.ws_manager import manager

router = APIRouter(tags=["health"])

_START_TIME = time.monotonic()


@router.get("/health")
async def health(request: Request) -> dict:
    pipeline = request.app.state.pipeline
    llm_ok = await pipeline.llm.health_check()
    return {
        "status": "ok",
        "session_id": pipeline.session_id,
        "tts_enabled": pipeline.tts_enabled,
        "voice_available": getattr(request.app.state, "voice_available", False),
        "llm_available": llm_ok,
        "ws_clients": manager.connection_count,
        "uptime_seconds": round(time.monotonic() - _START_TIME, 1),
    }
