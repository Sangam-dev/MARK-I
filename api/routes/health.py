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

    # RAG is optional; report it as unavailable rather than failing the
    # whole health check when it is off or broken.
    rag_manager = getattr(pipeline, "rag", None)
    rag_ok = False
    rag_documents = 0
    if rag_manager is not None:
        try:
            rag_ok = await rag_manager.health_check()
            rag_documents = len(await rag_manager.list_documents())
        except Exception:  # noqa: BLE001
            rag_ok = False

    return {
        "status": "ok",
        "session_id": pipeline.session_id,
        "tts_enabled": pipeline.tts_enabled,
        "voice_available": getattr(request.app.state, "voice_available", False),
        "llm_available": llm_ok,
        "rag_enabled": getattr(pipeline, "rag_enabled", False),
        "rag_available": rag_ok,
        "rag_documents": rag_documents,
        "ws_clients": manager.connection_count,
        "uptime_seconds": round(time.monotonic() - _START_TIME, 1),
    }
