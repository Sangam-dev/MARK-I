"""Backs the HISTORY panel with the in-memory short-term conversation buffer.

Note: this is the same in-process buffer ReasoningCoordinator reads for LLM
context (memory.short_term) — it resets on backend restart, same as before
this integration. Durable facts (not full history) persist across restarts
via /api/memory/facts.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

router = APIRouter(tags=["history"])


@router.get("/history")
async def get_history(
    request: Request, limit: int = Query(default=20, ge=1, le=200)
) -> list[dict]:
    pipeline = request.app.state.pipeline
    return pipeline.memory.short_term.get_recent(limit)
