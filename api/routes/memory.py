"""Backs the MEMORY and (partly) HISTORY panels with real MemoryManager data."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/facts")
async def get_facts(request: Request) -> list[dict]:
    pipeline = request.app.state.pipeline
    return await pipeline.memory.get_all_facts()


@router.delete("/session")
async def clear_session(request: Request) -> dict:
    pipeline = request.app.state.pipeline
    cleared = await pipeline.memory.clear_session()
    return {"cleared": cleared}
