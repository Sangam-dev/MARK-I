"""Exposes actions/weather.py over REST (same code path as the `get_weather`
task type) for a future WEATHER widget."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query

from actions.weather import get_weather

router = APIRouter(prefix="/weather", tags=["weather"])


@router.get("")
async def weather(
    city: str = Query(...), date: str | None = None, units: str | None = None
) -> dict:
    result = await asyncio.to_thread(get_weather, city, date, units)
    return {"success": result.success, "message": result.message}
