"""Backs the FILES panel with actions/file_controller.py (same code path
TaskExecutor uses for the `file_operation` task type)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query

from actions.file_controller import file_controller

router = APIRouter(prefix="/files", tags=["files"])


@router.get("")
async def list_files(path: str = Query(default="desktop")) -> dict:
    message = await asyncio.to_thread(file_controller, {"action": "list", "path": path})
    return {"path": path, "message": message}


@router.get("/disk-usage")
async def disk_usage(path: str = Query(default="home")) -> dict:
    message = await asyncio.to_thread(
        file_controller, {"action": "disk_usage", "path": path}
    )
    return {"path": path, "message": message}
