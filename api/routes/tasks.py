"""Backs the AUTOMATION panel's alarm/reminder/timer list with actions/alarms.py.

Calls the same synchronous functions the TaskExecutor dispatches to for the
`set_alarm` / `list_alarms` / `cancel_alarms` task types, so behavior is
identical whether triggered by voice/text ("set an alarm for 5 minutes") or
by a REST call from the UI.
"""

from __future__ import annotations

from fastapi import APIRouter

from actions.alarms import cancel_alarms, list_alarms, set_alarm
from api.schemas import ActionResult, AlarmCreateRequest

router = APIRouter(prefix="/alarms", tags=["alarms"])


@router.get("", response_model=ActionResult)
async def get_alarms() -> ActionResult:
    result = list_alarms()
    return ActionResult(success=result.success, message=result.message)


@router.post("", response_model=ActionResult)
async def create_alarm(body: AlarmCreateRequest) -> ActionResult:
    result = set_alarm(body.command, body.delay_seconds)
    return ActionResult(success=result.success, message=result.message)


@router.delete("", response_model=ActionResult)
async def clear_alarms() -> ActionResult:
    result = cancel_alarms()
    return ActionResult(success=result.success, message=result.message)
