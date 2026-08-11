from __future__ import annotations

import re
from dataclasses import dataclass

from .alarms import cancel_alarms, list_alarms, set_alarm
from .apps import open_app
from .weather import get_weather


@dataclass(slots=True)
class DeviceCommandResult:
    handled: bool
    message: str = ""


_OPEN_RE = re.compile(
    r"^\s*(?:open|launch|start|run)\s+(?:the\s+)?(?P<app>[\w .+-]+?)\s*$",
    re.IGNORECASE,
)

_ALARM_RE = re.compile(
    r"\b(?:set\s+)?(?:an?\s+)?(?:alarm|timer|reminder|remainder)\b",
    re.IGNORECASE,
)

_WEATHER_RE = re.compile(
    r"\b(?:weather|forecast|temperature|raining)\b",
    re.IGNORECASE,
)


def handle_device_command(user_input: str) -> DeviceCommandResult:
    text = " ".join(user_input.strip().lower().split())
    text = text.rstrip(".!?")
    if not text:
        return DeviceCommandResult(False)

    open_match = _OPEN_RE.match(user_input)
    if open_match:
        result = open_app(open_match.group("app"))
        return DeviceCommandResult(True, result.message)

    if text.startswith((
        "set alarm",
        "set an alarm",
        "set a alarm",
        "an alarm",
        "a alarm",
        "alarm",
        "set timer",
        "set a timer",
        "set an timer",
        "a timer",
        "timer",
        "wake me",
        "remind me",
        "remainder",
    )) or (_ALARM_RE.search(text) and re.search(r"\b(?:in|for|after|at)\b", text)):
        result = set_alarm(user_input)
        return DeviceCommandResult(True, result.message)

    if _WEATHER_RE.search(text):
        result = get_weather(user_input)
        return DeviceCommandResult(True, result.message)

    if text in {"alarms", "list alarms", "show alarms"}:
        result = list_alarms()
        return DeviceCommandResult(True, result.message)

    if text in {"cancel alarms", "clear alarms", "delete alarms", "stop alarms"}:
        result = cancel_alarms()
        return DeviceCommandResult(True, result.message)

    if text in {
        "shutdown",
        "shut down",
        "shut down now",
        "restart",
        "restart now",
        "reboot",
        "reboot now",
        "sleep",
        "sleep now",
        "suspend",
        "power off",
        "turn off my computer",
    }:
        # Power-state control was removed from the assistant. This branch
        # answers instead of acting — the sleep()/shutdown()/restart()
        # calls that used to live here are gone, along with actions/power.py.
        return DeviceCommandResult(
            True,
            "I can't change this machine's power state.",
        )

    return DeviceCommandResult(False)
