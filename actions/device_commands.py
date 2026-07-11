from __future__ import annotations

import re
from dataclasses import dataclass

from .alarms import cancel_alarms, list_alarms, set_alarm
from .apps import open_app
from .power import restart, shutdown, sleep
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
        "sleep now",
        "suspend now",
        "go to sleep",
        "put pc to sleep",
        "put my pc to sleep",
        "put computer to sleep",
        "put my computer to sleep",
        "put this device to sleep",
        "sleep the computer now",
    }:
        result = sleep()
        return DeviceCommandResult(True, result.message)

    if text in {
        "shut down now",
        "shut down the computer now",
        "shut down my computer now",
        "shut down this device",
        "turn off now",
        "turn off the computer now",
        "turn off my computer",
        "power off now",
        "power off this device",
    }:
        result = shutdown()
        return DeviceCommandResult(True, result.message)

    if text in {
        "restart the computer now",
        "restart my computer now",
        "restart this device now",
        "reboot now",
        "reboot the computer now",
        "reboot my device",
        "reboot this device now",
    }:
        result = restart()
        return DeviceCommandResult(True, result.message)

    if text in {"shutdown", "shut down", "restart", "reboot", "sleep", "suspend", "power off"}:
        return DeviceCommandResult(
            True,
            "For power actions, say the full command with now, like shutdown now or restart now.",
        )

    return DeviceCommandResult(False)
