from __future__ import annotations

import ctypes
import platform
import subprocess
from dataclasses import dataclass
from shutil import which


@dataclass(slots=True)
class PowerResult:
    success: bool
    message: str


def _run_power_command(command: list[str], success_message: str, action: str) -> PowerResult:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError as exc:
        return PowerResult(False, f"The operating system refused the {action} command: {exc}.")
    except subprocess.TimeoutExpired:
        return PowerResult(True, success_message)

    if completed.returncode == 0:
        return PowerResult(True, success_message)

    error = (completed.stderr or completed.stdout or "No details returned.").strip()
    return PowerResult(False, f"The operating system refused the {action} command: {error}")


def _run_first_available(
    commands: tuple[tuple[str, ...], ...],
    success_message: str,
    action: str,
) -> PowerResult:
    for command in commands:
        executable = which(command[0])
        if executable:
            return _run_power_command([executable, *command[1:]], success_message, action)

    command_names = ", ".join(command[0] for command in commands)
    return PowerResult(False, f"I couldn't find a supported {action} command. Tried: {command_names}.")


def shutdown() -> PowerResult:
    system = platform.system().lower()
    if system == "linux":
        return _run_first_available(
            (
                ("systemctl", "poweroff"),
                ("shutdown", "now"),
                ("poweroff",),
            ),
            "Shutting down now.",
            "shutdown",
        )
    if system == "darwin":
        return _run_power_command(["shutdown", "-h", "now"], "Shutting down now.", "shutdown")

    return _run_power_command(["shutdown", "/s", "/t", "1"], "Shutting down now.", "shutdown")


def restart() -> PowerResult:
    system = platform.system().lower()
    if system == "linux":
        return _run_first_available(
            (
                ("systemctl", "reboot"),
                ("shutdown", "-r", "now"),
                ("reboot",),
            ),
            "Restarting now.",
            "restart",
        )
    if system == "darwin":
        return _run_power_command(["shutdown", "-r", "now"], "Restarting now.", "restart")

    return _run_power_command(["shutdown", "/r", "/t", "1"], "Restarting now.", "restart")


def sleep() -> PowerResult:
    system = platform.system().lower()
    if system == "linux":
        return _run_first_available(
            (
                ("systemctl", "suspend"),
                ("loginctl", "suspend"),
                ("pm-suspend",),
            ),
            "Putting this device to sleep.",
            "sleep",
        )
    if system == "darwin":
        return _run_power_command(["pmset", "sleepnow"], "Putting this device to sleep.", "sleep")

    try:
        result = ctypes.windll.powrprof.SetSuspendState(False, True, False)
        if result:
            return PowerResult(True, "Putting this device to sleep.")
        return PowerResult(False, "The operating system refused the sleep command.")
    except AttributeError:
        return _run_power_command(
            ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
            "Putting this device to sleep.",
            "sleep",
        )
    except OSError as exc:
        return PowerResult(False, f"The operating system refused the sleep command: {exc}.")
