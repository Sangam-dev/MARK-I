from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("kancha.tasks.registry")


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    description: str
    required_params: tuple[str, ...]
    optional_params: tuple[str, ...]
    requires_confirmation: bool = False
    is_destructive: bool = False
    param_types: dict[str, type] = field(default_factory=dict)


TASK_REGISTRY: dict[str, TaskSpec] = {
    "open_app": TaskSpec(
        name="open_app",
        description="Open an installed application by name.",
        required_params=("app_name",),
        optional_params=(),
        param_types={"app_name": str},
    ),
    "set_alarm": TaskSpec(
        name="set_alarm",
        description="Set an alarm, timer, or reminder from a natural language command.",
        required_params=("description", "delay_seconds"),
        optional_params=(),
        param_types={"description": str, "delay_seconds": int},
    ),
    "list_alarms": TaskSpec(
        name="list_alarms",
        description="List scheduled alarms and reminders.",
        required_params=(),
        optional_params=(),
        param_types={},
    ),
    "cancel_alarms": TaskSpec(
        name="cancel_alarms",
        description="Cancel all scheduled alarms and reminders.",
        required_params=(),
        optional_params=(),
        param_types={},
    ),
    "get_weather": TaskSpec(
        name="get_weather",
        description="Get weather information for a place.",
        required_params=("city",),
        optional_params=("date", "units"),
        param_types={"city": str, "date": str, "units": str},
    ),
    # Power-state tasks — sleep, shutdown, restart — have been REMOVED on
    # purpose, not merely gated. The assistant has no route to power the
    # machine off, restart it, or suspend it: no entry here, no dispatch
    # branch in tasks/executor.py, no action in actions/system_tool.py.
    #
    # This registry is the catalog rendered into *both* LLM prompts
    # (planning/prompts.py for the Task LLM, reasoning/coordinator.py for
    # the Conversation LLM), so removing the entry removes the capability
    # from what either model believes it can do. Re-adding one here is
    # enough to hand the assistant the power button again — don't.
    #
    # Enforced by tests/test_system_tool.py::test_registry_has_no_power_tasks.
    "file_operation": TaskSpec(
        name="file_operation",
        description=(
            "Perform file system operations: list, create_file, create_folder, delete, "
            "move, copy, rename, read, write, find, largest, disk_usage, organize_desktop, info."
        ),
        required_params=("action",),
        optional_params=(
            "path",
            "name",
            "content",
            "destination",
            "new_name",
            "extension",
            "max_results",
            "count",
            "append",
        ),
        param_types={"action": str, "path": str, "name": str},
    ),
    "execute_protocol": TaskSpec(
        name="execute_protocol",
        description="Execute a predefined protocol script (e.g., genesis protocol).",
        required_params=("protocol_name",),
        optional_params=("original_request",),
        param_types={"protocol_name": str, "original_request": str},
    ),
    "desktop_control": TaskSpec(
        name="desktop_control",
        description=(
            "Control the Linux desktop: wallpaper (set/get/from-url), window management "
            "(list_windows/focus/close/minimize/maximize), virtual desktops "
            "(list_workspaces/switch_workspace/move_to_workspace/window_workspace), "
            "desktop file management (organize/clean/list/stats), or run an AI-driven "
            "sandboxed desktop task."
        ),
        required_params=(),  # action is optional; falls back to natural-language task
        optional_params=(
            "action",
            "path",
            "url",
            "app",
            "target",
            "workspace",
            "mode",
            "follow",
            "task",
            "description",
            "confirm",
        ),
        param_types={
            "action": str,
            "path": str,
            "url": str,
            "app": str,
            "target": str,
            "workspace": str,
            "mode": str,
            "follow": bool,
            "task": str,
            "description": str,
            "confirm": bool,
        },
        # Wallpaper from URL, AI-driven sandboxed exec, and window actions
        # could all benefit from a confirmation gate — left as opt-in via
        # the ``confirm`` param so callers can decide per-request.
    ),
    "web_search": TaskSpec(
        name="web_search",
        description="Search the web for current information: news, prices, scores, events, documentation, research, and anything that requires live data.",
        required_params=("query",),
        optional_params=(),
        param_types={"query": str},
    ),
    "system": TaskSpec(
        name="system",
        description=(
            "Control the Linux system through an allowlist of structured "
            "actions: open_application (target=app), open_path (target=file "
            "or folder), wifi (operation=on|off|status), cpu, memory, disk "
            "(target=mount point), processes (operation=cpu|memory, "
            "limit=1..50), kill_process (pid or name), lock_screen, "
            "system_info, service (name, operation=status|start|stop|restart, "
            "scope=user|system). There is NO shutdown, reboot, restart or "
            "sleep action — this assistant cannot change the machine's power "
            "state at all. Destructive actions (kill_process, and "
            "start/stop/restart of a service) ALWAYS refuse the first time "
            "and report that approval is needed: ask the user, and only "
            "after they agree, repeat the same request with confirm=true. "
            "Setting confirm=true up front does nothing — the refusal is "
            "enforced by the tool, not by you."
        ),
        required_params=("action",),
        optional_params=(
            "operation",
            "target",
            "name",
            "pid",
            "scope",
            "limit",
            "confirm",
        ),
        # NOTE: deliberately not `requires_confirmation=True`. That flag
        # blocks a task wholesale in tasks/executor.py, which would take
        # "what's my disk usage" down with "reboot". The confirmation gate
        # lives per-action inside actions/system_tool.py instead.
        param_types={
            "action": str,
            "operation": str,
            "target": str,
            "name": str,
            "pid": int,
            "scope": str,
            "limit": int,
            "confirm": bool,
        },
    ),
    "system_monitor": TaskSpec(
        name="system_monitor",
        description=(
            "Inspect or tune the background system monitor: snapshot live "
            "CPU/RAM/temp/GPU/uptime (action='status'), run a single "
            "threshold check (action='check_alerts'), change a metric's alert "
            "threshold (action='set_threshold', needs metric+threshold), or "
            "toggle the background alert loop (action='enable'/'disable', "
            "needs enabled=true|false)."
        ),
        required_params=("action",),
        optional_params=("metric", "threshold", "enabled"),
        param_types={
            "action": str,
            "metric": str,
            "threshold": float,
            "enabled": bool,
        },
    ),
}


def validate_task(task_type: str, params: dict) -> tuple[bool, str]:
    if task_type not in TASK_REGISTRY:
        return False, f"Task '{task_type}' is not in the registry."

    spec = TASK_REGISTRY[task_type]

    for required in spec.required_params:
        if required not in params:
            return False, f"Missing required param: '{required}'"

    for param, value in params.items():
        if param in spec.param_types:
            expected = spec.param_types[param]
            if not isinstance(value, expected):
                try:
                    if expected is int:
                        params[param] = int(value)
                    elif expected is float:
                        params[param] = float(value)
                    elif expected is str:
                        params[param] = str(value)
                except (ValueError, TypeError):
                    return (
                        False,
                        f"Param '{param}' must be {expected.__name__}, got {type(value).__name__}",
                    )

    return True, ""


def get_allowed_tasks() -> list[str]:
    return sorted(TASK_REGISTRY.keys())
