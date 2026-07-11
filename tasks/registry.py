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
    "sleep": TaskSpec(
        name="sleep",
        description="Put the current device to sleep.",
        required_params=(),
        optional_params=(),
        requires_confirmation=True,
        is_destructive=True,
        param_types={},
    ),
    "shutdown": TaskSpec(
        name="shutdown",
        description="Shut the current device down.",
        required_params=(),
        optional_params=(),
        requires_confirmation=True,
        is_destructive=True,
        param_types={},
    ),
    "restart": TaskSpec(
        name="restart",
        description="Restart the current device.",
        required_params=(),
        optional_params=(),
        requires_confirmation=True,
        is_destructive=True,
        param_types={},
    ),
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
