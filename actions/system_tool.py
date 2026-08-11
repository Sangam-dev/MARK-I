from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger("kancha.actions.system_tool")

# How long an armed confirmation stays valid. Long enough for the user to
# answer "yes", short enough that an approval can't be reused an hour
# later against a machine in a different state.
CONFIRMATION_TTL_S: float = 120.0

# Default per-command wall clock. Generous enough for `systemctl status`
# on a loaded box, short enough that a hung binary can't wedge a turn.
DEFAULT_TIMEOUT_S: float = 15.0

# Arguments that reach a command line must look like plain identifiers.
# No whitespace, no quotes, no shell metacharacters, no path traversal —
# and never a leading "-", which a program would read as an option.
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9@._+-]*$")

_MAX_OUTPUT_CHARS = 4000


# ── Results ───────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class CommandOutcome:
    """What one subprocess did. Produced by the runner, never by an LLM."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@dataclass(slots=True)
class SystemToolResult:
    """Structured result handed back to the Task LLM layer."""

    success: bool
    output: str = ""
    error: str | None = None
    action: str = ""
    requires_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        """The wire shape: ``{"success": true, "output": "...", "error": null}``."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "action": self.action,
            "requires_confirmation": self.requires_confirmation,
        }


# ── The allowlist ─────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class ActionSpec:
    """One permitted operation.

    ``destructive`` is the confirmation gate. ``operations``, when set,
    is the closed set of values the ``operation`` argument may take —
    anything else is refused before a command is built.
    """

    name: str
    summary: str
    destructive: bool = False
    operations: tuple[str, ...] = ()
    binaries: tuple[str, ...] = field(default=())


ACTIONS: dict[str, ActionSpec] = {
    "open_application": ActionSpec(
        name="open_application",
        summary="Launch an installed application by name (target=app name).",
    ),
    "open_path": ActionSpec(
        name="open_path",
        summary="Open a file or directory in its default handler (target=path).",
        binaries=("xdg-open",),
    ),
    "wifi": ActionSpec(
        name="wifi",
        summary="Turn Wi-Fi on or off, or report its state (operation=on|off|status).",
        operations=("on", "off", "status"),
        binaries=("nmcli",),
    ),
    "brightness": ActionSpec(
        name="brightness",
        summary=(
            "Screen backlight (operation=get|set|up|down, level=1..100 for "
            "set, step=1..50 for up/down)."
        ),
        operations=("get", "set", "up", "down"),
    ),
    "volume": ActionSpec(
        name="volume",
        summary=(
            "Audio output (operation=get|set|up|down|mute|unmute, "
            "level=0..100 for set, step=1..50 for up/down)."
        ),
        operations=("get", "set", "up", "down", "mute", "unmute"),
        binaries=("pactl", "wpctl", "amixer"),
    ),
    "bluetooth": ActionSpec(
        name="bluetooth",
        summary="Turn Bluetooth on or off, or report its state (operation=on|off|status).",
        operations=("on", "off", "status"),
        binaries=("rfkill", "bluetoothctl"),
    ),
    "battery": ActionSpec(
        name="battery",
        summary="Charge level, and whether the machine is plugged in.",
    ),
    "cpu": ActionSpec(name="cpu", summary="Current CPU load."),
    "memory": ActionSpec(name="memory", summary="Current RAM usage."),
    "disk": ActionSpec(
        name="disk",
        summary="Disk usage for a mount point (target=path, default /).",
        binaries=("df",),
    ),
    "processes": ActionSpec(
        name="processes",
        summary="List the heaviest processes (limit=1..50, operation=cpu|memory).",
        operations=("cpu", "memory"),
        binaries=("ps",),
    ),
    "kill_process": ActionSpec(
        name="kill_process",
        summary="Terminate a process by pid or name.",
        destructive=True,
        binaries=("kill",),
    ),
    "lock_screen": ActionSpec(
        name="lock_screen",
        summary="Lock the desktop session.",
        binaries=("loginctl", "xdg-screensaver", "gnome-screensaver-command"),
    ),
    "system_info": ActionSpec(
        name="system_info",
        summary="Kernel, distribution, hostname and uptime.",
    ),
    "service": ActionSpec(
        name="service",
        summary=(
            "Manage a systemd unit (name=unit, operation=status|start|stop|"
            "restart, scope=user|system)."
        ),
        destructive=True,  # relaxed for `status` — see _handle_service
        operations=("status", "start", "stop", "restart"),
        binaries=("systemctl",),
    ),
}

# Power-state changes — shutdown, reboot, halt, suspend — are deliberately
# absent, and there is no handler for them anywhere in this module. They
# were removed rather than gated behind confirmation: a machine that
# powers off mid-conversation is not a recoverable mistake, and no
# approval flow is worth that risk. An `action` naming one of these is
# refused by the allowlist like any other unknown action.
#
# Enforced by tests/test_system_tool.py::test_power_actions_do_not_exist.
REMOVED_POWER_ACTIONS: frozenset[str] = frozenset(
    {"shutdown", "reboot", "restart", "poweroff", "halt", "sleep", "suspend", "hibernate"}
)

# Actions whose confirmation gate depends on the arguments rather than the
# action alone (reading a service's status is harmless; stopping it is not).
_CONDITIONALLY_DESTRUCTIVE = {"service"}


def describe_actions() -> str:
    """Render the allowlist for prompts and error messages."""
    return "\n".join(
        f"- {spec.name}: {spec.summary}"
        + (" [confirmation required]" if spec.destructive else "")
        for spec in ACTIONS.values()
    )


# ── Runner ────────────────────────────────────────────────────────────────

CommandRunner = Callable[[list[str], float], Awaitable[CommandOutcome]]


async def run_argv(argv: list[str], timeout: float = DEFAULT_TIMEOUT_S) -> CommandOutcome:
    """Execute *argv* with no shell, capturing output, bounded by *timeout*.

    Note the signature: a **list**, never a string. There is no code path
    in this module that turns text into a command line, which is what
    makes injection structurally impossible rather than merely filtered.
    """
    logger.info("SystemTool: exec %s (timeout=%.1fs)", argv, timeout)
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return CommandOutcome(tuple(argv), 127, stderr=f"{argv[0]}: not found")
    except OSError as exc:
        return CommandOutcome(tuple(argv), 126, stderr=str(exc))

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        # Leaving the child running would hold the pipe (and the box)
        # after we've stopped waiting for it.
        try:
            process.kill()
            await process.wait()
        except ProcessLookupError:
            pass
        logger.warning("SystemTool: %s timed out after %.1fs", argv[0], timeout)
        return CommandOutcome(
            tuple(argv),
            -1,
            stderr=f"timed out after {timeout:.0f}s",
            timed_out=True,
        )

    return CommandOutcome(
        argv=tuple(argv),
        returncode=process.returncode if process.returncode is not None else -1,
        stdout=stdout.decode("utf-8", errors="replace").strip(),
        stderr=stderr.decode("utf-8", errors="replace").strip(),
    )


# ── Validation helpers ────────────────────────────────────────────────────


class ArgumentError(ValueError):
    """Raised when an argument fails validation. Never reaches a command."""


def _require_token(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ArgumentError(f"'{field_name}' is required")
    if not _SAFE_TOKEN_RE.match(text):
        raise ArgumentError(
            f"'{field_name}' must be a plain name (letters, digits, . _ + - @); "
            f"got {text!r}"
        )
    return text


def _require_choice(value: Any, field_name: str, choices: tuple[str, ...]) -> str:
    text = str(value or "").strip().lower()
    if text not in choices:
        raise ArgumentError(
            f"'{field_name}' must be one of {', '.join(choices)}; got {text or '(missing)'}"
        )
    return text


def _require_existing_path(value: Any, field_name: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ArgumentError(f"'{field_name}' is required")
    path = Path(os.path.expanduser(text)).resolve()
    if not path.exists():
        raise ArgumentError(f"path does not exist: {path}")
    return path


def _require_int(value: Any, field_name: str, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ArgumentError(f"'{field_name}' must be a whole number; got {value!r}") from None
    if not low <= number <= high:
        raise ArgumentError(f"'{field_name}' must be between {low} and {high}; got {number}")
    return number


def _first_available(*binaries: str) -> str | None:
    for binary in binaries:
        found = shutil.which(binary)
        if found:
            return found
    return None


def _truncate(text: str) -> str:
    text = text.strip()
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS].rstrip() + "\n… (truncated)"


# ── The tool ──────────────────────────────────────────────────────────────


class SystemTool:
    """Executes structured system actions. No conversation, no intent."""

    def __init__(
        self,
        runner: CommandRunner | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        app_opener: Callable[[str], Any] | None = None,
    ) -> None:
        # Injectable so tests drive the full validation/dispatch path
        # without touching the real machine.
        self._run = runner or run_argv
        self._timeout = timeout
        self._app_opener = app_opener
        # fingerprint -> (expires_at, plan_id that armed it)
        self._armed: dict[str, tuple[float, str]] = {}

    # ── entry point ───────────────────────────────────────────────────

    async def execute(self, params: dict[str, Any]) -> SystemToolResult:
        """Run one structured action. Never raises."""
        action = str(params.get("action") or "").strip().lower()

        spec = ACTIONS.get(action)
        if spec is None:
            return SystemToolResult(
                success=False,
                error=(
                    f"Unknown system action '{action or '(missing)'}'. "
                    f"Supported: {', '.join(sorted(ACTIONS))}."
                ),
                action=action,
            )

        # Confirmation gate, before any argument work: a destructive
        # action must not even be planned without approval.
        if self._is_gated(spec, params):
            gate = self._check_confirmation(action, params)
            if gate is not None:
                return gate

        handler = getattr(self, f"_handle_{action}")
        try:
            return await handler(spec, params)
        except ArgumentError as exc:
            logger.info("SystemTool: rejected %s — %s", action, exc)
            return SystemToolResult(success=False, error=str(exc), action=action)
        except Exception as exc:  # noqa: BLE001
            logger.exception("SystemTool: %s crashed", action)
            return SystemToolResult(success=False, error=str(exc), action=action)

    @staticmethod
    def _is_gated(spec: ActionSpec, params: dict[str, Any]) -> bool:
        """True if this exact request needs user approval to run."""
        if not spec.destructive:
            return False
        if spec.name in _CONDITIONALLY_DESTRUCTIVE:
            # Read-only operations on an otherwise destructive action.
            if str(params.get("operation") or "").strip().lower() == "status":
                return False
        return True

    @staticmethod
    def _fingerprint(action: str, params: dict[str, Any]) -> str:
        """Identity of a request, so an approval can't be reused elsewhere.

        Approving "stop nginx" must not also approve "stop postgres", so
        the arguments are part of the identity. ``confirm`` and the task
        layer's private ``_plan_id``/``_task_id`` keys are excluded —
        they differ between the arming call and the approving one.
        """
        relevant = {
            key: value
            for key, value in sorted(params.items())
            if not key.startswith("_") and key != "confirm"
        }
        return action + ":" + json.dumps(relevant, sort_keys=True, default=str)

    def _check_confirmation(
        self, action: str, params: dict[str, Any]
    ) -> SystemToolResult | None:
        """Two-phase gate. Returns a refusal, or None to let the action run.

        The first request for a destructive action **never** executes,
        no matter what ``confirm`` says — it arms the confirmation and
        asks. Only a later request, carrying ``confirm=true`` and coming
        from a different plan (so: after the user spoke again), is let
        through. That makes "reboot the machine" impossible to satisfy
        in one turn even if the model sets the flag itself.
        """
        now = time.monotonic()
        # Drop expired arms so a stale approval can never be redeemed.
        self._armed = {
            key: value for key, value in self._armed.items() if value[0] > now
        }

        fingerprint = self._fingerprint(action, params)
        plan_id = str(params.get("_plan_id") or "")
        confirmed = bool(params.get("confirm", False))
        armed = self._armed.get(fingerprint)

        if confirmed and armed is not None:
            _, armed_plan_id = armed
            # A second task inside the *same* plan is still one turn —
            # the user has not been asked in between.
            same_plan = bool(plan_id) and plan_id == armed_plan_id
            if not same_plan:
                self._armed.pop(fingerprint, None)
                logger.info("SystemTool: '%s' confirmed by the user — executing", action)
                return None

        self._armed[fingerprint] = (now + CONFIRMATION_TTL_S, plan_id)
        logger.info(
            "SystemTool: '%s' requires confirmation — armed for %.0fs%s",
            action,
            CONFIRMATION_TTL_S,
            " (confirm flag ignored on first request)" if confirmed else "",
        )
        return SystemToolResult(
            success=False,
            error=(
                f"'{action}' changes the system, so it needs the user's approval "
                "first. Ask them to confirm, and only if they agree, send the "
                "same request again with confirm=true."
            ),
            action=action,
            requires_confirmation=True,
        )

    # ── shared plumbing ───────────────────────────────────────────────

    async def _exec(
        self, spec: ActionSpec, argv: list[str], success_output: str | None = None
    ) -> SystemToolResult:
        """Run a built argv and shape the outcome into a result."""
        outcome = await self._run(argv, self._timeout)

        if outcome.timed_out:
            return SystemToolResult(
                success=False,
                error=f"'{spec.name}' timed out after {self._timeout:.0f}s",
                action=spec.name,
            )
        if not outcome.ok:
            detail = outcome.stderr or outcome.stdout or f"exit code {outcome.returncode}"
            return SystemToolResult(
                success=False, error=_truncate(detail), action=spec.name
            )

        return SystemToolResult(
            success=True,
            output=_truncate(success_output if success_output is not None else outcome.stdout),
            action=spec.name,
        )

    def _resolve_binary(self, spec: ActionSpec) -> str:
        binary = _first_available(*spec.binaries)
        if binary is None:
            raise ArgumentError(
                f"'{spec.name}' needs one of {', '.join(spec.binaries)}, "
                "and none is installed"
            )
        return binary

    # ── applications and files ────────────────────────────────────────

    async def _handle_open_application(
        self, spec: ActionSpec, params: dict[str, Any]
    ) -> SystemToolResult:
        target = str(params.get("target") or params.get("name") or "").strip()
        if not target:
            raise ArgumentError("'target' is required (the application name)")

        # Reuse the existing launcher: it already owns the alias table
        # (vscode -> code), the platform split and the launch verification.
        opener = self._app_opener
        if opener is None:
            from actions.apps import open_app  # noqa: PLC0415 — optional dep chain

            opener = open_app

        result = await asyncio.to_thread(opener, target)
        return SystemToolResult(
            success=bool(result.success),
            output=result.message if result.success else "",
            error=None if result.success else result.message,
            action=spec.name,
        )

    async def _handle_open_path(
        self, spec: ActionSpec, params: dict[str, Any]
    ) -> SystemToolResult:
        path = _require_existing_path(params.get("target") or params.get("path"), "target")
        binary = self._resolve_binary(spec)
        return await self._exec(
            spec, [binary, str(path)], success_output=f"Opened {path}."
        )

    # ── networking ────────────────────────────────────────────────────

    async def _handle_wifi(
        self, spec: ActionSpec, params: dict[str, Any]
    ) -> SystemToolResult:
        operation = _require_choice(
            params.get("operation", "status"), "operation", spec.operations
        )
        binary = self._resolve_binary(spec)

        if operation == "status":
            outcome = await self._run([binary, "radio", "wifi"], self._timeout)
            if outcome.timed_out:
                return SystemToolResult(
                    success=False,
                    error=f"'wifi' timed out after {self._timeout:.0f}s",
                    action=spec.name,
                )
            if not outcome.ok:
                return SystemToolResult(
                    success=False,
                    error=_truncate(outcome.stderr or "could not read the Wi-Fi state"),
                    action=spec.name,
                )
            state = outcome.stdout.strip().lower() or "unknown"
            return SystemToolResult(
                success=True, output=f"Wi-Fi is {state}.", action=spec.name
            )

        return await self._exec(
            spec,
            [binary, "radio", "wifi", operation],
            success_output=f"Wi-Fi turned {operation}.",
        )

    # ── display ───────────────────────────────────────────────────────

    # Backlight control has no single Linux answer, so the handler walks a
    # chain. On this class of machine (GNOME on Wayland) the sysfs node is
    # root-owned and `xrandr --brightness` only dims XWayland clients, so
    # the D-Bus route is the one that actually moves the backlight; the
    # CLI tools are preferred when installed because they work outside
    # GNOME too.
    _GNOME_POWER_DBUS = (
        "--session",
        "--dest",
        "org.gnome.SettingsDaemon.Power",
        "--object-path",
        "/org/gnome/SettingsDaemon/Power",
        "--method",
    )

    async def _handle_brightness(
        self, spec: ActionSpec, params: dict[str, Any]
    ) -> SystemToolResult:
        operation = _require_choice(
            params.get("operation", "get"), "operation", spec.operations
        )

        current = await self._read_brightness()

        if operation == "get":
            if current is None:
                return SystemToolResult(
                    success=False,
                    error="could not read the screen brightness",
                    action=spec.name,
                )
            return SystemToolResult(
                success=True, output=f"Brightness is at {current}%.", action=spec.name
            )

        if operation == "set":
            level = _require_int(params.get("level"), "level", 1, 100)
        else:
            step = _require_int(params.get("step", 10), "step", 1, 50)
            if current is None:
                raise ArgumentError(
                    "could not read the current brightness, so it cannot be "
                    "adjusted relatively — set an explicit level instead"
                )
            level = current + step if operation == "up" else current - step
            # Clamp rather than reject: "dim it" at 5% should go to the
            # floor, not fail. 1% not 0% — a black screen looks broken.
            level = max(1, min(100, level))

        applied = await self._write_brightness(level)
        if applied is None:
            return SystemToolResult(
                success=False,
                error=(
                    "no working brightness control found (tried GNOME's "
                    "D-Bus interface, brightnessctl, light and sysfs). "
                    "Installing brightnessctl would fix this."
                ),
                action=spec.name,
            )
        return SystemToolResult(
            success=True, output=f"Brightness set to {applied}%.", action=spec.name
        )

    async def _read_brightness(self) -> int | None:
        """Current backlight as a percentage, or None if unreadable."""
        gdbus = _first_available("gdbus")
        if gdbus:
            outcome = await self._run(
                [
                    gdbus,
                    "call",
                    *self._GNOME_POWER_DBUS,
                    "org.freedesktop.DBus.Properties.Get",
                    "org.gnome.SettingsDaemon.Power.Screen",
                    "Brightness",
                ],
                self._timeout,
            )
            if outcome.ok:
                match = re.search(r"-?\d+", outcome.stdout)
                if match:
                    value = int(match.group())
                    # GNOME reports -1 when no backlight is controllable.
                    if value >= 0:
                        return value

        brightnessctl = _first_available("brightnessctl")
        if brightnessctl:
            outcome = await self._run(
                [brightnessctl, "--machine-readable", "info"], self._timeout
            )
            if outcome.ok:
                # device,class,current,percent%,max
                fields = outcome.stdout.split(",")
                if len(fields) >= 4:
                    match = re.search(r"\d+", fields[3])
                    if match:
                        return int(match.group())

        raw = self._read_sysfs_brightness()
        if raw is not None:
            return raw

        return None

    @staticmethod
    def _read_sysfs_brightness() -> int | None:
        """Read /sys/class/backlight/*/brightness as a percentage."""
        root = Path("/sys/class/backlight")
        if not root.is_dir():
            return None
        for device in sorted(root.iterdir()):
            try:
                current = int((device / "brightness").read_text().strip())
                maximum = int((device / "max_brightness").read_text().strip())
            except (OSError, ValueError):
                continue
            if maximum > 0:
                return round(current * 100 / maximum)
        return None

    async def _write_brightness(self, level: int) -> int | None:
        """Apply *level* (1..100). Returns the level set, or None if nothing worked."""
        brightnessctl = _first_available("brightnessctl")
        if brightnessctl:
            outcome = await self._run(
                [brightnessctl, "set", f"{level}%"], self._timeout
            )
            if outcome.ok:
                return level

        light = _first_available("light")
        if light:
            outcome = await self._run([light, "-S", str(level)], self._timeout)
            if outcome.ok:
                return level

        gdbus = _first_available("gdbus")
        if gdbus:
            outcome = await self._run(
                [
                    gdbus,
                    "call",
                    *self._GNOME_POWER_DBUS,
                    "org.freedesktop.DBus.Properties.Set",
                    "org.gnome.SettingsDaemon.Power.Screen",
                    "Brightness",
                    # gdbus needs the variant spelled out. `level` is an
                    # int validated to 1..100, so this is not user text.
                    f"<int32 {level}>",
                ],
                self._timeout,
            )
            if outcome.ok:
                return level

        return None

    # ── audio ─────────────────────────────────────────────────────────

    async def _handle_volume(
        self, spec: ActionSpec, params: dict[str, Any]
    ) -> SystemToolResult:
        operation = _require_choice(
            params.get("operation", "get"), "operation", spec.operations
        )
        binary = self._resolve_binary(spec)
        tool_name = Path(binary).name

        if tool_name == "pactl":
            sink = "@DEFAULT_SINK@"
            if operation == "get":
                outcome = await self._run(
                    [binary, "get-sink-volume", sink], self._timeout
                )
                if not outcome.ok:
                    return SystemToolResult(
                        success=False,
                        error=_truncate(outcome.stderr or "could not read the volume"),
                        action=spec.name,
                    )
                match = re.search(r"(\d+)%", outcome.stdout)
                level = match.group(1) if match else "unknown"
                return SystemToolResult(
                    success=True, output=f"Volume is at {level}%.", action=spec.name
                )

            if operation in {"mute", "unmute"}:
                argv = [binary, "set-sink-mute", sink, "1" if operation == "mute" else "0"]
                return await self._exec(
                    spec, argv, success_output=f"Audio {operation}d."
                )

            if operation == "set":
                level = _require_int(params.get("level"), "level", 0, 100)
                argv = [binary, "set-sink-volume", sink, f"{level}%"]
                return await self._exec(
                    spec, argv, success_output=f"Volume set to {level}%."
                )

            step = _require_int(params.get("step", 10), "step", 1, 50)
            delta = f"+{step}%" if operation == "up" else f"-{step}%"
            return await self._exec(
                spec,
                [binary, "set-sink-volume", sink, delta],
                success_output=f"Volume {operation} {step}%.",
            )

        # wpctl / amixer fallbacks keep the action working on machines
        # without PulseAudio's CLI.
        if tool_name == "wpctl":
            target = "@DEFAULT_AUDIO_SINK@"
            if operation == "get":
                outcome = await self._run([binary, "get-volume", target], self._timeout)
                if not outcome.ok:
                    return SystemToolResult(
                        success=False,
                        error=_truncate(outcome.stderr or "could not read the volume"),
                        action=spec.name,
                    )
                match = re.search(r"([\d.]+)", outcome.stdout)
                level = round(float(match.group(1)) * 100) if match else "unknown"
                return SystemToolResult(
                    success=True, output=f"Volume is at {level}%.", action=spec.name
                )
            if operation in {"mute", "unmute"}:
                argv = [binary, "set-mute", target, "1" if operation == "mute" else "0"]
                return await self._exec(spec, argv, success_output=f"Audio {operation}d.")
            if operation == "set":
                level = _require_int(params.get("level"), "level", 0, 100)
                return await self._exec(
                    spec,
                    [binary, "set-volume", target, f"{level / 100:.2f}"],
                    success_output=f"Volume set to {level}%.",
                )
            step = _require_int(params.get("step", 10), "step", 1, 50)
            delta = f"{step / 100:.2f}{'+' if operation == 'up' else '-'}"
            return await self._exec(
                spec,
                [binary, "set-volume", target, delta],
                success_output=f"Volume {operation} {step}%.",
            )

        # amixer
        if operation == "get":
            outcome = await self._run([binary, "get", "Master"], self._timeout)
            if not outcome.ok:
                return SystemToolResult(
                    success=False,
                    error=_truncate(outcome.stderr or "could not read the volume"),
                    action=spec.name,
                )
            match = re.search(r"(\d+)%", outcome.stdout)
            level = match.group(1) if match else "unknown"
            return SystemToolResult(
                success=True, output=f"Volume is at {level}%.", action=spec.name
            )
        if operation in {"mute", "unmute"}:
            return await self._exec(
                spec,
                [binary, "set", "Master", operation],
                success_output=f"Audio {operation}d.",
            )
        if operation == "set":
            level = _require_int(params.get("level"), "level", 0, 100)
            return await self._exec(
                spec,
                [binary, "set", "Master", f"{level}%"],
                success_output=f"Volume set to {level}%.",
            )
        step = _require_int(params.get("step", 10), "step", 1, 50)
        return await self._exec(
            spec,
            [binary, "set", "Master", f"{step}%{'+' if operation == 'up' else '-'}"],
            success_output=f"Volume {operation} {step}%.",
        )

    # ── radios ────────────────────────────────────────────────────────

    async def _handle_bluetooth(
        self, spec: ActionSpec, params: dict[str, Any]
    ) -> SystemToolResult:
        operation = _require_choice(
            params.get("operation", "status"), "operation", spec.operations
        )
        rfkill = _first_available("rfkill")

        if operation == "status":
            if rfkill:
                outcome = await self._run(
                    [rfkill, "list", "bluetooth"], self._timeout
                )
                if outcome.ok:
                    blocked = "yes" in outcome.stdout.lower().split("soft blocked:")[-1][:6]
                    return SystemToolResult(
                        success=True,
                        output=f"Bluetooth is {'off' if blocked else 'on'}.",
                        action=spec.name,
                    )
            bluetoothctl = _first_available("bluetoothctl")
            if bluetoothctl:
                outcome = await self._run([bluetoothctl, "show"], self._timeout)
                if outcome.ok:
                    powered = "powered: yes" in outcome.stdout.lower()
                    return SystemToolResult(
                        success=True,
                        output=f"Bluetooth is {'on' if powered else 'off'}.",
                        action=spec.name,
                    )
            return SystemToolResult(
                success=False,
                error="could not read the Bluetooth state",
                action=spec.name,
            )

        if rfkill:
            return await self._exec(
                spec,
                [rfkill, "unblock" if operation == "on" else "block", "bluetooth"],
                success_output=f"Bluetooth turned {operation}.",
            )

        bluetoothctl = _first_available("bluetoothctl")
        if bluetoothctl:
            return await self._exec(
                spec,
                [bluetoothctl, "power", operation],
                success_output=f"Bluetooth turned {operation}.",
            )

        raise ArgumentError("no Bluetooth control found (tried rfkill, bluetoothctl)")

    # ── power supply ──────────────────────────────────────────────────

    async def _handle_battery(
        self, spec: ActionSpec, params: dict[str, Any]
    ) -> SystemToolResult:
        def _read() -> str | None:
            import psutil  # noqa: PLC0415

            battery = psutil.sensors_battery()
            if battery is None:
                return None
            parts = [f"Battery at {battery.percent:.0f}%"]
            if battery.power_plugged:
                parts.append("plugged in")
            elif battery.secsleft and battery.secsleft > 0:
                hours, minutes = divmod(int(battery.secsleft) // 60, 60)
                parts.append(f"about {hours}h {minutes}m left")
            return ", ".join(parts) + "."

        text = await asyncio.to_thread(_read)
        if text is None:
            return SystemToolResult(
                success=False,
                error="no battery detected on this machine",
                action=spec.name,
            )
        return SystemToolResult(success=True, output=text, action=spec.name)

    # ── metrics ───────────────────────────────────────────────────────

    async def _handle_cpu(
        self, spec: ActionSpec, params: dict[str, Any]
    ) -> SystemToolResult:
        snapshot = await asyncio.to_thread(self._status_snapshot)
        cpu = snapshot.get("cpu_percent")
        if cpu is None:
            return SystemToolResult(
                success=False, error="CPU usage is unavailable", action=spec.name
            )
        parts = [f"CPU at {cpu:.0f}%"]
        temp = snapshot.get("cpu_temp_c")
        if temp:
            parts.append(f"temp {temp:.0f}°C")
        processes = snapshot.get("process_count")
        if processes:
            parts.append(f"{processes} processes")
        return SystemToolResult(success=True, output=", ".join(parts) + ".", action=spec.name)

    async def _handle_memory(
        self, spec: ActionSpec, params: dict[str, Any]
    ) -> SystemToolResult:
        snapshot = await asyncio.to_thread(self._status_snapshot)
        used = snapshot.get("ram_used_gb")
        total = snapshot.get("ram_total_gb")
        percent = snapshot.get("ram_percent")
        if used is None or total is None or percent is None:
            return SystemToolResult(
                success=False, error="RAM usage is unavailable", action=spec.name
            )
        return SystemToolResult(
            success=True,
            output=f"RAM {used:.1f} of {total:.0f} GB in use ({percent:.0f}%).",
            action=spec.name,
        )

    async def _handle_disk(
        self, spec: ActionSpec, params: dict[str, Any]
    ) -> SystemToolResult:
        raw_target = params.get("target") or params.get("path") or "/"
        path = _require_existing_path(raw_target, "target")
        binary = self._resolve_binary(spec)
        outcome = await self._run([binary, "-h", "--output=size,used,avail,pcent", str(path)], self._timeout)

        if outcome.timed_out:
            return SystemToolResult(
                success=False,
                error=f"'disk' timed out after {self._timeout:.0f}s",
                action=spec.name,
            )
        if not outcome.ok:
            return SystemToolResult(
                success=False,
                error=_truncate(outcome.stderr or "could not read disk usage"),
                action=spec.name,
            )

        return SystemToolResult(
            success=True,
            output=self._format_df(outcome.stdout, path),
            action=spec.name,
        )

    @staticmethod
    def _format_df(stdout: str, path: Path) -> str:
        """Turn ``df`` columns into one spoken line, falling back to raw text."""
        lines = [ln for ln in stdout.splitlines() if ln.strip()]
        if len(lines) >= 2:
            fields = lines[1].split()
            if len(fields) >= 4:
                size, used, avail, pcent = fields[:4]
                return (
                    f"{path}: {used} of {size} used ({pcent}), {avail} free."
                )
        return stdout.strip()

    @staticmethod
    def _status_snapshot() -> dict[str, Any]:
        """Reuse the existing psutil snapshot rather than a second sampler."""
        from actions.system_monitor import get_system_status  # noqa: PLC0415

        return get_system_status()

    # ── processes ─────────────────────────────────────────────────────

    async def _handle_processes(
        self, spec: ActionSpec, params: dict[str, Any]
    ) -> SystemToolResult:
        limit = _require_int(params.get("limit", 10), "limit", 1, 50)
        operation = _require_choice(
            params.get("operation", "cpu"), "operation", spec.operations
        )
        sort_key = "-%cpu" if operation == "cpu" else "-%mem"
        binary = self._resolve_binary(spec)

        outcome = await self._run(
            [binary, "-eo", "pid,comm,%cpu,%mem", "--sort", sort_key], self._timeout
        )
        if outcome.timed_out:
            return SystemToolResult(
                success=False,
                error=f"'processes' timed out after {self._timeout:.0f}s",
                action=spec.name,
            )
        if not outcome.ok:
            return SystemToolResult(
                success=False,
                error=_truncate(outcome.stderr or "could not list processes"),
                action=spec.name,
            )

        lines = [ln for ln in outcome.stdout.splitlines() if ln.strip()]
        body = lines[1 : limit + 1] if len(lines) > 1 else []
        if not body:
            return SystemToolResult(
                success=True, output="No processes reported.", action=spec.name
            )

        rendered = []
        for line in body:
            fields = line.split(None, 3)
            if len(fields) >= 4:
                pid, comm, cpu, mem = fields[:4]
                rendered.append(f"{comm} (pid {pid}) — CPU {cpu}%, RAM {mem}%")
            else:
                rendered.append(line.strip())

        header = f"Top {len(rendered)} processes by {'CPU' if operation == 'cpu' else 'memory'}"
        return SystemToolResult(
            success=True,
            output=_truncate(header + ":\n" + "\n".join(rendered)),
            action=spec.name,
        )

    async def _handle_kill_process(
        self, spec: ActionSpec, params: dict[str, Any]
    ) -> SystemToolResult:
        pids = await self._resolve_pids(params)
        binary = self._resolve_binary(spec)
        # SIGTERM, not SIGKILL: ask the process to exit and let it clean up.
        argv = [binary, "-TERM", *[str(pid) for pid in pids]]
        return await self._exec(
            spec,
            argv,
            success_output=f"Sent SIGTERM to {', '.join(str(p) for p in pids)}.",
        )

    async def _resolve_pids(self, params: dict[str, Any]) -> list[int]:
        """Turn either an explicit pid or a process name into PIDs.

        Name lookup goes through psutil rather than ``pkill`` so the tool
        knows exactly which processes it is about to signal, and can
        refuse an over-broad match instead of terminating dozens.
        """
        raw_pid = params.get("pid")
        if raw_pid not in (None, ""):
            pid = _require_int(raw_pid, "pid", 1, 2**22)
            return [pid]

        name = _require_token(params.get("name") or params.get("target"), "name")

        def _lookup() -> list[int]:
            import psutil  # noqa: PLC0415

            found = []
            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    if (proc.info.get("name") or "").lower() == name.lower():
                        found.append(int(proc.info["pid"]))
                except Exception:  # noqa: BLE001 — process vanished mid-scan
                    continue
            return found

        pids = await asyncio.to_thread(_lookup)
        if not pids:
            raise ArgumentError(f"no running process is named {name!r}")
        if len(pids) > 10:
            raise ArgumentError(
                f"{len(pids)} processes are named {name!r} — refusing a bulk kill; "
                "pass an explicit pid"
            )
        return pids

    # ── session and host ──────────────────────────────────────────────

    async def _handle_lock_screen(
        self, spec: ActionSpec, params: dict[str, Any]
    ) -> SystemToolResult:
        # Desktop environments disagree about which of these exists;
        # take the first one actually installed.
        for binary, args in (
            ("loginctl", ["lock-session"]),
            ("xdg-screensaver", ["lock"]),
            ("gnome-screensaver-command", ["-l"]),
        ):
            found = shutil.which(binary)
            if found:
                return await self._exec(
                    spec, [found, *args], success_output="Screen locked."
                )
        raise ArgumentError(
            "no supported screen locker found (tried loginctl, xdg-screensaver, "
            "gnome-screensaver-command)"
        )

    async def _handle_system_info(
        self, spec: ActionSpec, params: dict[str, Any]
    ) -> SystemToolResult:
        snapshot = await asyncio.to_thread(self._status_snapshot)
        parts = [
            f"{platform.system()} {platform.release()}",
            f"host {platform.node()}",
            f"{platform.machine()}",
        ]
        uptime = snapshot.get("uptime")
        if uptime:
            parts.append(f"up {uptime}")

        uname = shutil.which("uname")
        if uname:
            outcome = await self._run([uname, "-o"], self._timeout)
            if outcome.ok and outcome.stdout:
                parts.insert(1, outcome.stdout.strip())

        return SystemToolResult(
            success=True, output=", ".join(parts) + ".", action=spec.name
        )

    # ── services ──────────────────────────────────────────────────────

    async def _handle_service(
        self, spec: ActionSpec, params: dict[str, Any]
    ) -> SystemToolResult:
        operation = _require_choice(
            params.get("operation"), "operation", spec.operations
        )
        unit = _require_token(params.get("name") or params.get("target"), "name")
        scope = _require_choice(params.get("scope", "user"), "scope", ("user", "system"))
        binary = self._resolve_binary(spec)

        argv = [binary]
        if scope == "user":
            argv.append("--user")
        if operation == "status":
            # --no-pager keeps systemctl from trying to page into a tty
            # that doesn't exist here and hanging until the timeout.
            argv += ["status", unit, "--no-pager", "--lines=0"]
        else:
            argv += [operation, unit]

        outcome = await self._run(argv, self._timeout)

        if outcome.timed_out:
            return SystemToolResult(
                success=False,
                error=f"'service' timed out after {self._timeout:.0f}s",
                action=spec.name,
            )

        if operation == "status":
            # `systemctl status` exits non-zero for a stopped unit; that
            # is a valid answer to the question, not a failure.
            text = outcome.stdout or outcome.stderr
            if not text:
                return SystemToolResult(
                    success=False,
                    error=f"no status reported for {unit}",
                    action=spec.name,
                )
            return SystemToolResult(
                success=True, output=_truncate(text), action=spec.name
            )

        if not outcome.ok:
            return SystemToolResult(
                success=False,
                error=_truncate(outcome.stderr or outcome.stdout or "command failed"),
                action=spec.name,
            )
        return SystemToolResult(
            success=True, output=f"{unit} {operation}ed.", action=spec.name
        )

    # ── power ─────────────────────────────────────────────────────────
    #
    # Intentionally empty. There is no _handle_shutdown / _handle_reboot /
    # _handle_sleep, and nothing in this module imports actions.power.
    # See REMOVED_POWER_ACTIONS above.


# ── Process-wide instance ─────────────────────────────────────────────────

_shared_tool: SystemTool | None = None


def get_shared_system_tool() -> SystemTool:
    """Lazily construct (and cache) the SystemTool used by the executor."""
    global _shared_tool
    if _shared_tool is None:
        _shared_tool = SystemTool()
    return _shared_tool


def set_shared_system_tool(tool: SystemTool | None) -> None:
    """Swap the shared instance (tests inject a tool with a fake runner)."""
    global _shared_tool
    _shared_tool = tool
