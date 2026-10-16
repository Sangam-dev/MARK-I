"""PulseAudio echo-cancellation plumbing shared by STT capture and TTS playback.

``module-echo-cancel`` binds its ``source_master``/``sink_master`` **once, at
load time**, and never follows later default-device changes.  A module left
bound to an old microphone keeps existing under the name ``echo-cancel-source``
and silently feeds the assistant silence, while Jarvis keeps forcing the system
default source onto it.  This module therefore treats the loaded module's
masters as part of its health: whenever they no longer match the device the user
is actually using, the module is unloaded and reloaded against the current
defaults (see :func:`load_module`).

``KANCHA_AEC=0`` disables the AEC path entirely — callers then fall back to
pause-while-speaking (barge-in is unavailable without echo cancellation).
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time

logger = logging.getLogger("kancha.core.aec")

ECHO_CANCEL_SOURCE = "echo-cancel-source"
ECHO_CANCEL_SINK = "echo-cancel-sink"

# Marker returned by ``prepare()`` when the active output is a Bluetooth
# handsfree device: echo is already cancelled in hardware, so the AEC
# modules are intentionally NOT loaded/routed. Callers treat this as an
# active (barge-in capable) session, while ``restore()`` does nothing.
HANDSFREE_NO_AEC = object()

_AEC_DISABLED_VALUES = {"0", "false", "off", "no"}

_aec_module_lock = threading.Lock()
_aec_path_unhealthy: bool | None = None
_aec_verified_at = 0.0
_AEC_REVERIFY_EVERY = 1.0

# WirePlumber (the session manager behind PulseAudio on PipeWire) keeps the
# user's *configured* default devices in a state file. We fall back to it when
# the runtime default is Jarvis's own echo-cancel node and can no longer tell
# us what the physical device is.
_WIREPLUMBER_STATE_FILE = os.path.join(
    os.getenv("XDG_STATE_HOME", os.path.expanduser("~/.local/state")),
    "wireplumber",
    "default-nodes",
)


def aec_enabled() -> bool:
    """False when ``KANCHA_AEC`` explicitly disables echo cancellation."""
    value = os.getenv("KANCHA_AEC")
    if value is None:
        return True
    return value.strip().lower() not in _AEC_DISABLED_VALUES


def pactl(*args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            ["pactl", *args], capture_output=True, text=True, timeout=15
        )
    except Exception:
        return None


def default_source() -> str:
    r = pactl("get-default-source")
    return r.stdout.strip() if r and r.returncode == 0 else ""


def default_sink() -> str:
    r = pactl("get-default-sink")
    return r.stdout.strip() if r and r.returncode == 0 else ""


def set_default_source(name: str) -> None:
    pactl("set-default-source", name)


def set_default_sink(name: str) -> None:
    pactl("set-default-sink", name)


def is_handsfree_sink(name: str) -> bool:
    """Bluetooth HFP/HSP sinks already do their own hardware echo cancellation,
    so routing them through the AEC nodes is redundant and degrades audio."""
    if not name or not name.startswith("bluez_output."):
        return False
    if ".handsfree" in name or ".headset" in name:
        return True
    return _sink_uses_sco(name)


def _sink_uses_sco(name: str) -> bool:
    """An SCO (handsfree) sink exposes factory.name=api.bluez5.sco.sink or an
    api.bluez5.profile of headset-head-unit*. Fall back to the sample rate:
    HFP transports run mono/16kHz, while A2DP is stereo/48kHz."""
    out = pactl("list", "sinks")
    if not out or out.returncode != 0:
        return False
    block: list[str] = []
    collecting = False
    for raw in out.stdout.splitlines():
        line = raw.strip()
        if collecting and line.startswith("Sink #"):
            break
        if line.startswith("Name: "):
            in_name = line.split("Name: ", 1)[1].strip()
            collecting = in_name == name
            block = []
            continue
        if collecting:
            block.append(line)
    props = " ".join(block)
    return (
        "api.bluez5.sco.sink" in props
        or "headset-head-unit" in props
        or "1ch 16000Hz" in props
    )


def _list_short(kind: str) -> list[tuple[str, str]]:
    """``pactl list short sources|sinks`` as ``(name, state)`` pairs."""
    out = pactl("list", "short", kind)
    if not out or out.returncode != 0:
        return []
    rows: list[tuple[str, str]] = []
    for line in out.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) > 1:
            state = fields[4] if len(fields) > 4 else ""
            rows.append((fields[1], state))
    return rows


def _count_nodes() -> tuple[int, int]:
    """(source, sink) nodes named exactly echo-cancel-source/-sink."""
    nsrc = sum(1 for name, _ in _list_short("sources") if name == ECHO_CANCEL_SOURCE)
    nsink = sum(1 for name, _ in _list_short("sinks") if name == ECHO_CANCEL_SINK)
    return nsrc, nsink


def _nodes_present() -> bool:
    """Exactly one echo-cancel source and one echo-cancel sink exist.

    Stricter than a name substring match: a duplicated or renamed node is
    *unhealthy*, because it means a stale module is still loaded alongside (or
    instead of) the current one.
    """
    return _count_nodes() == (1, 1)


def _echo_cancel_modules() -> list[tuple[str, dict[str, str]]]:
    """Loaded ``module-echo-cancel`` instances as ``(module_id, args)``."""
    out = pactl("list", "short", "modules")
    if not out or out.returncode != 0:
        return []
    modules: list[tuple[str, dict[str, str]]] = []
    for line in out.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) < 2 or "module-echo-cancel" not in fields[1]:
            continue
        args: dict[str, str] = {}
        if len(fields) > 2:
            for token in fields[2].split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    args[key] = value
        modules.append((fields[0].strip(), args))
    return modules


def _module_masters(args: dict[str, str]) -> tuple[str, str]:
    return args.get("source_master", ""), args.get("sink_master", "")


def _configured_defaults(kind: str) -> list[str]:
    """Ordered configured default device names from WirePlumber (best-effort)."""
    key = "source" if kind == "sources" else "sink"
    prefix = f"default.configured.audio.{key}"
    try:
        with open(_WIREPLUMBER_STATE_FILE, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return []
    entries: list[tuple[int, str]] = []
    for line in lines:
        if "=" not in line:
            continue
        name, value = (part.strip() for part in line.split("=", 1))
        if name == prefix:
            entries.append((-1, value))
        elif name.startswith(prefix + "."):
            suffix = name[len(prefix) + 1 :]
            if suffix.isdigit():
                entries.append((int(suffix), value))
    entries.sort(key=lambda item: item[0])
    return [value for _, value in entries]


def _best_physical(kind: str) -> str:
    """Last-resort physical device: a running one if any, else the first."""
    our = ECHO_CANCEL_SOURCE if kind == "sources" else ECHO_CANCEL_SINK
    rows = [
        (name, state)
        for name, state in _list_short(kind)
        if name != our and not name.endswith(".monitor")
    ]
    for name, state in rows:
        if state == "RUNNING":
            return name
    return rows[0][0] if rows else ""


def _resolve_master(kind: str) -> str:
    """The physical device the AEC module should be bound to.

    Prefers the live default; when Jarvis has already rerouted the defaults to
    its own nodes, falls back to WirePlumber's configured defaults and finally
    to a running physical device. Empty means "could not determine" — callers
    then leave the existing binding alone rather than guess.
    """
    our = ECHO_CANCEL_SOURCE if kind == "sources" else ECHO_CANCEL_SINK
    runtime = default_source() if kind == "sources" else default_sink()
    if runtime and runtime != our:
        return runtime
    available = {name for name, _ in _list_short(kind)}
    for name in _configured_defaults(kind):
        if name and name != our and name in available:
            return name
    return _best_physical(kind)


def _rebind_needed(
    modules: list[tuple[str, dict[str, str]]], want_source: str, want_sink: str
) -> bool:
    """True when the loaded module(s) don't match the wanted binding."""
    if len(modules) != 1:
        return True
    cur_source, cur_sink = _module_masters(modules[0][1])
    if want_source and cur_source != want_source:
        return True
    if want_sink and cur_sink != want_sink:
        return True
    return False


def load_module() -> bool:
    """Ensure exactly one echo-cancel module, bound to the current devices.

    A master mismatch (the device changed since the module was loaded, or the
    module was left stale by a previous run) is treated as unhealthy: the stale
    module is unloaded and reloaded against the device the user is actually
    using. This is what stops the mic from silently capturing a dead input.
    Returns True once a healthy node pair exists.
    """
    with _aec_module_lock:
        modules = _echo_cancel_modules()
        want_source = _resolve_master("sources")
        want_sink = _resolve_master("sinks")

        if not _rebind_needed(modules, want_source, want_sink) and _nodes_present():
            return True

        logger.info(
            "Reloading module-echo-cancel bound to source=%s sink=%s (was %d module(s))",
            want_source or "<default>",
            want_sink or "<default>",
            len(modules),
        )
        for module_id, _ in modules:
            pactl("unload-module", module_id)

        args = ["load-module", "module-echo-cancel", "aec_method=webrtc"]
        if want_source:
            args.append(f"source_master={want_source}")
        if want_sink:
            args.append(f"sink_master={want_sink}")
        pactl(*args)

        if not _nodes_present():
            logger.warning("module-echo-cancel did not create the expected nodes")
            return False
        return True


def path_is_unhealthy() -> bool:
    global _aec_path_unhealthy
    if _aec_path_unhealthy is None:
        nsrc, nsink = _count_nodes()
        _aec_path_unhealthy = nsrc > 1 or nsink > 1
    return _aec_path_unhealthy


def repair_duplicates() -> bool:
    global _aec_path_unhealthy
    for module_id, _ in _echo_cancel_modules()[1:]:
        pactl("unload-module", module_id)
    _aec_path_unhealthy = None
    return not path_is_unhealthy()


def _reset_verification() -> None:
    global _aec_verified_at
    _aec_verified_at = 0.0


def prepare() -> tuple[str, str] | object | None:
    """Route default mic+speaker through the AEC nodes.

    Returns the previous defaults (for restore); ``HANDSFREE_NO_AEC`` when the
    active output is a Bluetooth handsfree device (hardware echo cancel, no AEC
    needed); or None when AEC is disabled/unavailable/untrustworthy — callers
    then fall back to pause-while-speaking.
    """
    if not aec_enabled():
        return None
    if is_handsfree_sink(default_sink()):
        return HANDSFREE_NO_AEC
    if not load_module():
        return None
    if path_is_unhealthy() and not repair_duplicates():
        return None
    prev_source, prev_sink = default_source(), default_sink()
    if prev_source != ECHO_CANCEL_SOURCE:
        set_default_source(ECHO_CANCEL_SOURCE)
    if prev_sink != ECHO_CANCEL_SINK:
        set_default_sink(ECHO_CANCEL_SINK)
    _reset_verification()
    return prev_source, prev_sink


def restore(prev: tuple[str, str] | object | None) -> None:
    if prev is None or prev is HANDSFREE_NO_AEC:
        return
    prev_source, prev_sink = prev
    if prev_source and prev_source != ECHO_CANCEL_SOURCE:
        set_default_source(prev_source)
    if prev_sink and prev_sink != ECHO_CANCEL_SINK:
        set_default_sink(prev_sink)
    _reset_verification()


def ensure_routing() -> bool:
    """Re-assert AEC routing; True when mic+speaker go through the AEC nodes.

    Throttled (max one round of pactl calls per ``_AEC_REVERIFY_EVERY`` seconds)
    so per-listen re-checks cost nothing when nothing has drifted. Unlike the
    old version this also *re-binds* the module: if the user switched microphones
    since the session started, the AEC node is reloaded against the new device
    instead of fighting the change with a node wired to the old one.
    """
    global _aec_verified_at
    now = time.monotonic()
    if now - _aec_verified_at < _AEC_REVERIFY_EVERY:
        return _aec_verified_at > 0
    _aec_verified_at = 0.0
    if not aec_enabled():
        return False
    if is_handsfree_sink(default_sink()):
        _aec_verified_at = now
        return True
    if not load_module():
        return False
    if default_source() != ECHO_CANCEL_SOURCE:
        set_default_source(ECHO_CANCEL_SOURCE)
    if default_sink() != ECHO_CANCEL_SINK:
        set_default_sink(ECHO_CANCEL_SINK)
    ok = default_source() == ECHO_CANCEL_SOURCE and default_sink() == ECHO_CANCEL_SINK
    if ok:
        _aec_verified_at = now
    return ok
