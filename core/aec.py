"""PulseAudio echo-cancellation plumbing shared by STT capture and TTS playback."""
from __future__ import annotations

import subprocess
import threading
import time

ECHO_CANCEL_SOURCE = "echo-cancel-source"
ECHO_CANCEL_SINK = "echo-cancel-sink"

_aec_module_lock = threading.Lock()
_aec_path_unhealthy: bool | None = None
_aec_verified_at = 0.0
_AEC_REVERIFY_EVERY = 1.0


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


def _nodes_exist() -> bool:
    sources = pactl("list", "short", "sources")
    sinks = pactl("list", "short", "sinks")
    return bool(
        sources
        and sinks
        and sources.returncode == 0
        and sinks.returncode == 0
        and any(ECHO_CANCEL_SOURCE in line for line in sources.stdout.splitlines())
        and any(ECHO_CANCEL_SINK in line for line in sinks.stdout.splitlines())
    )


def load_module() -> bool:
    """Load module-echo-cancel once (race-free); True when the nodes exist."""
    with _aec_module_lock:
        if _nodes_exist():
            return True
        args = ["load-module", "module-echo-cancel", "aec_method=webrtc"]
        source, sink = default_source(), default_sink()
        if source:
            args.append(f"source_master={source}")
        if sink:
            args.append(f"sink_master={sink}")
        pactl(*args)
        return _nodes_exist()


def _count_nodes() -> tuple[int, int]:
    """(source, sink) nodes named exactly echo-cancel-source/-sink."""

    def names(out) -> list[str]:
        if not out or out.returncode != 0:
            return []
        return [f[1] for f in (line.split() for line in out.stdout.splitlines()) if len(f) > 1]

    nsrc = sum(1 for n in names(pactl("list", "short", "sources")) if n == ECHO_CANCEL_SOURCE)
    nsink = sum(1 for n in names(pactl("list", "short", "sinks")) if n == ECHO_CANCEL_SINK)
    return nsrc, nsink


def path_is_unhealthy() -> bool:
    global _aec_path_unhealthy
    if _aec_path_unhealthy is None:
        nsrc, nsink = _count_nodes()
        _aec_path_unhealthy = nsrc > 1 or nsink > 1
    return _aec_path_unhealthy


def repair_duplicates() -> bool:
    global _aec_path_unhealthy
    modules = pactl("list", "short", "modules")
    if not modules or modules.returncode != 0:
        return False
    ids = [
        line.split("\t")[0]
        for line in modules.stdout.splitlines()
        if len(line.split("\t")) >= 2 and "module-echo-cancel" in line.split("\t")[1]
    ]
    for module_id in ids[1:]:
        pactl("unload-module", module_id)
    _aec_path_unhealthy = None
    return not path_is_unhealthy()


def prepare() -> tuple[str, str] | None:
    """Route default mic+speaker through the AEC nodes.

    Returns the previous defaults (for restore), or None when AEC is
    unavailable/untrustworthy — callers then fall back to pause-while-speaking.
    """
    if not load_module():
        return None
    if path_is_unhealthy() and not repair_duplicates():
        return None
    prev_source, prev_sink = default_source(), default_sink()
    if prev_source != ECHO_CANCEL_SOURCE:
        set_default_source(ECHO_CANCEL_SOURCE)
    if prev_sink != ECHO_CANCEL_SINK:
        set_default_sink(ECHO_CANCEL_SINK)
    return prev_source, prev_sink


def restore(prev: tuple[str, str] | None) -> None:
    if not prev:
        return
    prev_source, prev_sink = prev
    if prev_source and prev_source != ECHO_CANCEL_SOURCE:
        set_default_source(prev_source)
    if prev_sink and prev_sink != ECHO_CANCEL_SINK:
        set_default_sink(prev_sink)


def ensure_routing() -> bool:
    """Re-assert AEC routing; True when mic+speaker go through the AEC nodes.

    Throttled (max one pactl round-trip per ``_AEC_REVERIFY_EVERY`` seconds)
    so per-listen re-checks cost nothing when nothing has drifted.
    """
    global _aec_verified_at
    now = time.monotonic()
    if now - _aec_verified_at < _AEC_REVERIFY_EVERY:
        return _aec_verified_at > 0
    _aec_verified_at = 0.0
    if not _nodes_exist():
        return False
    if default_source() != ECHO_CANCEL_SOURCE:
        set_default_source(ECHO_CANCEL_SOURCE)
    if default_sink() != ECHO_CANCEL_SINK:
        set_default_sink(ECHO_CANCEL_SINK)
    ok = default_source() == ECHO_CANCEL_SOURCE and default_sink() == ECHO_CANCEL_SINK
    if ok:
        _aec_verified_at = now
    return ok