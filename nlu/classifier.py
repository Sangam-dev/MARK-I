from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from core.audio_state import audio_state
from core.bus import EventBus
from core.events import Intent, IntentIdentified, TextInputReceived, TranscriptReady
from nlu.schemas import NLUResult
from reasoning.llm_client import GeminiClient

logger = logging.getLogger("kancha.nlu.classifier")


@dataclass(frozen=True, slots=True)
class ToolDecision:
    task_name: str
    parameters: dict[str, Any]


# Matches single-app open commands only. The negative-lookahead (?!.*\band\b)
# prevents "open firefox and file explorer" from being captured as one app
# ("firefox and file explorer") — those must fall through to the NLU LLM
# so the planner can decompose them into two separate open_app tasks.
_OPEN_RE = re.compile(
    r"^\s*(?:open|launch|start|run)\s+(?:the\s+)?(?P<app>(?!.*\band\b)[\w .+-]+?)\s*$",
    re.IGNORECASE,
)

_FILE_LOCATION_RE = re.compile(
    r"\b(?:in|from|inside|on)\s+(?P<path>desktop|downloads|documents|pictures|music|videos|home)\b",
    re.IGNORECASE,
)

# ── Web search patterns ───────────────────────────────────────────────────────

# Explicit search commands: "search for X", "look up X", "google X", etc.
# Capture group <query> is the normalised search query passed to the action.
# Each alternative must NOT consume its own trailing whitespace — the outer
# \s+ separator handles that so the query group always starts cleanly.
_SEARCH_CMD_RE = re.compile(
    r"^\s*(?:"
    r"search(?:\s+the\s+(?:web|internet|net|online))?\s+for|search|look\s+up|google|bing|"
    r"find\s+out(?:\s+about)?"
    r")\s+(?P<query>.+?)\s*$",
    re.IGNORECASE,
)

# Current / live information patterns: "what's the latest on X",
# "current price of bitcoin", "what's happening with X", etc.
# Weather/alarm/system keywords are intentionally absent — those patterns
# are checked earlier in classify_tool_request and take priority.
_LIVE_INFO_RE = re.compile(
    r"^\s*(?:"
    r"what(?:'s|\s+is|\s+are)?\s+(?:the\s+)?(?:latest|current|live|today'?s?|recent)\s+.+|"
    r"(?:latest|current|live|recent)\s+(?:news|prices?|scores?|updates?|results?|rankings?|info(?:rmation)?)\s+(?:on|about|of|for)\s+.+|"
    r"what(?:'s|\s+is|\s+was)?\s+(?:happening|going\s+on)\s+(?:in|with|to|at)\s+.+"
    r")\s*$",
    re.IGNORECASE,
)

# Factual web questions — things the LLM can't answer reliably from training
# data alone because the answer is person/event/price specific or time-sensitive.
#
# "who is/was/are/were X" — public figures, office holders, etc.
# Negative lookahead drops self-referential questions ("who are you",
# "who is jarvis") which belong in the conversational path.
#
# "what is/was happening/going on in/with X" — catches the "what is"
# form that _LIVE_INFO_RE misses (it uses 'what'?s?' not 'what\s+is').
#
# "what is X right now / currently / at the moment" — live prices,
# scores, status queries with an explicit currency marker.
_FACTUAL_WEB_RE = re.compile(
    r"^\s*(?:"
    # "who is/was/are/were X" — public figures, role holders, etc.
    r"who\s+(?:is|are|was|were)\s+(?!(?:you|your|jarvis|kancha|the\s+assistant)\b).+|"
    # "what is/was happening/going on in/with X"
    r"what(?:'s|\s+is|\s+was)?\s+(?:happening|going\s+on)\s+(?:in|with|to|at)\s+.+|"
    # "what's the situation/status/news/update in/about/of X"
    r"what(?:'s|\s+is|\s+are)?\s+(?:the\s+)?(?:situation|status|news|update|latest)\s+(?:in|with|about|on|for|of)\s+.+|"
    # "what is X right now / currently / at the moment / these days"
    r"what(?:'s|\s+is|\s+are)?\s+\S.+\s+(?:right\s+now|currently|at\s+the\s+moment|these\s+days)\s*"
    r")\s*$",
    re.IGNORECASE,
)


def _extract_file_location(text: str, default: str = "desktop") -> str:
    match = _FILE_LOCATION_RE.search(text)
    return match.group("path").lower() if match else default


def _strip_file_location(text: str) -> str:
    return _FILE_LOCATION_RE.sub("", text).strip(" .")


def _classify_file_request(cleaned: str) -> ToolDecision | None:
    lowered = cleaned.lower()

    if re.search(
        r"\b(?:list|show)\s+(?:my\s+)?(?:files|folders|directory|contents)\b", lowered
    ):
        return ToolDecision(
            task_name="file_operation",
            parameters={"action": "list", "path": _extract_file_location(cleaned)},
        )

    if re.search(r"\borganize\s+(?:my\s+)?desktop\b", lowered):
        return ToolDecision(
            task_name="file_operation",
            parameters={"action": "organize_desktop"},
        )

    if re.search(r"\b(?:disk usage|storage usage|free space)\b", lowered):
        return ToolDecision(
            task_name="file_operation",
            parameters={
                "action": "disk_usage",
                "path": _extract_file_location(cleaned, "home"),
            },
        )

    match = re.search(
        r"\b(?:read|open)\s+(?:the\s+)?(?:file\s+)?(?P<name>[\w ._+-]+\.[\w]+)\b",
        cleaned,
        re.IGNORECASE,
    )
    if match:
        return ToolDecision(
            task_name="file_operation",
            parameters={
                "action": "read",
                "path": _extract_file_location(cleaned),
                "name": match.group("name").strip(),
            },
        )

    match = re.search(
        r"\b(?:delete|remove|trash)\s+(?:the\s+)?(?:file\s+)?(?P<name>[\w ._+-]+\.[\w]+)\b",
        cleaned,
        re.IGNORECASE,
    )
    if match:
        return ToolDecision(
            task_name="file_operation",
            parameters={
                "action": "delete",
                "path": _extract_file_location(cleaned),
                "name": match.group("name").strip(),
            },
        )

    match = re.search(
        r"\b(?:create|make)\s+(?:a\s+)?folder\s+(?:named\s+|called\s+)?(?P<name>[\w ._+-]+)",
        cleaned,
        re.IGNORECASE,
    )
    if match:
        return ToolDecision(
            task_name="file_operation",
            parameters={
                "action": "create_folder",
                "path": _extract_file_location(cleaned),
                "name": _strip_file_location(match.group("name")).strip(),
            },
        )

    match = re.search(
        r"\b(?:create|make)\s+(?:a\s+)?file\s+(?:named\s+|called\s+)?(?P<name>[\w ._+-]+\.[\w]+)",
        cleaned,
        re.IGNORECASE,
    )
    if match:
        return ToolDecision(
            task_name="file_operation",
            parameters={
                "action": "create_file",
                "path": _extract_file_location(cleaned),
                "name": match.group("name").strip(),
            },
        )

    match = re.search(
        r"\b(?:find|search for)\s+(?P<name>[\w ._+-]+?)(?:\s+files?)?(?:\s+in\b|$)",
        cleaned,
        re.IGNORECASE,
    )
    if match and "file" in lowered:
        return ToolDecision(
            task_name="file_operation",
            parameters={
                "action": "find",
                "path": _extract_file_location(cleaned, "home"),
                "name": match.group("name").strip(),
            },
        )

    return None


def _classify_protocol_request(cleaned: str) -> ToolDecision | None:
    """Detect protocol execution requests like 'run genesis protocol', 'execute jarvis protocol', etc."""
    lowered = cleaned.lower()

    # Pattern to match protocol execution commands
    protocol_pattern = re.compile(
        r"\b(?:run|execute|trigger|initiate|launch|start|activate)\s+(?:the\s+)?(?P<protocol>\w+(?:\s+\w+)?)\s+protocol\b",
        re.IGNORECASE,
    )

    match = protocol_pattern.search(lowered)
    if match:
        protocol_name = match.group("protocol").strip().lower()
        # Normalize protocol names
        protocol_map = {
            "genesis": "genesis",
            "jarvis": "jarvis",
            "core": "jarvis",
            "core protocol": "jarvis",
        }
        normalized_protocol = protocol_map.get(
            protocol_name, protocol_name.replace(" ", "_")
        )

        return ToolDecision(
            task_name="execute_protocol",
            parameters={
                "protocol_name": normalized_protocol,
                "original_request": cleaned,
            },
        )

    # Also match "genesis protocol" without explicit verb
    if re.search(r"\bgenesis\s+protocol\b", lowered):
        return ToolDecision(
            task_name="execute_protocol",
            parameters={
                "protocol_name": "genesis",
                "original_request": cleaned,
            },
        )

    # Match "jarvis protocol" or "core protocol" without explicit verb
    if re.search(r"\b(?:jarvis|core)\s+protocol\b", lowered):
        return ToolDecision(
            task_name="execute_protocol",
            parameters={
                "protocol_name": "jarvis",
                "original_request": cleaned,
            },
        )

    return None


def _classify_desktop_control_request(cleaned: str) -> ToolDecision | None:
    """Detect Linux desktop automation requests.

    Covers wallpaper, window management, virtual desktops, and the
    high-frequency file-management actions. Long-form natural-language
    tasks ("click on the Save button", "drag this file there") are NOT
    matched here — those fall through to the LLM classifier and end up
    in the ``task`` action of ``desktop_control`` for sandboxed exec.
    """
    lowered = cleaned.lower()

    def _dc(action: str, **params) -> ToolDecision:
        return ToolDecision(
            task_name="desktop_control", parameters={"action": action, **params}
        )

    # ── Wallpaper ─────────────────────────────────────────────────────────
    # "set wallpaper to X", "change my wallpaper to X", "use X as wallpaper"
    m = re.search(
        r"\b(?:set|change|switch|use)\s+(?:the\s+|my\s+)?wallpaper(?:\s+(?:to|as|of))?\s+(?P<rest>.+)$",
        cleaned,
        re.IGNORECASE,
    )
    if m:
        rest = m.group("rest").strip().rstrip(".?!")
        if rest.lower().startswith("to ") or rest.lower().startswith("as "):
            rest = rest.split(" ", 1)[1].strip()
        # Heuristic: looks like a URL?
        if rest.lower().startswith(("http://", "https://")):
            return _dc("wallpaper_url", url=rest)
        return _dc("wallpaper", path=rest)

    # "use <path> as wallpaper" / "use <path> as my wallpaper"
    m = re.search(
        r"\buse\s+(?P<path>.+?)\s+as\s+(?:my\s+|the\s+)?wallpaper\s*$",
        cleaned,
        re.IGNORECASE,
    )
    if m:
        rest = m.group("path").strip().rstrip(".?!")
        if rest:
            if rest.lower().startswith(("http://", "https://")):
                return _dc("wallpaper_url", url=rest)
            return _dc("wallpaper", path=rest)

    if re.search(r"\b(?:what(?:'s| is)\s+)?(?:the\s+)?current\s+wallpaper\b", lowered):
        return _dc("current_wallpaper")

    # ── Window management ────────────────────────────────────────────────
    if re.search(
        r"\b(?:list|show)\s+(?:all\s+)?(?:the\s+)?(?:open\s+)?windows\b", lowered
    ):
        return _dc("list_windows")

    # "list running apps" / "list running applications" / "what apps are
    # running" / "what apps are open" / "show running apps" — same target
    # as list_windows (alias for user convenience). Both word orders
    # covered: "<verb> running apps" AND "<verb> apps that are running".
    if re.search(
        r"\b(?:(?:list|show)\s+(?:all\s+)?(?:the\s+)?(?:my\s+)?"
        r"(?:[\w]+\s+)?"
        r"(?:running|open)\s+(?:apps?|applications?|programs?))"
        r"|(?:what|which)\s+(?:apps?|applications?|programs?)\s+"
        r"(?:are|is)\s+(?:running|open)\b",
        lowered,
    ):
        return _dc("list_windows")

    # focus / bring to front — handled earlier in classify_tool_request
    # so it doesn't get captured by _OPEN_RE (open chrome).

    m_close = re.search(
        r"\bclose\s+(?:the\s+)?(?P<app>.+?)\s+window\s*$", cleaned, re.IGNORECASE
    )
    if not m_close:
        m_close = re.search(
            r"\bclose\s+(?:the\s+)?(?P<app>.+?)\s*$", cleaned, re.IGNORECASE
        )
    if m_close:
        app = m_close.group("app").strip().rstrip(".?!")
        # Strip trailing "window" (when it's a noun, not part of the app name)
        app = re.sub(r"\s+window\b", "", app, flags=re.IGNORECASE).strip()
        # Strip trailing "please" / "for me" / "now"
        app = re.sub(
            r"\s+(?:please|for\s+me|now)\s*$", "", app, flags=re.IGNORECASE
        ).strip()
        app = re.sub(r"^the\s+", "", app, flags=re.IGNORECASE).strip()
        if app:
            return _dc("close_window", app=app)

    m_min = re.search(
        r"\bminimize\s+(?:the\s+)?(?P<app>.+?)\s+window\s*$", cleaned, re.IGNORECASE
    )
    if not m_min:
        m_min = re.search(
            r"\bminimize\s+(?:the\s+)?(?P<app>.+?)\s*$", cleaned, re.IGNORECASE
        )
    if m_min:
        raw = m_min.group("app").strip().rstrip(".?!")
        raw = re.sub(r"\s+window\b", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(
            r"\s+(?:please|for\s+me|now)\s*$", "", raw, flags=re.IGNORECASE
        ).strip()
        raw = re.sub(r"^the\s+", "", raw, flags=re.IGNORECASE).strip()
        # "minimize all windows" / "minimize everything" — bulk action, no app arg
        if raw.lower() in {"all", "all windows", "everything", "all of them", ""}:
            return _dc("minimize", target="all")
        if raw:
            return _dc("minimize", app=raw)

    m_max = re.search(
        r"\b(?:maximize|toggle\s+maximize)\s+(?:the\s+)?(?P<app>.+?)\s+window\s*$",
        cleaned,
        re.IGNORECASE,
    )
    if not m_max:
        m_max = re.search(
            r"\b(?:maximize|toggle\s+maximize)\s+(?:the\s+)?(?P<app>.+?)\s*$",
            cleaned,
            re.IGNORECASE,
        )
    if m_max:
        raw = m_max.group("app").strip().rstrip(".?!")
        raw = re.sub(r"\s+window\b", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(
            r"\s+(?:please|for\s+me|now)\s*$", "", raw, flags=re.IGNORECASE
        ).strip()
        raw = re.sub(r"^the\s+", "", raw, flags=re.IGNORECASE).strip()
        if raw.lower() in {"all", "all windows", "everything", "all of them", ""}:
            return _dc("maximize", target="all")
        if raw:
            return _dc("maximize", app=raw)

    # ── Virtual desktops (workspaces) ─────────────────────────────────────
    if re.search(
        r"\b(?:list|show|what(?:'s| is)|give\s+me)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(?:virtual\s+)?(?:desktops|workspaces)\b",
        lowered,
    ):
        return _dc("list_workspaces")

    m = re.search(
        r"\b(?:switch\s+to|go\s+to|move\s+to|change\s+to)\s+(?:desktop|workspace)\s+(?P<target>.+)$",
        cleaned,
        re.IGNORECASE,
    )
    if m:
        target = m.group("target").strip().rstrip(".?!")
        return _dc("switch_workspace", target=target)

    # "move X to desktop N" (X is the app/window name)
    m = re.search(
        r"\b(?:move|send)\s+(?:the\s+)?(?P<app>(?:the\s+)?[\w+-]+?)\s+(?:window\s+)?to\s+(?:desktop|workspace)\s+(?P<target>.+)$",
        cleaned,
        re.IGNORECASE,
    )
    if m:
        app = m.group("app").strip()
        target = m.group("target").strip().rstrip(".?!")
        # If "app" is literally "window" or "the", skip the app param
        if app.lower() in {"window", "the", "the window"}:
            return _dc("move_to_workspace", target=target, follow=False)
        return _dc("move_to_workspace", app=app, target=target, follow=False)

    m = re.search(
        r"\bwhich\s+(?:desktop|workspace)\s+is\s+(?P<app>.+?)\s+on\b",
        cleaned,
        re.IGNORECASE,
    )
    if m:
        app = m.group("app").strip().rstrip(".?!")
        return _dc("window_workspace", app=app)

    # ── Desktop file management ───────────────────────────────────────────
    if re.search(r"\borganize\s+(?:my\s+|the\s+)?desktop\b", lowered):
        return _dc("organize", mode="by_type")

    if re.search(r"\bclean\s+(?:my\s+|the\s+)?desktop\b", lowered):
        return _dc("clean")

    if re.search(
        r"\b(?:list|show|what(?:'s| is)\s+on)\s+(?:my\s+|the\s+)?desktop(?:\s+files)?\b",
        lowered,
    ):
        return _dc("list")

    if re.search(
        r"\b(?:desktop\s+stats|stats\s+(?:for\s+)?(?:my\s+|the\s+)?desktop|how\s+big\s+is\s+my\s+desktop)\b",
        lowered,
    ):
        return _dc("stats")

    return None


# ── System monitor ──────────────────────────────────────────────────────────────
#
# Matches the four-action surface exposed by ``system_monitor`` in
# tasks/registry.py / tasks/executor.py:
#   status         — one-shot CPU/RAM/temp/GPU snapshot
#   check_alerts   — run a single threshold check
#   set_threshold  — change a metric's alert threshold
#   enable / disable — toggle the background alert loop
#
# Order matters: more specific patterns (set_threshold with a number,
# enable/disable) go BEFORE the general "system status" catch-all so the
# "set CPU threshold to 50" sentence isn't misread as "system status".

_METRIC_WORDS = r"(?:cpu|ram|memory|temperature|temp|gpu)"

# set / change / raise / lower / adjust <metric> threshold [to / at <value>]
_SET_THRESHOLD_RE = re.compile(
    rf"\b(?:set|change|raise|lower|adjust|update)\s+(?:the\s+|my\s+)?"
    rf"(?P<metric>{_METRIC_WORDS})\s+(?:alert\s+|warning\s+|critical\s+)?"
    rf"threshold\s*(?:to|at|of)?\s*(?P<val>\d+)?",
    re.IGNORECASE,
)

# enable / disable / turn on / turn off / start / stop <system monitor|monitoring|alerts>
_MONITOR_TOGGLE_RE = re.compile(
    r"\b(?P<verb>enable|disable|turn\s+on|turn\s+off|start|stop|pause|resume)"
    r"\s+(?:the\s+|my\s+)?(?P<target>system\s+monitoring|system\s+monitor|"
    r"system\s+alerts|monitoring|monitor|alerts)\b",
    re.IGNORECASE,
)

# "any system alerts?" / "check system health" / "run a system check"
#
# IMPORTANT: every alternative here must end with an alert-y word
# (alerts/issues/health/check/for+alerts). Bare "check the system" is
# intentionally NOT included — it would steal "check the system status"
# away from the status branch, since status is checked second.
_CHECK_ALERTS_RE = re.compile(
    r"\b(?:any\s+system\s+(?:alerts|issues|problems?)|"
    r"run\s+(?:a\s+)?system\s+check|"
    r"check\s+system\s+health|"
    r"check\s+for\s+(?:system\s+)?(?:alerts|issues|problems?)|"
    r"system\s+(?:health|issues))\b",
    re.IGNORECASE,
)

# "system status" / "system health" / "how's my cpu" / "cpu usage" / "ram level"
# / "gpu temperature" / "what temperature is my cpu at"
_STATUS_RE = re.compile(
    rf"\b(?:system\s+(?:status|health|stats|info|metrics)|"
    rf"how(?:'s| is)\s+(?:my\s+)?(?:cpu|ram|memory|gpu|temperature|temp|pc|computer|laptop|system)|"
    rf"(?:cpu|ram|memory|gpu|temperature|temp)\s+"
    rf"(?:usage|use|level|load|temperature|temp|status)|"
    rf"what(?:'s| is)\s+(?:my\s+)?(?:cpu|ram|memory|gpu|temperature|temp)\s+(?:at|usage|level))\b",
    re.IGNORECASE,
)


def _normalize_metric(metric: str) -> str:
    """Map user surface words (``memory``, ``temperature``) to canonical
    keys (``ram``, ``temp``) used by ``actions.system_monitor.SystemMonitor``.
    """
    m = metric.lower().strip()
    if m in {"memory"}:
        return "ram"
    if m in {"temperature"}:
        return "temp"
    return m


def _classify_system_monitor_request(cleaned: str) -> ToolDecision | None:
    """Detect system-monitor intents and return a ToolDecision.

    Precedence (most specific → least):
        1. set_threshold with a value
        2. enable / disable toggle
        3. check_alerts
        4. status (catch-all)
    """
    lowered = cleaned.lower()

    # 1. set_threshold — must come before _STATUS_RE because
    #    "set cpu threshold to 50" contains "cpu" and could plausibly be
    #    read as a status query by a loose matcher.
    m = _SET_THRESHOLD_RE.search(cleaned)
    if m:
        metric = _normalize_metric(m.group("metric"))
        val = m.group("val")
        if val is None:
            # Ask the user for the value — the executor also returns a
            # nice message if it gets here without a number.
            return ToolDecision(
                task_name="system_monitor",
                parameters={"action": "set_threshold", "metric": metric},
            )
        return ToolDecision(
            task_name="system_monitor",
            parameters={
                "action": "set_threshold",
                "metric": metric,
                "threshold": float(val),
            },
        )

    # 2. enable / disable
    m = _MONITOR_TOGGLE_RE.search(cleaned)
    if m:
        verb = m.group("verb").lower().strip()
        on = verb in {"enable", "turn on", "start", "resume"}
        return ToolDecision(
            task_name="system_monitor",
            parameters={"action": "enable" if on else "disable", "enabled": on},
        )

    # 3. check_alerts
    if _CHECK_ALERTS_RE.search(lowered):
        return ToolDecision(
            task_name="system_monitor",
            parameters={"action": "check_alerts"},
        )

    # 4. status (catch-all). The bare word "status" alone is too greedy —
    #    only match when it sits next to a system/monitor word OR a metric.
    if _STATUS_RE.search(lowered):
        return ToolDecision(
            task_name="system_monitor",
            parameters={"action": "status"},
        )

    return None


def _classify_web_search_request(cleaned: str) -> ToolDecision | None:
    """Detect explicit search commands and live-info queries.

    Called last in :func:`classify_tool_request` so every more-specific
    pattern (weather, alarms, desktop control, etc.) takes priority.
    """
    # Explicit command: "search for X", "look up X", "google X", ...
    m = _SEARCH_CMD_RE.match(cleaned)
    if m:
        return ToolDecision(
            task_name="web_search",
            parameters={"query": m.group("query").strip()},
        )
    # Live/current-information or factual web question — use the full
    # cleaned text as the query so the search engine sees the whole intent.
    if _LIVE_INFO_RE.match(cleaned) or _FACTUAL_WEB_RE.match(cleaned):
        return ToolDecision(
            task_name="web_search",
            parameters={"query": cleaned},
        )
    return None


def classify_tool_request(text: str) -> ToolDecision | None:
    cleaned = " ".join(text.strip().split()).rstrip(".?!")
    if not cleaned:
        return None

    lowered = cleaned.lower()

    # "focus X" / "bring X to the front" — check before _OPEN_RE so it
    # doesn't get stolen by the open_app path. These always mean an
    # existing window.
    focus_app = None
    m1 = re.match(
        r"^\s*(?:focus|bring\s+up)\s+(?:the\s+)?(?P<app>.+?)(?:\s+window)?\s*$",
        cleaned,
        re.IGNORECASE,
    )
    m2 = re.match(
        r"^\s*bring\s+(?:the\s+)?(?P<app>.+?)\s+(?:to\s+the\s+front|forward)\s*$",
        cleaned,
        re.IGNORECASE,
    )
    if m1:
        focus_app = m1.group("app").strip().rstrip(".?!")
    elif m2:
        focus_app = m2.group("app").strip().rstrip(".?!")
    if focus_app:
        focus_app = re.sub(r"^the\s+", "", focus_app, flags=re.IGNORECASE).strip()
        if focus_app:
            return ToolDecision(
                task_name="desktop_control",
                parameters={"action": "focus", "app": focus_app},
            )

    open_match = _OPEN_RE.match(cleaned)
    if open_match:
        return ToolDecision(
            task_name="open_app",
            parameters={"app_name": open_match.group("app").strip()},
        )

    file_decision = _classify_file_request(cleaned)
    if file_decision is not None:
        return file_decision

    # Check for desktop automation requests (wallpaper, windows, workspaces, etc.)
    desktop_decision = _classify_desktop_control_request(cleaned)
    if desktop_decision is not None:
        return desktop_decision

    # Check for system-monitor requests (CPU/RAM/temp/GPU snapshots, threshold
    # changes, enable/disable the background alert loop).
    monitor_decision = _classify_system_monitor_request(cleaned)
    if monitor_decision is not None:
        return monitor_decision

    # Check for protocol execution requests
    protocol_decision = _classify_protocol_request(cleaned)
    if protocol_decision is not None:
        return protocol_decision

    if re.search(
        r"\b(?:set|create|make|schedule|add)\s+(?:an?\s+)?(?:alarm|reminder|timer)\b",
        lowered,
    ) or lowered.startswith(
        (
            "set alarm",
            "set an alarm",
            "set a alarm",
            "alarm",
            "timer",
            "remind me",
            "wake me",
        )
    ):
        # Try to parse delay_seconds if present
        delay_match = re.search(
            r"\b(\d+)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\b", lowered
        )
        delay_seconds = 10  # default fallback
        if delay_match:
            val = int(delay_match.group(1))
            unit = delay_match.group(2)
            if "minute" in unit or "min" in unit:
                delay_seconds = val * 60
            elif "hour" in unit or "hr" in unit:
                delay_seconds = val * 3600
            else:
                delay_seconds = val

        desc = cleaned
        # Strip alarm command prefixes to make description cleaner
        desc_cleaned = re.sub(
            r"^(set alarm|set an alarm|set a alarm|alarm|timer|remind me|wake me)\s*(to|for|in|at)?\s*",
            "",
            desc,
            flags=re.IGNORECASE,
        )
        return ToolDecision(
            task_name="set_alarm",
            parameters={
                "description": desc_cleaned.strip(),
                "delay_seconds": str(delay_seconds),
            },
        )

    if lowered in {"alarms", "list alarms", "show alarms"} or re.search(
        r"\b(?:list|show|what are)\s+(?:my\s+)?(?:alarms?|reminders?|timers?)\b",
        lowered,
    ):
        return ToolDecision(task_name="list_alarms", parameters={})

    if lowered in {"cancel alarms", "clear alarms", "delete alarms", "stop alarms"}:
        return ToolDecision(task_name="cancel_alarms", parameters={})

    if re.search(
        r"\b(?:weather|forecast|temperature|temp|rain|raining|sunny|cloudy)\b",
        lowered,
    ):
        # Clean query, extract place
        place_match = re.search(r"\b(?:in|at|for)\s+([a-zA-Z\s]+)$", cleaned)
        city = place_match.group(1).strip() if place_match else cleaned
        return ToolDecision(task_name="get_weather", parameters={"city": city})

    if lowered in {
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
        return ToolDecision(task_name="sleep", parameters={})

    if lowered in {
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
        return ToolDecision(task_name="shutdown", parameters={})

    if lowered in {
        "restart the computer now",
        "restart my computer now",
        "restart this device now",
        "reboot now",
        "reboot the computer now",
        "reboot my device",
        "reboot this device now",
    }:
        return ToolDecision(task_name="restart", parameters={})

    # Web search — checked last so every specific tool pattern takes priority.
    web_decision = _classify_web_search_request(cleaned)
    if web_decision is not None:
        return web_decision

    return None


NLU_SYSTEM_PROMPT = """You are the NLU intent classifier for KANCHA, a smart assistant.
Your job is to classify the user's input intent and extract parameters if a task/tool is requested.

Classify intent into one of:
- "query": User is asking a question or seeking information (e.g., "what is the capital of France?", "who is the president?").
- "task": User is asking to perform a device action/tool. Allowed tasks:
  * "open_app" (params: app_name)
  * "set_alarm" (params: description, delay_seconds)
  * "list_alarms" (no params)
  * "cancel_alarms" (no params)
  * "get_weather" (params: city, optional: date, units)
  * "sleep" (no params)
  * "shutdown" (no params)
  * "restart" (no params)
  * "file_operation" (params: action (required), path, name, content, destination, new_name, extension (optional). Action values: list, create_file, create_folder, delete, move, copy, rename, read, write, find, largest, disk_usage, organize_desktop, info)- "execute_protocol" (params: protocol_name (required), original_request (optional))- "conversational": Casual greetings, chitchat, or social statements (e.g., "hi", "how are you?", "nice to meet you").
- "desktop_control" (params: action (required), plus action-specific optional params). Action values: wallpaper, wallpaper_url, current_wallpaper, organize, clean, list, stats, list_windows, focus, close_window, minimize, maximize, list_workspaces, switch_workspace, move_to_workspace, window_workspace, task. For action="task" the natural-language description goes in the "task" param. For action="focus"/"close_window"/"minimize"/"maximize" the app name goes in "app". For action="switch_workspace" the desktop/workspace name or number goes in "target". For action="wallpaper" the image path goes in "path"; for "wallpaper_url" the URL goes in "url".
- "system_monitor" (params: action (required); optional: metric, threshold, enabled). Action values: status (one-shot CPU/RAM/temp/GPU/uptime snapshot), check_alerts (run a single threshold check), set_threshold (needs metric ∈ {cpu,ram,temp,gpu} and threshold ∈ 1..99), enable/disable (toggle the background alert loop using enabled=true|false).
- "web_search" (params: query (required)). Use for any request that needs live or current information from the web: news, prices, scores, events, documentation, research, public figures, recent announcements, etc. The query should be the user's question or search phrase as-is.

Return ONLY valid JSON matching this schema:
{
  "intent": "query" | "task" | "conversational",
  "confidence": float (0.0 to 1.0),
  "requires_task_execution": boolean,
  "task_type": string or null (e.g. "open_app", "set_alarm"),
  "task_params": object (parameters for the task)
}
Do NOT include markdown formatting or code blocks in your response. Return raw JSON string only."""


class NLUClassifier:
    """LLM-based intent classification and entity extraction, with offline regex fallback."""

    def __init__(self, llm_client: GeminiClient, bus: EventBus) -> None:
        self.llm_client = llm_client
        self.bus = bus

    def register(self) -> None:
        """Subscribe to text and transcript events."""
        self.bus.subscribe(TextInputReceived, self.on_text_input)
        self.bus.subscribe(TranscriptReady, self.on_transcript_ready)

    async def classify(self, text: str, session_id: str = "default") -> NLUResult:
        """Classify input text using fast regex path first, falling back to LLM."""
        # 1. Regex Fast Path
        decision = classify_tool_request(text)
        if decision is not None:
            logger.info(
                "Regex matched task: %s with params: %s",
                decision.task_name,
                decision.parameters,
            )
            return NLUResult(
                intent=Intent.TASK,
                requires_task_execution=True,
                task_type=decision.task_name,
                task_params=decision.parameters,
                confidence=1.0,
            )

        # 2. LLM Fallback
        prompt = f'Classify the following user input:\n\n"{text}"'
        try:
            result_dict = await self.llm_client.generate_json(
                prompt=prompt,
                schema_description="JSON object matching NLUResult schema.",
                system=NLU_SYSTEM_PROMPT,
            )
            if not result_dict:
                return NLUResult(intent=Intent.CONVERSATIONAL)

            return NLUResult.model_validate(result_dict)
        except Exception as e:
            logger.exception("LLM classification failed: %s", e)
            return NLUResult(intent=Intent.CONVERSATIONAL)

    async def on_text_input(self, event: TextInputReceived) -> None:
        """Handle text input events."""
        await self._process_text(event.text, event.session_id)

    async def on_transcript_ready(self, event: TranscriptReady) -> None:
        """Handle STT transcript events."""
        await self._process_text(event.text, event.session_id)

    async def _process_text(self, text: str, session_id: str) -> None:
        """Perform classification and emit IntentIdentified event.

        The thinking gate is raised here (before any async work) and is
        released by the ReasoningCoordinator when it finishes — that
        ensures the microphone cannot reopen mid-classify or mid-stream,
        which would otherwise let audio leak in as a separate turn.
        """
        logger.info("Processing user input text: '%s'", text)
        audio_state.thinking_started()
        try:
            result = await self.classify(text, session_id)

            intent_event = IntentIdentified(
                intent=result.intent,
                raw_input=text,
                confidence=result.confidence,
                session_id=session_id,
                requires_task=result.requires_task_execution,
                task_type=result.task_type,
                task_params=result.task_params,
            )
            self.bus.emit(intent_event)
        except Exception:
            # Never leave the gate stuck if classify/emit blew up.
            audio_state.thinking_finished()
            raise
