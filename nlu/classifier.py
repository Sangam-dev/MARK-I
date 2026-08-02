from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from core.bus import EventBus
from core.events import Intent, IntentIdentified, TextInputReceived, TranscriptReady
from core.audio_state import audio_state
from nlu.schemas import NLUResult
from reasoning.llm_client import GeminiClient

logger = logging.getLogger("kancha.nlu.classifier")


@dataclass(frozen=True, slots=True)
class ToolDecision:
    task_name: str
    parameters: dict[str, Any]


_OPEN_RE = re.compile(
    r"^\s*(?:open|launch|start|run)\s+(?:the\s+)?(?P<app>[\w .+-]+?)\s*$",
    re.IGNORECASE,
)

_FILE_LOCATION_RE = re.compile(
    r"\b(?:in|from|inside|on)\s+(?P<path>desktop|downloads|documents|pictures|music|videos|home)\b",
    re.IGNORECASE,
)


def _extract_file_location(text: str, default: str = "desktop") -> str:
    match = _FILE_LOCATION_RE.search(text)
    return match.group("path").lower() if match else default


def _strip_file_location(text: str) -> str:
    return _FILE_LOCATION_RE.sub("", text).strip(" .")


def _classify_file_request(cleaned: str) -> ToolDecision | None:
    lowered = cleaned.lower()

    if re.search(r"\b(?:list|show)\s+(?:my\s+)?(?:files|folders|directory|contents)\b", lowered):
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
            parameters={"action": "disk_usage", "path": _extract_file_location(cleaned, "home")},
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
        normalized_protocol = protocol_map.get(protocol_name, protocol_name.replace(" ", "_"))

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
        return ToolDecision(task_name="desktop_control", parameters={"action": action, **params})

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
    if re.search(r"\b(?:list|show)\s+(?:all\s+)?(?:the\s+)?(?:open\s+)?windows\b", lowered):
        return _dc("list_windows")

    # focus / bring to front — handled earlier in classify_tool_request
    # so it doesn't get captured by _OPEN_RE (open chrome).

    m_close = re.search(r"\bclose\s+(?:the\s+)?(?P<app>.+?)\s+window\s*$", cleaned, re.IGNORECASE)
    if not m_close:
        m_close = re.search(r"\bclose\s+(?:the\s+)?(?P<app>.+?)\s*$", cleaned, re.IGNORECASE)
    if m_close:
        app = m_close.group("app").strip().rstrip(".?!")
        # Strip trailing "window" (when it's a noun, not part of the app name)
        app = re.sub(r"\s+window\b", "", app, flags=re.IGNORECASE).strip()
        # Strip trailing "please" / "for me" / "now"
        app = re.sub(r"\s+(?:please|for\s+me|now)\s*$", "", app, flags=re.IGNORECASE).strip()
        app = re.sub(r"^the\s+", "", app, flags=re.IGNORECASE).strip()
        if app:
            return _dc("close_window", app=app)

    m_min = re.search(r"\bminimize\s+(?:the\s+)?(?P<app>.+?)\s+window\s*$", cleaned, re.IGNORECASE)
    if not m_min:
        m_min = re.search(r"\bminimize\s+(?:the\s+)?(?P<app>.+?)\s*$", cleaned, re.IGNORECASE)
    if m_min:
        raw = m_min.group("app").strip().rstrip(".?!")
        raw = re.sub(r"\s+window\b", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"\s+(?:please|for\s+me|now)\s*$", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"^the\s+", "", raw, flags=re.IGNORECASE).strip()
        # "minimize all windows" / "minimize everything" — bulk action, no app arg
        if raw.lower() in {"all", "all windows", "everything", "all of them", ""}:
            return _dc("minimize", target="all")
        if raw:
            return _dc("minimize", app=raw)

    m_max = re.search(r"\b(?:maximize|toggle\s+maximize)\s+(?:the\s+)?(?P<app>.+?)\s+window\s*$", cleaned, re.IGNORECASE)
    if not m_max:
        m_max = re.search(r"\b(?:maximize|toggle\s+maximize)\s+(?:the\s+)?(?P<app>.+?)\s*$", cleaned, re.IGNORECASE)
    if m_max:
        raw = m_max.group("app").strip().rstrip(".?!")
        raw = re.sub(r"\s+window\b", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"\s+(?:please|for\s+me|now)\s*$", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"^the\s+", "", raw, flags=re.IGNORECASE).strip()
        if raw.lower() in {"all", "all windows", "everything", "all of them", ""}:
            return _dc("maximize", target="all")
        if raw:
            return _dc("maximize", app=raw)

    # ── Virtual desktops (workspaces) ─────────────────────────────────────
    if re.search(r"\b(?:list|show|what(?:'s| is)|give\s+me)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(?:virtual\s+)?(?:desktops|workspaces)\b", lowered):
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

    m = re.search(r"\bwhich\s+(?:desktop|workspace)\s+is\s+(?P<app>.+?)\s+on\b", cleaned, re.IGNORECASE)
    if m:
        app = m.group("app").strip().rstrip(".?!")
        return _dc("window_workspace", app=app)

    # ── Desktop file management ───────────────────────────────────────────
    if re.search(r"\borganize\s+(?:my\s+|the\s+)?desktop\b", lowered):
        return _dc("organize", mode="by_type")

    if re.search(r"\bclean\s+(?:my\s+|the\s+)?desktop\b", lowered):
        return _dc("clean")

    if re.search(r"\b(?:list|show|what(?:'s| is)\s+on)\s+(?:my\s+|the\s+)?desktop(?:\s+files)?\b", lowered):
        return _dc("list")

    if re.search(r"\b(?:desktop\s+stats|stats\s+(?:for\s+)?(?:my\s+|the\s+)?desktop|how\s+big\s+is\s+my\s+desktop)\b", lowered):
        return _dc("stats")

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
            return ToolDecision(task_name="desktop_control", parameters={"action": "focus", "app": focus_app})

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
