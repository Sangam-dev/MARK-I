from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from core.bus import EventBus
from core.events import Intent, IntentIdentified, TextInputReceived, TranscriptReady
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


def classify_tool_request(text: str) -> ToolDecision | None:
    cleaned = " ".join(text.strip().split()).rstrip(".?!")
    if not cleaned:
        return None

    open_match = _OPEN_RE.match(cleaned)
    if open_match:
        return ToolDecision(
            task_name="open_app",
            parameters={"app_name": open_match.group("app").strip()},
        )

    lowered = cleaned.lower()

    file_decision = _classify_file_request(cleaned)
    if file_decision is not None:
        return file_decision

    if lowered.startswith(
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

    if lowered in {"alarms", "list alarms", "show alarms"}:
        return ToolDecision(task_name="list_alarms", parameters={})

    if lowered in {"cancel alarms", "clear alarms", "delete alarms", "stop alarms"}:
        return ToolDecision(task_name="cancel_alarms", parameters={})

    if re.search(r"\b(?:weather|forecast|temperature|rain|raining)\b", lowered):
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
  * "file_operation" (params: action (required), path, name, content, destination, new_name, extension (optional). Action values: list, create_file, create_folder, delete, move, copy, rename, read, write, find, largest, disk_usage, organize_desktop, info)
- "conversational": Casual greetings, chitchat, or social statements (e.g., "hi", "how are you?", "nice to meet you").

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
        """Perform classification and emit IntentIdentified event."""
        logger.info("Processing user input text: '%s'", text)
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
