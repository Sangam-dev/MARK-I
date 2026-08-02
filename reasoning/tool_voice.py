"""Per-tool JARVIS voice.

Maps a tool name to a deterministic string rewriter that turns the tool's
machine-style result message into a short, conversational reply.

Why a registry and not a per-tool rewrite
-----------------------------------------
The tool layer (``actions/*.py``) returns data-shaped strings
(``"Created folder: Test"``, ``"Launch command sent for firefox."``).
Rewriting every tool's source message is fragile and pollutes the tool
layer with presentation logic. Instead, we wrap the tool layer here:
when a plan finishes, each task's raw result runs through this
registry before being shown to the user.

Adding a new tool to :data:`tasks.registry.TASK_REGISTRY` requires **no**
change here. The default voice handles unfamiliar tools by canonicalizing
colon-separated log lines (``"X: y"`` → ``"Done, sir — y"``) and passing
clean text through. If a tool needs more nuanced phrasing, register a
voice function under its name.

Voice functions are pure (``str, dict -> str``) and never call the LLM.
The LLM paraphrase pass in :mod:`reasoning.naturalize` runs **only**
when a plan has multiple tasks or any failure — single-task successes
go straight to the user through this registry.
"""

from __future__ import annotations

from typing import Any, Callable

# ---------------------------------------------------------------------------
# Voice primitives
# ---------------------------------------------------------------------------


def _passthrough(message: str, _args: dict[str, Any]) -> str:
    """Return the message unchanged.

    Use for tools that already speak JARVIS-shape (``set_alarm``,
    ``cancel_alarms``, power actions, ``execute_protocol``).
    """
    if not message:
        return "Done, sir."
    return message


def _default_voice(message: str, _args: dict[str, Any]) -> str:
    """Default voice for any tool without a registered voice function.

    If the message looks like ``"Label: content"`` (a colon-separated
    log line), drop the label and prefix with ``"Done, sir — "``. Pass
    through any text that doesn't look log-shaped.
    """
    text = (message or "").strip()
    if not text:
        return "Done, sir."
    # Already conversational? leave alone.
    if _already_natural(text):
        return text
    # Colon-separated log line: "Created folder: Test"
    if ":" in text and text.index(":") < 30 and "\n" not in text.split(":", 1)[0]:
        head, _, tail = text.partition(":")
        head = head.strip()
        tail = tail.strip()
        # If head is a short title-cased label (≤4 words), drop it.
        if 0 < len(head) <= 40 and head == head.strip().strip("."):
            return f"Done, sir — {tail}."
    return text


# ---------------------------------------------------------------------------
# Per-tool voices
# ---------------------------------------------------------------------------


def _voice_open_app(message: str, args: dict[str, Any]) -> str:
    text = (message or "").strip()
    app = (args.get("app_name") or "").strip()
    # Already in JARVIS shape?
    if text.lower().startswith(("opening ", "could not confirm")):
        return text
    if text.lower().startswith("launch command sent for"):
        # Re-extract the app name from the message if we have one.
        # Format: "Launch command sent for firefox."
        rest = text.split("for", 1)[-1].strip().rstrip(".")
        if rest:
            return f"Opening {rest} for you, sir."
        if app:
            return f"Opening {app} for you, sir."
        return "Done, sir."
    if not text:
        if app:
            return f"Opening {app} for you, sir."
        return "Done, sir."
    # Anything else (failure message, etc.) — pass through with the sir prefix.
    if app and app.lower() not in text.lower():
        return f"Done, sir — {text}"
    return text


def _voice_weather(message: str, args: dict[str, Any]) -> str:
    text = (message or "").strip()
    if not text:
        return "I couldn't get the weather just now, sir."
    if text.lower().startswith("currently in"):
        return text
    # The raw wttr.in format from `_format_weather` looks like:
    # "Weather in London: Clear, 18C/64F, feels like 17C, humidity 65%, wind 12 km/h."
    if text.lower().startswith("weather in"):
        body = text.split(":", 1)[-1].strip()
        # Pull out the place name if the args carry it, else leave the
        # body alone (it already contains the place as a leading word).
        place = (args.get("city") or args.get("place") or "").strip()
        prefix = f"Currently in {place}, sir — " if place else "Currently, sir — "
        return f"{prefix}{body}"
    return text


def _voice_file_operation(message: str, args: dict[str, Any]) -> str:
    text = (message or "").strip()
    if not text:
        return "Done, sir."
    action = (args.get("action") or "").lower().strip()
    name = (args.get("name") or "").strip()
    path = (args.get("path") or "").strip()

    if action == "create_folder":
        if name:
            return f"Done — folder '{name}' created, sir."
        return "Folder created, sir."
    if action == "create_file":
        if name:
            return f"File '{name}' saved, sir."
        return "File saved, sir."
    if action == "delete":
        return f"Removed, sir." if "moved to trash" in text.lower() else text
    if action in ("move", "copy"):
        if "could not" in text.lower() or "not found" in text.lower():
            return text
        return f"Done, sir — {text.rstrip('.')}."
    if action == "rename":
        if "could not" in text.lower() or "already exists" in text.lower():
            return text
        return f"Done, sir — {text.rstrip('.')}."
    if action == "write":
        if "could not" in text.lower():
            return text
        return f"Done, sir — {text.rstrip('.')}."
    if action == "read":
        return text  # long-form content — leave intact.
    if action == "list":
        return text  # directory listing — leave intact.
    if action == "find":
        return text
    if action == "largest":
        return text
    if action == "disk_usage":
        return text
    if action == "organize_desktop":
        return text
    if action == "info":
        return text
    # Unknown action — fall through to default normalization.
    return _default_voice(text, args)


# ---------------------------------------------------------------------------
# Registry + dispatcher
# ---------------------------------------------------------------------------


ToolVoice = Callable[[str, dict[str, Any]], str]

TOOL_VOICE: dict[str, ToolVoice] = {
    "open_app": _voice_open_app,
    "set_alarm": _passthrough,
    "list_alarms": _passthrough,
    "cancel_alarms": _passthrough,
    "get_weather": _voice_weather,
    "file_operation": _voice_file_operation,
    "sleep": _passthrough,
    "shutdown": _passthrough,
    "restart": _passthrough,
    "execute_protocol": _passthrough,
    "desktop_control": _passthrough,
}


def naturalize_single_tool(
    tool: str, message: str, arguments: dict[str, Any] | None
) -> str:
    """Apply the voice function for *tool* to *message*.

    Falls back to :func:`_default_voice` for unknown tools, which is why
    adding a new tool to ``TASK_REGISTRY`` requires no change here.
    """
    voice = TOOL_VOICE.get(tool, _default_voice)
    try:
        return voice(message or "", arguments or {})
    except Exception:
        # Voice functions must never break the response path.
        return (message or "").strip() or "Done, sir."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _already_natural(text: str) -> bool:
    """True if *text* already sounds like a JARVIS reply.

    Used by the default voice to skip the colon-rewrite when the
    message already reads as conversation.
    """
    lowered = text.lower()
    prefixes = (
        "opening ",
        "currently",
        "done",
        "shutting",
        "restarting",
        "putting",
        "alarm set",
        "scheduled",
        "cancelled",
        "file '",
        "folder '",
        "moved ",
        "copied ",
        "renamed ",
        "wrote ",
        "wrote to ",
        "appended ",
        "here are",
        "weather in",
        "found ",
        "disk usage",
        "top ",
        "contents of",
        "i couldn't",
        "could not",
        "tell me",
        "no ",
    )
    
    return any(lowered.startswith(p) for p in prefixes)