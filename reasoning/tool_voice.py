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

import re
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


def _summarise_windows(text: str) -> str:
    """Turn the multi-line ``list_windows`` output into one spoken sentence.

    Raw output looks like::

        Open windows (6):
          [0] power.py - kancha - Visual Studio Code  (id=0x02400004)
          [0] Sakshyam messaged you - Brave  (id=0x02600004)
          ...

    Strip the ``[N]`` desktop tag and the trailing ``(id=0x...)`` so the
    voice reply reads as a clean list of window titles. Cap at 5 titles
    and append a tail summary when there are more, so a busy desktop
    doesn't produce a 30-second speech.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "I couldn't read the window list, sir."

    # First line is the header "Open windows (N):" — extract the count.
    count = 0
    titles: list[str] = []
    for ln in lines:
        if ln.lower().startswith("open windows"):
            m = re.search(r"\((\d+)\)", ln)
            if m:
                count = int(m.group(1))
            continue
        # Each line looks like: "  [0] title  (id=0x...)"
        # Drop leading "[N] " desktop tag and trailing "(id=...)".
        title = re.sub(r"^\s*\[\d+\]\s*", "", ln)
        title = re.sub(r"\s*\(id=[^)]+\)\s*$", "", title).strip()
        if title:
            titles.append(title)

    if not titles:
        return "I couldn't read the window list, sir."
    if count == 0:
        count = len(titles)

    shown = titles[:5]
    extra = count - len(shown)

    # Grammar: when extras exist, use ", plus N more" instead of cramming
    # another "and" into the human-joined list. Avoids "..., and App 4,
    # and 3 more" which reads as two consecutive "and"s.
    if extra > 0 and shown:
        listed = _human_join(shown) + f", plus {extra} more"
    elif extra > 0:
        listed = f"{extra} more"
    else:
        listed = _human_join(shown)

    if count == 1:
        return f"One window open, sir — {listed}."
    return f"{count} windows open, sir — {listed}."


def _summarise_workspaces(text: str) -> str:
    """Turn ``list_workspaces`` output into one spoken sentence."""
    # Skip the leading "Virtual desktops:" header line — it's raw label
    # text that shouldn't be in the spoken reply.
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not raw_lines:
        return "I couldn't read the workspace list, sir."

    desktops: list[tuple[str, bool]] = []  # (label, is_current)
    for ln in raw_lines:
        # Drop the header line(s) — anything that starts with "virtual desktops".
        if ln.lower().startswith("virtual desktop"):
            continue
        is_current = "◀ current" in ln
        label = ln.replace("◀ current", "").strip()
        if label:
            desktops.append((label, is_current))

    if not desktops:
        return text  # fallback to raw if we couldn't parse anything.

    count = len(desktops)
    current_label = next((lbl for lbl, cur in desktops if cur), None)
    listed = _human_join([lbl for lbl, _ in desktops])
    if count == 1:
        return f"One virtual desktop, sir — {listed}."
    if current_label:
        return f"{count} virtual desktops, sir. Current: {current_label}."
    return f"{count} virtual desktops, sir — {listed}."


def _human_join(items: list[str]) -> str:
    """Join a short list the way JARVIS speaks it: A, B, and C."""
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _voice_desktop_control(message: str, args: dict[str, Any]) -> str:
    """Voice for desktop_control.

    Listing-style actions (list_windows, list_workspaces, list, stats)
    return multi-line machine text that the user shouldn't hear raw —
    summarise them into one spoken sentence. Pass through everything
    else (focus / close / wallpaper / etc.) unchanged so we don't
    accidentally paraphrase a confirmation message.
    """
    text = (message or "").strip()
    if not text:
        return "Done, sir."

    action = (args.get("action") or "").lower().strip()
    if action == "list_windows":
        # Only summarise if the raw text is multi-line — single-line
        # failure strings should still pass through.
        if "\n" in text:
            return _summarise_windows(text)
        return text
    if action == "list_workspaces":
        if "\n" in text:
            return _summarise_workspaces(text)
        return text
    if action == "list":
        # "list" returns the desktop file listing — multi-line, but the
        # user explicitly asked to see files, so pass through.
        return text
    if action == "stats":
        # Single-line; pass through.
        return text
    # window_workspace / focus / close / wallpaper / etc. — pass through.
    return text


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
    target = (args.get("target") or "").strip()
    # Already in JARVIS shape?
    if text.lower().startswith(("opening ", "could not confirm")):
        return text
    if text.lower().startswith("launch command sent for"):
        if app and target:
            # Speak the name the user used, not the absolute path the
            # launcher resolved it to.
            return f"Opening {target} in {app} for you, sir."
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

    # Where it landed is worth saying out loud — it is the one detail the
    # user can't otherwise check without going and looking.
    where = f" in {path}" if path else ""

    if action == "create_folder":
        if "already exists" in text.lower() or "access denied" in text.lower():
            return text
        if name:
            return f"Done — folder '{name}' created{where}, sir."
        return f"Folder created{where}, sir."
    if action == "create_file":
        if "could not" in text.lower() or "access denied" in text.lower():
            return text
        if name:
            return f"File '{name}' saved{where}, sir."
        return f"File saved{where}, sir."
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


def _voice_web_search(message: str, _args: dict[str, Any]) -> str:
    """Voice for web_search.

    Gemini grounding already returns plain text shaped to 2-3 sentences;
    DuckDuckGo answers are also plain text. Both are already
    TTS-ready — pass them through unchanged. The only transformation
    needed is a graceful fallback for empty results.
    """
    text = (message or "").strip()
    if not text:
        return "I couldn't find anything on that right now, sir."
    return text


def _voice_agent_task(message: str, args: dict[str, Any]) -> str:
    """Voice for the delegated coding agent.

    Two things must survive intact: that a delegation has *started* and
    is not finished, and the progress report itself, which is already
    written to be read aloud (see agent/progress.py). The default
    colon-rewrite would turn both into something shorter and wrong.
    """
    text = (message or "").strip()
    if not text:
        return "Done, sir."

    action = (args.get("action") or "").lower().strip()

    if action in {"delegate", "follow_up"}:
        # The tool's output is written for the LLM ("it is NOT finished")
        # rather than for the user; say the short human version instead.
        label = (args.get("label") or "").strip()
        if text.lower().startswith("the coding agent has started"):
            what = f"'{label}'" if label else "it"
            return (
                f"The coding agent is on it, sir — I'll let you know when "
                f"{what} is done. Ask me for the progress any time."
            )
        return text

    return text


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
    # No sleep/shutdown/restart voices — those tasks no longer exist.
    "execute_protocol": _passthrough,
    "desktop_control": _voice_desktop_control,
    "system_monitor": _passthrough,
    # SystemTool already returns spoken-shape text ("Wi-Fi turned off.",
    # "CPU at 42%, 287 processes."), and its listings are multi-line —
    # the default colon-rewrite would mangle those into one blob.
    "system": _passthrough,
    # GmailTool output is already spoken-shape ("Email sent to …",
    # "'Invoice' moved to Trash."), and its listings are multi-line —
    # the default colon-rewrite would mangle those into one blob.
    "gmail": _passthrough,
    # "Playing 'Despacito' by Luis Fonsi on YouTube." is already the
    # sentence to say — the colon-rewrite would mangle the quoted title.
    "youtube_video": _passthrough,
    "web_search": _voice_web_search,
    "agent_task": _voice_agent_task,
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
