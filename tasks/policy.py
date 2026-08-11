"""Argument policy: what a tool call still needs, and whether it is risky.

:data:`tasks.registry.TASK_REGISTRY` answers "does this tool exist and
are its declared parameters well-typed". It cannot answer the two
questions the Orchestrator actually has to ask before running something:

1. **Is anything missing?** Registry ``required_params`` is per *tool*,
   but the tools that matter here dispatch on an ``action``. ``gmail``
   requires only ``action`` at the registry level — yet ``send_email``
   without a ``body`` is not runnable, and the assistant should ask for
   it rather than send an empty message or fail.

2. **Is this sensitive?** Sending mail, trashing mail, killing a
   process and deleting a file all deserve the user's explicit approval,
   and which ones those are is policy, not tool mechanics.

Both live here, in one auditable place, rather than being spread across
prompt text where a model could talk itself out of them.

Relationship to the tool-level gates
------------------------------------
:mod:`actions.gmail_tool` and :mod:`actions.system_tool` each enforce
their own two-phase confirmation, and those stay exactly as they are.
This module is the layer *above*: it lets the Orchestrator ask the user
before any tool runs, so the user is asked once, in conversation, rather
than discovering the requirement through a refused tool call. The tool
gate remains the backstop — if this policy ever misses an action, the
tool still refuses to run it unapproved.
"""

from __future__ import annotations

from typing import Any

from tasks.registry import TASK_REGISTRY

# ── Per-action required fields ────────────────────────────────────────
#
# Keyed by tool, then by the value of that tool's dispatch argument.
# Only fields a human would have to supply — anything the tool can
# default is left out on purpose, so the assistant asks as little as
# possible.

_GMAIL_REQUIRED: dict[str, tuple[str, ...]] = {
    "send_email": ("to", "body"),
    "read_email": ("message_id",),
    "mark_read": ("message_id",),
    "mark_unread": ("message_id",),
    "star_email": ("message_id",),
    "unstar_email": ("message_id",),
    "archive_email": ("message_id",),
    "trash_email": ("message_id",),
    "search_emails": ("query",),
    "list_emails": (),
}

_SYSTEM_REQUIRED: dict[str, tuple[str, ...]] = {
    "open_application": ("target",),
    "open_path": ("target",),
    "service": ("name",),
}

_FILE_REQUIRED: dict[str, tuple[str, ...]] = {
    "delete": ("path",),
    "read": ("path",),
    "write": ("path", "content"),
    "move": ("path", "destination"),
    "copy": ("path", "destination"),
    "rename": ("path", "new_name"),
    "create_file": ("name",),
    "create_folder": ("name",),
    "find": ("name",),
}

_ACTION_REQUIRED: dict[str, tuple[str, dict[str, tuple[str, ...]]]] = {
    # tool -> (name of the dispatch argument, action -> required fields)
    "gmail": ("action", _GMAIL_REQUIRED),
    "system": ("action", _SYSTEM_REQUIRED),
    "file_operation": ("action", _FILE_REQUIRED),
}

# ── Sensitive actions ─────────────────────────────────────────────────
#
# Everything here is gated behind an explicit user confirmation that must
# arrive in a *subsequent* user message.

_GMAIL_SENSITIVE: dict[str, str] = {
    "send_email": "send this email",
    "trash_email": "move this email to Trash",
    "archive_email": "archive this email",
    "mark_read": "mark this email as read",
    "mark_unread": "mark this email as unread",
    "star_email": "star this email",
    "unstar_email": "unstar this email",
}

_SYSTEM_SENSITIVE: dict[str, str] = {
    "kill_process": "kill this process",
}

_FILE_SENSITIVE: dict[str, str] = {
    "delete": "delete this",
    "move": "move this",
    "rename": "rename this",
    "write": "overwrite this file",
    "organize_desktop": "reorganise the desktop",
}


def _dispatch_action(tool: str, arguments: dict[str, Any]) -> str:
    spec = _ACTION_REQUIRED.get(tool)
    key = spec[0] if spec else "action"
    return str(arguments.get(key) or "").strip().lower()


def missing_required_fields(tool: str, arguments: dict[str, Any]) -> list[str]:
    """Fields this call needs before it can run, in the order to ask.

    Combines the registry's per-tool ``required_params`` with the
    per-action table above. A field present but blank counts as missing:
    an empty subject is a choice, an empty recipient is an oversight.
    """
    spec = TASK_REGISTRY.get(tool)
    if spec is None:
        return []

    required: list[str] = list(spec.required_params)
    action_spec = _ACTION_REQUIRED.get(tool)
    if action_spec is not None:
        dispatch_key, table = action_spec
        action = str(arguments.get(dispatch_key) or "").strip().lower()
        # An unknown action is not a missing *field* — the tool will
        # reject it with a much better message than we could invent.
        required.extend(table.get(action, ()))

    missing: list[str] = []
    for name in required:
        value = arguments.get(name)
        if value is None:
            missing.append(name)
        elif isinstance(value, str) and not value.strip():
            missing.append(name)
        elif isinstance(value, (list, tuple, dict)) and not value:
            missing.append(name)
    # Preserve order, drop duplicates (a field can be required twice).
    seen: set[str] = set()
    return [f for f in missing if not (f in seen or seen.add(f))]


def describe_sensitive_action(tool: str, arguments: dict[str, Any]) -> str | None:
    """A short description of the risk, or None if the call is safe.

    The description is written to slot into a question — "Do you want me
    to *send this email*?" — because the Conversation LLM has to turn it
    into something a person would say.
    """
    action = _dispatch_action(tool, arguments)

    if tool == "gmail":
        return _GMAIL_SENSITIVE.get(action)

    if tool == "system":
        if action == "service":
            operation = str(arguments.get("operation") or "").strip().lower()
            if operation and operation != "status":
                name = str(arguments.get("name") or "the service").strip()
                return f"{operation} the {name} service"
            return None
        return _SYSTEM_SENSITIVE.get(action)

    if tool == "file_operation":
        return _FILE_SENSITIVE.get(action)

    if tool == "desktop_control":
        # The sandboxed AI-driven path can do anything a desktop can.
        if str(arguments.get("task") or "").strip():
            return "run that desktop automation"
        if action == "close":
            return "close that window"
        return None

    return None


# ── Question phrasing ─────────────────────────────────────────────────
#
# A draft question, deterministic so it is testable and costs no LLM
# call. The Conversation LLM rewrites it in its own voice before the
# user ever hears it — see reasoning/coordinator.py.

_FIELD_QUESTIONS: dict[str, str] = {
    "to": "Who should this go to?",
    "body": "What should the message say?",
    "subject": "What subject should it have?",
    "message_id": "Which message did you mean?",
    "query": "What should I search for?",
    "path": "Which path did you mean?",
    "destination": "Where should it go?",
    "new_name": "What should the new name be?",
    "name": "What should it be called?",
    "content": "What should it contain?",
    "target": "Which one did you mean?",
    "city": "Which city?",
    "app_name": "Which application?",
    "action": "What should I do exactly?",
    "request": "What would you like me to play?",
}


def question_for_fields(fields: list[str], tool: str = "") -> str:
    """Draft one question covering *fields*.

    One field gets its natural phrasing; several get a single combined
    question, because asking three questions in a row is how an
    assistant turns a favour into an interrogation.
    """
    if not fields:
        return "What should I do?"
    if len(fields) == 1:
        return _FIELD_QUESTIONS.get(
            fields[0], f"What should I use for the {fields[0].replace('_', ' ')}?"
        )
    readable = [f.replace("_", " ") for f in fields]
    joined = ", ".join(readable[:-1]) + f" and {readable[-1]}"
    return f"I still need the {joined}. What should they be?"
