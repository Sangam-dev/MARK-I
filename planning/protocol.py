"""The Task LLM response protocol.

The Task LLM is not allowed to talk to the user. Everything it needs to
express comes back as one of five structured responses, and this module
is the only place that decides whether a given payload is one of them.

    input_required          I need a value I do not have.
    confirmation_required   This changes something; get approval first.
    execute                 Running these tools with these arguments.
    completed               Here is what happened.
    failed                  Here is why it did not happen.

Validation is strict and total: :func:`parse_task_response` either
returns a typed object or raises :class:`ProtocolError`. There is no
"mostly valid" path, because the failure mode being prevented is a
malformed payload turning into a question the user cannot answer, or —
worse — an ``execute`` that nobody vetted.

Dataclasses rather than Pydantic: the events, the plan models and the
task registry in this project are all plain dataclasses, and a parser
that raises on bad input gives the same guarantee here without adding a
validation framework to one module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResponseType(str, Enum):
    INPUT_REQUIRED = "input_required"
    CONFIRMATION_REQUIRED = "confirmation_required"
    EXECUTE = "execute"
    COMPLETED = "completed"
    FAILED = "failed"


class ProtocolError(ValueError):
    """A Task LLM response that does not conform. Never acted on."""


@dataclass(slots=True)
class InputRequired:
    """The Task LLM needs values it does not have.

    ``question`` is a *draft* for the Conversation LLM to deliver in its
    own voice — the Task LLM does not address the user directly, so this
    is never spoken verbatim.
    """

    task_id: str
    missing_fields: list[str]
    question: str
    type: ResponseType = ResponseType.INPUT_REQUIRED


@dataclass(slots=True)
class ConfirmationRequired:
    """The work is sensitive and needs the user's approval first."""

    task_id: str
    action: str
    description: str
    confirmation_data: dict[str, Any] = field(default_factory=dict)
    type: ResponseType = ResponseType.CONFIRMATION_REQUIRED


@dataclass(slots=True)
class Execute:
    """Tools are being run. Informational — the Orchestrator logs it."""

    task_id: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    type: ResponseType = ResponseType.EXECUTE


@dataclass(slots=True)
class Completed:
    task_id: str
    result: dict[str, Any] = field(default_factory=dict)
    type: ResponseType = ResponseType.COMPLETED


@dataclass(slots=True)
class Failed:
    task_id: str
    error: str = ""
    type: ResponseType = ResponseType.FAILED


TaskResponse = (
    InputRequired | ConfirmationRequired | Execute | Completed | Failed
)


def _require_str(payload: dict[str, Any], key: str, kind: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{kind}: '{key}' must be a non-empty string")
    return value.strip()


def _require_dict(payload: dict[str, Any], key: str, kind: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ProtocolError(f"{kind}: '{key}' must be an object")
    return dict(value)


def _require_field_list(payload: dict[str, Any], key: str, kind: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, (list, tuple)) or not value:
        raise ProtocolError(f"{kind}: '{key}' must be a non-empty list")
    fields: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ProtocolError(f"{kind}: '{key}' entries must be non-empty strings")
        fields.append(item.strip())
    return fields


def parse_task_response(
    response_type: str, payload: dict[str, Any], task_id: str = ""
) -> TaskResponse:
    """Validate one Task LLM response.

    *task_id* is the id the Orchestrator dispatched. A payload naming a
    different task is rejected outright rather than reconciled: a
    response that cannot say which task it belongs to must not be
    allowed to advance any of them.

    :raises ProtocolError: on any malformed or unknown response.
    """
    if not isinstance(payload, dict):
        raise ProtocolError("payload must be an object")

    kind = str(response_type or "").strip().lower()
    try:
        parsed_type = ResponseType(kind)
    except ValueError:
        raise ProtocolError(
            f"unknown response type {response_type!r}; expected one of "
            + ", ".join(t.value for t in ResponseType)
        ) from None

    # Required in the payload, not merely inferable from the dispatch.
    # A response that cannot say which task it belongs to must not be
    # allowed to advance one.
    claimed = str(payload.get("task_id") or "").strip()
    if not claimed:
        raise ProtocolError(f"{kind}: 'task_id' is required")
    if task_id and claimed != task_id:
        raise ProtocolError(
            f"{kind}: task_id {claimed!r} does not match the dispatched task {task_id!r}"
        )

    if parsed_type is ResponseType.INPUT_REQUIRED:
        return InputRequired(
            task_id=claimed,
            missing_fields=_require_field_list(payload, "missing_fields", kind),
            question=_require_str(payload, "question", kind),
        )

    if parsed_type is ResponseType.CONFIRMATION_REQUIRED:
        return ConfirmationRequired(
            task_id=claimed,
            action=_require_str(payload, "action", kind),
            description=_require_str(payload, "description", kind),
            confirmation_data=_require_dict(payload, "confirmation_data", kind),
        )

    if parsed_type is ResponseType.EXECUTE:
        return Execute(
            task_id=claimed,
            action=_require_str(payload, "action", kind),
            params=_require_dict(payload, "params", kind),
        )

    if parsed_type is ResponseType.COMPLETED:
        return Completed(
            task_id=claimed,
            result=_require_dict(payload, "result", kind),
        )

    return Failed(task_id=claimed, error=str(payload.get("error") or "").strip())
