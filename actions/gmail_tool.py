"""Gmail tool — the allowlist and safety gate between the LLM and Gmail.

The Task LLM never reaches :mod:`actions.gmail_client` directly. It emits
a structured parameter dict — ``{"action": "send_email", "to": "...",
...}`` — and this module decides whether that is a permitted operation,
validates every argument, applies the confirmation gate, and only then
calls the client.

Why a second layer at all
-------------------------
The client speaks Gmail; this speaks *policy*. Keeping them apart means
the rule "sending needs approval" lives in exactly one place and is
enforced no matter which caller asks, and the client stays testable
without any notion of confirmation, plans, or turns.

The confirmation gate
---------------------
Reads run immediately. Everything that changes mailbox state — sending,
trashing, archiving, flag edits — is *two-phase*, and deliberately
cannot be satisfied within a single turn:

1. The first request for a mutating action **never executes**, whatever
   ``confirm`` is set to. It arms an approval and returns a refusal that
   tells the model to ask the user.
2. Only a later request carrying ``confirm=true``, with the same
   arguments, from a **different plan**, runs.

Step 1 ignoring ``confirm`` is the load-bearing part. If the flag alone
were enough, a model that decided on its own to set ``confirm: true``
would have sent the email — the user would be told about it afterwards,
which is not consent. Requiring a different ``_plan_id`` proves a
separate turn happened, and a turn only happens when the user speaks.

Approvals are fingerprinted over the arguments, so agreeing to
"trash the newsletter" does not also approve trashing anything else,
and they expire after :data:`CONFIRMATION_TTL_S`.

This module never sees the app password: it holds a
:class:`~actions.gmail_client.GmailClient`, and credentials live inside
that client. No parameter this tool accepts can set or read them.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from actions.gmail_client import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    MAX_SEND_BODY_CHARS,
    MAX_SUBJECT_CHARS,
    GmailClient,
    GmailResult,
    ValidationError,
    get_shared_gmail_client,
    validate_addresses,
    validate_limit,
    validate_message_id,
    validate_offset,
    validate_query,
    validate_text,
)

logger = logging.getLogger("kancha.actions.gmail_tool")

# Long enough for the user to answer "yes", short enough that an
# approval cannot be redeemed against a mailbox that has since moved on.
CONFIRMATION_TTL_S: float = 120.0

# Parameter names that would mean reading or attaching a local file.
# There is no attachment support, and silently ignoring these would be
# worse than refusing: the user would be told a file was attached.
_FILE_PARAMS = frozenset({"attachment", "attachments", "file", "files", "path"})

# Parameter names that would mean the caller is trying to supply or
# override credentials. Auth belongs to the client and the environment.
_CREDENTIAL_PARAMS = frozenset(
    {"password", "app_password", "credentials", "token", "auth", "user", "username"}
)


@dataclass(frozen=True, slots=True)
class GmailAction:
    """One permitted operation.

    ``mutating`` drives the confirmation gate — it is the single place
    that decides whether an action can run unattended.
    """

    name: str
    summary: str
    mutating: bool = False


ACTIONS: dict[str, GmailAction] = {
    "list_emails": GmailAction(
        name="list_emails",
        summary=(
            "List recent inbox messages, newest first "
            "(limit=1..50, offset for paging, optional query to filter)."
        ),
    ),
    "search_emails": GmailAction(
        name="search_emails",
        summary=(
            "Search the whole account with Gmail search syntax "
            "(query, e.g. 'from:alice is:unread newer_than:2d'; limit, offset)."
        ),
    ),
    "read_email": GmailAction(
        name="read_email",
        summary="Read one message in full by message_id. Does not mark it read.",
    ),
    "send_email": GmailAction(
        name="send_email",
        summary="Send a message (to, subject, body, optional cc/bcc).",
        mutating=True,
    ),
    "mark_read": GmailAction(
        name="mark_read", summary="Mark a message as read (message_id).", mutating=True
    ),
    "mark_unread": GmailAction(
        name="mark_unread",
        summary="Mark a message as unread (message_id).",
        mutating=True,
    ),
    "archive_email": GmailAction(
        name="archive_email",
        summary="Remove a message from the inbox, keeping it in All Mail (message_id).",
        mutating=True,
    ),
    "trash_email": GmailAction(
        name="trash_email",
        summary="Move a message to Trash (message_id).",
        mutating=True,
    ),
    "star_email": GmailAction(
        name="star_email", summary="Star a message (message_id).", mutating=True
    ),
    "unstar_email": GmailAction(
        name="unstar_email", summary="Unstar a message (message_id).", mutating=True
    ),
}


@dataclass(slots=True)
class GmailToolResult:
    """Structured result handed back to the task layer."""

    success: bool
    output: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    action: str = ""
    requires_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "data": self.data,
            "error": self.error,
            "action": self.action,
            "requires_confirmation": self.requires_confirmation,
        }


def describe_actions() -> str:
    """Render the allowlist for a prompt. The catalog has one source."""
    return "\n".join(f"  {spec.name}: {spec.summary}" for spec in ACTIONS.values())


class GmailTool:
    """Validates, gates, and dispatches structured Gmail operations."""

    def __init__(self, client: GmailClient | None = None) -> None:
        self._client_override = client
        # fingerprint -> (expires_at, plan_id that armed it)
        self._armed: dict[str, tuple[float, str]] = {}

    @property
    def client(self) -> GmailClient:
        return self._client_override or get_shared_gmail_client()

    # ── entry point ───────────────────────────────────────────────────

    async def execute(self, params: dict[str, Any]) -> GmailToolResult:
        """Run one structured Gmail action. Never raises."""
        action = str(params.get("action") or "").strip().lower()

        spec = ACTIONS.get(action)
        if spec is None:
            return GmailToolResult(
                success=False,
                error=(
                    f"Unknown Gmail action '{action or '(missing)'}'. "
                    f"Supported: {', '.join(sorted(ACTIONS))}."
                ),
                action=action,
            )

        rejection = self._reject_forbidden_params(action, params)
        if rejection is not None:
            return rejection

        # Gate before any argument work: a mutating action must not even
        # be prepared without approval.
        if spec.mutating:
            gate = self._check_confirmation(action, params)
            if gate is not None:
                return gate

        handler = getattr(self, f"_handle_{action}")
        try:
            return await handler(params)
        except ValidationError as exc:
            logger.info("GmailTool: rejected %s — %s", action, exc)
            return GmailToolResult(success=False, error=str(exc), action=action)
        except Exception as exc:  # noqa: BLE001
            logger.exception("GmailTool: %s crashed", action)
            return GmailToolResult(success=False, error=str(exc), action=action)

    @staticmethod
    def _reject_forbidden_params(
        action: str, params: dict[str, Any]
    ) -> GmailToolResult | None:
        """Refuse credential and file parameters outright."""
        supplied = {key.lower() for key in params}
        if supplied & _CREDENTIAL_PARAMS:
            return GmailToolResult(
                success=False,
                action=action,
                error=(
                    "Gmail credentials come from the environment and cannot be "
                    "passed as parameters."
                ),
            )
        if supplied & _FILE_PARAMS:
            return GmailToolResult(
                success=False,
                action=action,
                error="Attachments are not supported — send the content as body text.",
            )
        return None

    # ── confirmation ──────────────────────────────────────────────────

    @staticmethod
    def _fingerprint(action: str, params: dict[str, Any]) -> str:
        """Identity of a request, so approval cannot be reused elsewhere.

        Arguments are part of the identity: approving "trash message A"
        must not approve trashing B. ``confirm`` and the task layer's
        private ``_plan_id``/``_task_id`` keys are excluded, since those
        differ between the arming call and the approving one.
        """
        relevant = {
            key: value
            for key, value in sorted(params.items())
            if not key.startswith("_") and key != "confirm"
        }
        return action + ":" + json.dumps(relevant, sort_keys=True, default=str)

    def _check_confirmation(
        self, action: str, params: dict[str, Any]
    ) -> GmailToolResult | None:
        """Two-phase gate. Returns a refusal, or None to let it run."""
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
            # A second task inside the same plan is still one turn — the
            # user was never asked in between.
            same_plan = bool(plan_id) and plan_id == armed_plan_id
            if not same_plan:
                self._armed.pop(fingerprint, None)
                logger.info("GmailTool: '%s' confirmed by the user — executing", action)
                return None

        self._armed[fingerprint] = (now + CONFIRMATION_TTL_S, plan_id)
        logger.info(
            "GmailTool: '%s' requires confirmation — armed for %.0fs%s",
            action,
            CONFIRMATION_TTL_S,
            " (confirm flag ignored on first request)" if confirmed else "",
        )
        return GmailToolResult(
            success=False,
            action=action,
            requires_confirmation=True,
            error=(
                f"'{action}' changes the user's mailbox, so it needs their "
                "approval first. Tell them exactly what will happen and ask "
                "them to confirm; only if they agree, repeat this same "
                "request with confirm=true."
            ),
        )

    # ── read handlers ─────────────────────────────────────────────────

    async def _handle_list_emails(self, params: dict[str, Any]) -> GmailToolResult:
        limit = validate_limit(params.get("limit"))
        offset = validate_offset(params.get("offset"))
        raw_query = params.get("query")
        query = validate_query(raw_query) if raw_query else None

        result = await self.client.list_messages(
            query=query, limit=limit, offset=offset
        )
        return self._render_listing(result, "list_emails", "inbox")

    async def _handle_search_emails(self, params: dict[str, Any]) -> GmailToolResult:
        query = validate_query(params.get("query"))
        limit = validate_limit(params.get("limit"))
        offset = validate_offset(params.get("offset"))

        result = await self.client.search_messages(
            query=query, limit=limit, offset=offset
        )
        return self._render_listing(result, "search_emails", f"matching {query!r}")

    async def _handle_read_email(self, params: dict[str, Any]) -> GmailToolResult:
        message_id = validate_message_id(params.get("message_id") or params.get("id"))
        result = await self.client.get_message(message_id)
        if not result.success:
            return self._failure(result, "read_email")

        message = result.data.get("message", {})
        lines = [
            f"From: {message.get('from', '')}",
            f"To: {message.get('to', '')}",
        ]
        if message.get("cc"):
            lines.append(f"Cc: {message['cc']}")
        lines.extend(
            [
                f"Subject: {message.get('subject', '')}",
                f"Date: {message.get('timestamp', '')}",
            ]
        )
        if message.get("attachments"):
            lines.append(f"Attachments: {', '.join(message['attachments'])}")
        lines.append("")
        lines.append(message.get("body") or "(no text content)")

        return GmailToolResult(
            success=True,
            action="read_email",
            output="\n".join(lines),
            data=result.data,
        )

    def _render_listing(
        self, result: GmailResult, action: str, what: str
    ) -> GmailToolResult:
        if not result.success:
            return self._failure(result, action)

        messages = result.data.get("messages", [])
        if not messages:
            return GmailToolResult(
                success=True,
                action=action,
                output=f"No messages {what}.",
                data=result.data,
            )

        lines = [f"{len(messages)} message(s) {what}:"]
        for index, message in enumerate(messages, start=1):
            marks = []
            if message.get("unread"):
                marks.append("unread")
            if message.get("starred"):
                marks.append("starred")
            suffix = f" [{', '.join(marks)}]" if marks else ""
            lines.append(
                f"{index}. {message.get('subject', '(no subject)')} "
                f"— from {message.get('from', 'unknown')} "
                f"({message.get('timestamp', '')}){suffix} "
                f"id={message.get('message_id', '')}"
            )
        if result.data.get("has_more"):
            lines.append(
                f"({result.data.get('total_matched')} matched in total — "
                f"ask for more with offset={result.data.get('offset', 0) + len(messages)})"
            )
        return GmailToolResult(
            success=True, action=action, output="\n".join(lines), data=result.data
        )

    # ── mutating handlers ─────────────────────────────────────────────

    async def _handle_send_email(self, params: dict[str, Any]) -> GmailToolResult:
        to = validate_addresses(params.get("to") or params.get("recipient"), "to")
        if not to:
            raise ValidationError("'to' is required")
        cc = validate_addresses(params.get("cc"), "cc")
        bcc = validate_addresses(params.get("bcc"), "bcc")
        subject = validate_text(
            params.get("subject"), "subject", MAX_SUBJECT_CHARS, required=False
        )
        body = validate_text(
            params.get("body") or params.get("message"),
            "body",
            MAX_SEND_BODY_CHARS,
            required=True,
        )

        result = await self.client.send_message(
            to=to, subject=subject, body=body, cc=cc, bcc=bcc
        )
        if not result.success:
            return self._failure(result, "send_email")

        recipients = ", ".join(to)
        identifier = result.data.get("message_id") or result.data.get(
            "rfc822_message_id", ""
        )
        return GmailToolResult(
            success=True,
            action="send_email",
            output=f"Email sent to {recipients} (id={identifier}).",
            data=result.data,
        )

    async def _handle_mark_read(self, params: dict[str, Any]) -> GmailToolResult:
        return await self._flag_op(params, "mark_read", "marked as read")

    async def _handle_mark_unread(self, params: dict[str, Any]) -> GmailToolResult:
        return await self._flag_op(params, "mark_unread", "marked as unread")

    async def _handle_star_email(self, params: dict[str, Any]) -> GmailToolResult:
        return await self._flag_op(params, "star", "starred", action="star_email")

    async def _handle_unstar_email(self, params: dict[str, Any]) -> GmailToolResult:
        return await self._flag_op(params, "unstar", "unstarred", action="unstar_email")

    async def _handle_archive_email(self, params: dict[str, Any]) -> GmailToolResult:
        return await self._flag_op(
            params, "archive", "archived", action="archive_email"
        )

    async def _handle_trash_email(self, params: dict[str, Any]) -> GmailToolResult:
        return await self._flag_op(
            params, "trash", "moved to Trash", action="trash_email"
        )

    async def _flag_op(
        self,
        params: dict[str, Any],
        method: str,
        past_tense: str,
        action: str | None = None,
    ) -> GmailToolResult:
        action = action or method
        message_id = validate_message_id(params.get("message_id") or params.get("id"))
        result = await getattr(self.client, method)(message_id)
        if not result.success:
            return self._failure(result, action)

        subject = result.data.get("subject") or message_id
        return GmailToolResult(
            success=True,
            action=action,
            output=f"'{subject}' {past_tense}.",
            data=result.data,
        )

    # ── failure rendering ─────────────────────────────────────────────

    @staticmethod
    def _failure(result: GmailResult, action: str) -> GmailToolResult:
        """Turn a client error into something the LLM can act on."""
        hints = {
            "not_configured": "Gmail is not set up — GMAIL_USER and GMAIL_APP_PASSWORD must be in .env.",
            "auth_failed": "Gmail rejected the credentials.",
            "network_error": "Could not reach Gmail.",
            "rate_limited": "Gmail is rate limiting this account; try again shortly.",
            "not_found": "That message no longer exists.",
        }
        prefix = hints.get(result.error_kind or "", "")
        message = f"{prefix} {result.error}".strip() if prefix else (result.error or "")
        return GmailToolResult(
            success=False,
            action=action,
            error=message or "Gmail operation failed.",
            data={"error_kind": result.error_kind} if result.error_kind else {},
        )


# ── Shared instance ───────────────────────────────────────────────────────

_shared_tool: GmailTool | None = None


def get_shared_gmail_tool() -> GmailTool:
    """The process-wide tool, so armed confirmations survive between turns.

    A per-request instance would forget every armed approval, making the
    second phase of the gate unreachable — the user could never confirm.
    """
    global _shared_tool
    if _shared_tool is None:
        _shared_tool = GmailTool()
    return _shared_tool


def set_shared_gmail_tool(tool: GmailTool | None) -> None:
    """Swap the shared tool. For tests and for wiring at startup."""
    global _shared_tool
    _shared_tool = tool
