"""Gmail client — authentication, transport, validation, execution.

This layer knows Gmail and nothing else. It has no idea an LLM exists:
it takes typed arguments, talks to Google, and returns
:class:`GmailResult`. Deciding *which* operation to run belongs to
:mod:`actions.gmail_tool` and the Task LLM above it.

Transport: IMAP + SMTP, not the REST API
----------------------------------------
The credentials configured in ``.env`` are ``GMAIL_USER`` and
``GMAIL_APP_PASSWORD`` — a Google *app password*. That is an IMAP/SMTP
credential; it cannot sign REST calls, which need an OAuth access token
from a consent flow this project has never run. So the operations below
go over ``imaplib``/``smtplib`` (stdlib, no new dependencies).

Every capability the tool exposes survives that choice, because Gmail's
IMAP server implements the ``X-GM-EXT-1`` extensions:

===================  ==========================================
Capability           IMAP mechanism
===================  ==========================================
Gmail search syntax  ``UID SEARCH X-GM-RAW "<query>"``
Stable message id    ``X-GM-MSGID`` (64-bit, exposed as hex)
Archive              ``UID STORE -X-GM-LABELS (\\Inbox)``
Trash                ``UID STORE +X-GM-LABELS (\\Trash)``
Star / unstar        ``UID STORE ±FLAGS (\\Flagged)``
Read / unread        ``UID STORE ±FLAGS (\\Seen)``
===================  ==========================================

Archive and trash go through *labels* rather than folder copies on
purpose: ``[Gmail]/Trash`` is localised per account, and expunge
semantics depend on a per-account setting. Label edits are neither.

Reads use ``BODY.PEEK[...]`` throughout, so listing or reading a message
never silently marks it as seen — flipping ``\\Seen`` is its own
explicit, confirmed operation.
"""

from __future__ import annotations

import asyncio
import email.utils
import imaplib
import logging
import os
import re
import smtplib
import socket
import ssl
import time
from dataclasses import dataclass, field
from email.header import decode_header, make_header
from email.message import EmailMessage as MIMEMessage
from email.parser import BytesParser
from email.policy import default as default_policy
from typing import Any, Iterable

logger = logging.getLogger("kancha.actions.gmail_client")

IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465

# Network wall clock. Gmail is usually sub-second; anything past this is
# a hung socket, and a hung socket must not wedge the assistant's turn.
DEFAULT_TIMEOUT_S: float = 20.0

# Never hand an unbounded mailbox to the layer above — the result is
# eventually rendered into an LLM prompt.
MAX_LIMIT = 50
DEFAULT_LIMIT = 10

# Caps applied to anything that leaves this module.
MAX_BODY_CHARS = 8000
SNIPPET_CHARS = 180

# Caps applied to anything sent outward.
MAX_SUBJECT_CHARS = 500
MAX_SEND_BODY_CHARS = 100_000
MAX_RECIPIENTS = 25

_ADDRESS_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_MSGID_RE = re.compile(r"^[0-9a-fA-F]{1,16}$")

# Sentinel mailbox: "the All Mail folder, whatever it is called here".
# Its real name is localised — a Nepali account has [Gmail]/सबै मेल — so it
# is resolved at connect time from the RFC 6154 \All special-use flag
# rather than hardcoded.
ALL_MAIL = "\\All"

# `(\All \HasNoChildren) "/" "[Gmail]/All Mail"`
_LIST_RE = re.compile(r'^\((?P<flags>[^)]*)\)\s+"(?P<sep>[^"]*)"\s+(?P<name>.+)$')

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")


# ── Errors ────────────────────────────────────────────────────────────────


class GmailError(Exception):
    """Base class. Carries a machine-readable ``kind`` for the tool layer."""

    kind = "gmail_error"


class ConfigurationError(GmailError):
    """Credentials are missing or malformed in the environment."""

    kind = "not_configured"


class AuthenticationError(GmailError):
    """Google rejected the credentials."""

    kind = "auth_failed"


class NetworkError(GmailError):
    """DNS, TCP, or TLS failure reaching Google."""

    kind = "network_error"


class RateLimitError(GmailError):
    """Google is throttling this account."""

    kind = "rate_limited"


class NotFoundError(GmailError):
    """No message matches the given id."""

    kind = "not_found"


class ValidationError(GmailError):
    """An argument failed validation. Never reaches the network."""

    kind = "invalid_argument"


# ── Credentials ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GmailCredentials:
    """The account this client acts as.

    ``password`` is deliberately excluded from ``__repr__``. This object
    is reachable from the task layer, whose parameter dicts get logged
    and rendered into prompts; a default dataclass repr would put the
    app password in both.
    """

    address: str
    password: str = field(repr=False)

    @property
    def redacted(self) -> str:
        return f"{self.address} (app password: ****)"


def load_credentials(env: dict[str, str] | None = None) -> GmailCredentials:
    """Read Gmail credentials from the environment.

    Raises :class:`ConfigurationError` rather than returning a partial
    object, so a missing password can never be sent as an empty string.
    """
    source = env if env is not None else os.environ
    address = (source.get("GMAIL_USER") or source.get("GMAIL_ADDRESS") or "").strip()
    password = (
        source.get("GMAIL_APP_PASSWORD") or source.get("GMAIL_PASSWORD") or ""
    ).strip()

    missing = [
        name
        for name, value in (("GMAIL_USER", address), ("GMAIL_APP_PASSWORD", password))
        if not value
    ]
    if missing:
        raise ConfigurationError(
            "Gmail is not configured: missing " + ", ".join(missing) + " in .env"
        )
    if not _ADDRESS_RE.match(address):
        raise ConfigurationError(f"GMAIL_USER is not a valid address: {address!r}")

    # Google prints app passwords in groups of four; users paste them
    # that way and IMAP rejects the spaces with a bare "Invalid
    # credentials", which is a miserable thing to debug.
    return GmailCredentials(address=address, password=password.replace(" ", ""))


# ── Results ───────────────────────────────────────────────────────────────


@dataclass(slots=True)
class EmailMessage:
    """One message, flattened for the layer above."""

    message_id: str
    sender: str = ""
    recipient: str = ""
    cc: str = ""
    subject: str = ""
    timestamp: str = ""
    snippet: str = ""
    body: str = ""
    unread: bool = False
    starred: bool = False
    attachments: tuple[str, ...] = ()

    def to_dict(self, include_body: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "message_id": self.message_id,
            "from": self.sender,
            "to": self.recipient,
            "subject": self.subject,
            "timestamp": self.timestamp,
            "snippet": self.snippet,
            "unread": self.unread,
            "starred": self.starred,
        }
        if self.cc:
            data["cc"] = self.cc
        if self.attachments:
            # Names only. Attachment *content* never leaves this module.
            data["attachments"] = list(self.attachments)
        if include_body:
            data["body"] = self.body
        return data


@dataclass(slots=True)
class GmailResult:
    """Structured outcome. Never raises past the client boundary."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_kind: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "error_kind": self.error_kind,
        }


# ── Validation ────────────────────────────────────────────────────────────


def validate_address(value: Any, field_name: str = "address") -> str:
    """Return a single well-formed address, or raise."""
    raw = str(value or "").strip()
    if not raw:
        raise ValidationError(f"'{field_name}' is required")
    _, addr = email.utils.parseaddr(raw)
    addr = addr.strip()
    if not addr or not _ADDRESS_RE.match(addr):
        raise ValidationError(f"'{field_name}' is not a valid email address: {raw!r}")
    # A newline in a header lets a caller inject extra headers (BCC to a
    # third party, a forged Reply-To). parseaddr does not strip them.
    if any(ch in addr for ch in "\r\n"):
        raise ValidationError(f"'{field_name}' contains a line break")
    return addr


def validate_addresses(value: Any, field_name: str) -> list[str]:
    """Normalise one address or a list/comma-separated string of them."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        candidates: Iterable[Any] = [p for p in value.split(",") if p.strip()]
    elif isinstance(value, (list, tuple)):
        candidates = value
    else:
        raise ValidationError(f"'{field_name}' must be an address or a list of them")

    addresses = [validate_address(item, field_name) for item in candidates]
    if len(addresses) > MAX_RECIPIENTS:
        raise ValidationError(
            f"'{field_name}' has {len(addresses)} recipients; the limit is {MAX_RECIPIENTS}"
        )
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    unique = []
    for addr in addresses:
        low = addr.lower()
        if low not in seen:
            seen.add(low)
            unique.append(addr)
    return unique


def validate_message_id(value: Any) -> str:
    """Gmail message ids are hex renderings of a 64-bit X-GM-MSGID."""
    raw = str(value or "").strip()
    if not raw:
        raise ValidationError("'message_id' is required")
    if not _MSGID_RE.match(raw):
        raise ValidationError(
            f"'message_id' must be a Gmail message id (hex); got {raw!r}"
        )
    return raw.lower()


def validate_limit(value: Any) -> int:
    if value in (None, ""):
        return DEFAULT_LIMIT
    try:
        limit = int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"'limit' must be a whole number; got {value!r}") from None
    if not 1 <= limit <= MAX_LIMIT:
        raise ValidationError(f"'limit' must be between 1 and {MAX_LIMIT}; got {limit}")
    return limit


def validate_offset(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        offset = int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"'offset' must be a whole number; got {value!r}") from None
    if offset < 0:
        raise ValidationError(f"'offset' must not be negative; got {offset}")
    return offset


def validate_query(value: Any) -> str:
    """A Gmail search expression, e.g. ``from:alice is:unread newer_than:2d``."""
    query = str(value or "").strip()
    if not query:
        raise ValidationError("'query' is required")
    if len(query) > 500:
        raise ValidationError("'query' is too long (max 500 characters)")
    if any(ch in query for ch in "\r\n"):
        raise ValidationError("'query' must be a single line")
    return query


def validate_text(value: Any, field_name: str, max_chars: int, required: bool) -> str:
    text = str(value if value is not None else "")
    if required and not text.strip():
        raise ValidationError(f"'{field_name}' is required")
    if len(text) > max_chars:
        raise ValidationError(
            f"'{field_name}' is too long ({len(text)} chars; max {max_chars})"
        )
    return text


# ── Helpers ───────────────────────────────────────────────────────────────


def _decode_header_value(raw: str | None) -> str:
    """Decode RFC 2047 (``=?utf-8?B?...?=``) into plain text."""
    if not raw:
        return ""
    try:
        return str(make_header(decode_header(raw))).strip()
    except (UnicodeDecodeError, LookupError, ValueError):
        return raw.strip()


def _clean(text: str) -> str:
    text = _WS_RE.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [line.strip() for line in text.split("\n")]
    out: list[str] = []
    blank = 0
    for line in lines:
        if line:
            blank = 0
            out.append(line)
        else:
            blank += 1
            if blank <= 1:
                out.append("")
    return "\n".join(out).strip()


def _quote_imap(value: str) -> str:
    """Quote a string for an IMAP command literal."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _classify_imap_error(exc: Exception) -> GmailError:
    """Map a transport exception onto our typed error family."""
    text = str(exc).lower()
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return NetworkError("Gmail did not respond in time")
    if isinstance(exc, (socket.gaierror, ConnectionError, ssl.SSLError, OSError)):
        return NetworkError(f"could not reach Gmail: {exc}")
    if "authenticationfailed" in text or "invalid credentials" in text:
        return AuthenticationError(
            "Gmail rejected the credentials — check GMAIL_APP_PASSWORD "
            "(it must be an app password, and IMAP must be enabled)"
        )
    if "too many" in text or "rate" in text or "limit exceeded" in text:
        return RateLimitError(f"Gmail is throttling this account: {exc}")
    return GmailError(str(exc) or exc.__class__.__name__)


# ── The client ────────────────────────────────────────────────────────────


class GmailClient:
    """Talks to Gmail over IMAP/SMTP. Stateless between calls.

    Each operation opens a connection, does its work, and logs out. That
    costs a TLS handshake per call and buys not having to reason about a
    socket that Google dropped while the assistant was idle — which, for
    an assistant that goes minutes between requests, is the common case.
    """

    def __init__(
        self,
        credentials: GmailCredentials | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._credentials = credentials
        self._timeout = timeout
        # Resolved All Mail folder name, cached once discovered.
        self._all_mail: str | None = None

    # ── auth ──────────────────────────────────────────────────────────

    @property
    def credentials(self) -> GmailCredentials:
        if self._credentials is None:
            self._credentials = load_credentials()
        return self._credentials

    @property
    def address(self) -> str:
        return self.credentials.address

    def _imap(self) -> imaplib.IMAP4_SSL:
        creds = self.credentials
        try:
            connection = imaplib.IMAP4_SSL(
                IMAP_HOST, IMAP_PORT, timeout=self._timeout
            )
        except Exception as exc:  # noqa: BLE001
            raise _classify_imap_error(exc) from exc
        try:
            connection.login(creds.address, creds.password)
        except imaplib.IMAP4.error as exc:
            try:
                connection.logout()
            except Exception:  # noqa: BLE001
                pass
            raise _classify_imap_error(exc) from exc
        except Exception as exc:  # noqa: BLE001
            raise _classify_imap_error(exc) from exc
        return connection

    # ── low-level IMAP ────────────────────────────────────────────────

    @staticmethod
    def _ok(status: str, data: Any, what: str) -> Any:
        if status != "OK":
            raise GmailError(f"{what} failed: {status} {data!r}")
        return data

    def _resolve_mailbox(self, connection: imaplib.IMAP4_SSL, mailbox: str) -> str:
        """Turn the :data:`ALL_MAIL` sentinel into this account's real name."""
        if mailbox != ALL_MAIL:
            return mailbox
        cached = self._all_mail
        if cached:
            return cached

        status, data = connection.list()
        if status == "OK":
            for raw in data or []:
                line = (
                    raw.decode("utf-8", errors="replace")
                    if isinstance(raw, bytes)
                    else str(raw)
                )
                match = _LIST_RE.match(line.strip())
                if match and "\\All" in match.group("flags"):
                    resolved = match.group("name").strip().strip('"')
                    self._all_mail = resolved
                    return resolved

        # Untranslated default. Better than failing outright, since most
        # accounts are English and this is only reached if LIST misbehaves.
        self._all_mail = "[Gmail]/All Mail"
        return "[Gmail]/All Mail"

    def _select(self, connection: imaplib.IMAP4_SSL, mailbox: str) -> None:
        name = self._resolve_mailbox(connection, mailbox)
        # Quote unconditionally: Gmail's folder names contain spaces
        # ("[Gmail]/All Mail"), and an unquoted SELECT of those comes back
        # as `BAD Could not parse command`.
        status, data = connection.select(_quote_imap(name), readonly=False)
        if status != "OK":
            raise GmailError(f"could not open mailbox {name}: {data!r}")

    def _search_uids(
        self, connection: imaplib.IMAP4_SSL, criteria: str
    ) -> list[int]:
        status, data = connection.uid("SEARCH", None, criteria)  # type: ignore[arg-type]
        self._ok(status, data, "search")
        blob = (data[0] or b"").decode("ascii", errors="ignore")
        return [int(part) for part in blob.split() if part.isdigit()]

    def _uid_for_message_id(
        self, connection: imaplib.IMAP4_SSL, message_id: str
    ) -> int:
        """Resolve a Gmail message id (hex) to a UID in the current mailbox."""
        decimal = int(message_id, 16)
        uids = self._search_uids(connection, f"X-GM-MSGID {decimal}")
        if not uids:
            raise NotFoundError(f"no message with id {message_id}")
        return uids[-1]

    _ENVELOPE_PARTS = (
        "(UID X-GM-MSGID FLAGS INTERNALDATE "
        "BODY.PEEK[HEADER.FIELDS (FROM TO CC SUBJECT DATE)])"
    )

    def _fetch_summaries(
        self, connection: imaplib.IMAP4_SSL, uids: list[int]
    ) -> list[EmailMessage]:
        if not uids:
            return []
        uid_set = ",".join(str(uid) for uid in uids)
        status, data = connection.uid("FETCH", uid_set, self._ENVELOPE_PARTS)
        self._ok(status, data, "fetch")

        messages: dict[int, EmailMessage] = {}
        for item in data or []:
            if not isinstance(item, tuple) or len(item) < 2:
                continue
            meta = item[0].decode("utf-8", errors="replace")
            headers = BytesParser(policy=default_policy).parsebytes(item[1])

            uid_match = re.search(r"UID (\d+)", meta)
            msgid_match = re.search(r"X-GM-MSGID (\d+)", meta)
            flags_match = re.search(r"FLAGS \(([^)]*)\)", meta)
            date_match = re.search(r'INTERNALDATE "([^"]+)"', meta)
            if not uid_match or not msgid_match:
                continue

            flags = (flags_match.group(1) if flags_match else "").split()
            messages[int(uid_match.group(1))] = EmailMessage(
                message_id=format(int(msgid_match.group(1)), "x"),
                sender=_decode_header_value(headers.get("From")),
                recipient=_decode_header_value(headers.get("To")),
                cc=_decode_header_value(headers.get("Cc")),
                subject=_decode_header_value(headers.get("Subject")) or "(no subject)",
                timestamp=(
                    date_match.group(1)
                    if date_match
                    else _decode_header_value(headers.get("Date"))
                ),
                unread="\\Seen" not in flags,
                starred="\\Flagged" in flags,
            )
        # Preserve the caller's ordering; IMAP returns whatever it likes.
        return [messages[uid] for uid in uids if uid in messages]

    def _fetch_full(
        self, connection: imaplib.IMAP4_SSL, uid: int, message: EmailMessage
    ) -> EmailMessage:
        status, data = connection.uid("FETCH", str(uid), "(BODY.PEEK[])")
        self._ok(status, data, "fetch body")
        raw = b""
        for item in data or []:
            if isinstance(item, tuple) and len(item) >= 2:
                raw = item[1]
                break
        if not raw:
            return message

        parsed = BytesParser(policy=default_policy).parsebytes(raw)
        body, attachments = self._extract_body(parsed)
        message.body = body[:MAX_BODY_CHARS]
        if len(body) > MAX_BODY_CHARS:
            message.body += "\n… (truncated)"
        message.attachments = attachments
        message.snippet = _clean(body)[:SNIPPET_CHARS]
        return message

    @staticmethod
    def _extract_body(parsed: Any) -> tuple[str, tuple[str, ...]]:
        """Prefer text/plain; fall back to stripped HTML.

        Attachment *names* are collected for context. Their bytes are
        never decoded, returned, or written anywhere — the tool has no
        download operation, so nothing untrusted lands on the disk.
        """
        attachments: list[str] = []
        plain: list[str] = []
        html: list[str] = []

        for part in parsed.walk() if parsed.is_multipart() else [parsed]:
            disposition = (part.get_content_disposition() or "").lower()
            if disposition == "attachment":
                name = part.get_filename()
                attachments.append(_decode_header_value(name) if name else "(unnamed)")
                continue
            content_type = part.get_content_type()
            if content_type not in ("text/plain", "text/html"):
                continue
            try:
                text = part.get_content()
            except (LookupError, UnicodeDecodeError, ValueError):
                payload = part.get_payload(decode=True) or b""
                text = payload.decode("utf-8", errors="replace")
            if content_type == "text/plain":
                plain.append(text)
            else:
                html.append(text)

        if plain:
            return _clean("\n".join(plain)), tuple(attachments)
        if html:
            stripped = _HTML_TAG_RE.sub(" ", "\n".join(html))
            return _clean(stripped), tuple(attachments)
        return "", tuple(attachments)

    def _store(
        self,
        connection: imaplib.IMAP4_SSL,
        uid: int,
        item: str,
        value: str,
    ) -> None:
        status, data = connection.uid("STORE", str(uid), item, value)
        self._ok(status, data, f"store {item} {value}")

    # ── blocking operations ───────────────────────────────────────────

    def _run(self, operation: str, mailbox: str, work: Any) -> Any:
        """Open, select, run, log out — with typed errors on the way out."""
        connection: imaplib.IMAP4_SSL | None = None
        try:
            connection = self._imap()
            self._select(connection, mailbox)
            return work(connection)
        except GmailError:
            raise
        except imaplib.IMAP4.error as exc:
            raise _classify_imap_error(exc) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Gmail: %s failed", operation)
            raise _classify_imap_error(exc) from exc
        finally:
            if connection is not None:
                try:
                    connection.logout()
                except Exception:  # noqa: BLE001
                    pass

    def _list_blocking(
        self, query: str | None, limit: int, offset: int, mailbox: str
    ) -> dict[str, Any]:
        def work(connection: imaplib.IMAP4_SSL) -> dict[str, Any]:
            criteria = f"X-GM-RAW {_quote_imap(query)}" if query else "ALL"
            uids = self._search_uids(connection, criteria)
            # IMAP returns ascending UIDs; newest first is what a person means.
            uids.reverse()
            total = len(uids)
            window = uids[offset : offset + limit]
            messages = self._fetch_summaries(connection, window)
            for message in messages:
                message.snippet = message.snippet or ""
            return {
                "messages": [m.to_dict(include_body=False) for m in messages],
                "count": len(messages),
                "total_matched": total,
                "offset": offset,
                "limit": limit,
                "has_more": offset + len(messages) < total,
                "query": query or "",
                "mailbox": mailbox,
            }

        return self._run("list", mailbox, work)

    def _get_blocking(self, message_id: str, mailbox: str) -> dict[str, Any]:
        def work(connection: imaplib.IMAP4_SSL) -> dict[str, Any]:
            uid = self._uid_for_message_id(connection, message_id)
            summaries = self._fetch_summaries(connection, [uid])
            if not summaries:
                raise NotFoundError(f"no message with id {message_id}")
            message = self._fetch_full(connection, uid, summaries[0])
            return {"message": message.to_dict(include_body=True)}

        return self._run("get", mailbox, work)

    def _modify_blocking(
        self, message_id: str, item: str, value: str, mailbox: str, verb: str
    ) -> dict[str, Any]:
        def work(connection: imaplib.IMAP4_SSL) -> dict[str, Any]:
            uid = self._uid_for_message_id(connection, message_id)
            summaries = self._fetch_summaries(connection, [uid])
            subject = summaries[0].subject if summaries else ""
            self._store(connection, uid, item, value)
            return {
                "message_id": message_id,
                "subject": subject,
                "applied": verb,
            }

        return self._run(verb, mailbox, work)

    def _send_blocking(
        self,
        to: list[str],
        subject: str,
        body: str,
        cc: list[str],
        bcc: list[str],
    ) -> dict[str, Any]:
        creds = self.credentials
        message = MIMEMessage()
        message["From"] = creds.address
        message["To"] = ", ".join(to)
        if cc:
            message["Cc"] = ", ".join(cc)
        message["Subject"] = subject
        rfc_id = email.utils.make_msgid(domain=creds.address.split("@", 1)[-1])
        message["Message-ID"] = rfc_id
        message["Date"] = email.utils.formatdate(localtime=True)
        message.set_content(body)

        recipients = list(to) + list(cc) + list(bcc)
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(
                SMTP_HOST, SMTP_PORT, timeout=self._timeout, context=context
            ) as server:
                server.login(creds.address, creds.password)
                server.send_message(message, from_addr=creds.address, to_addrs=recipients)
        except smtplib.SMTPAuthenticationError as exc:
            raise AuthenticationError(
                "Gmail rejected the credentials when sending — check "
                "GMAIL_APP_PASSWORD"
            ) from exc
        except smtplib.SMTPRecipientsRefused as exc:
            raise ValidationError(f"Gmail refused the recipients: {exc.recipients}") from exc
        except smtplib.SMTPResponseException as exc:
            if exc.smtp_code in (421, 450, 451, 452, 454):
                raise RateLimitError(
                    f"Gmail is throttling or temporarily unavailable ({exc.smtp_code})"
                ) from exc
            raise GmailError(f"Gmail refused the message ({exc.smtp_code})") from exc
        except (socket.timeout, TimeoutError) as exc:
            raise NetworkError("Gmail did not respond while sending") from exc
        except (socket.gaierror, ConnectionError, ssl.SSLError, OSError) as exc:
            raise NetworkError(f"could not reach Gmail to send: {exc}") from exc

        result = {
            "status": "sent",
            "rfc822_message_id": rfc_id,
            "to": to,
            "cc": cc,
            "bcc_count": len(bcc),
            "subject": subject,
        }
        # SMTP does not report Gmail's own id, so look it up in the
        # mailbox. Best effort: the send already succeeded, and a failed
        # lookup must not turn that into a reported failure.
        gmail_id = self._lookup_sent_id(rfc_id)
        if gmail_id:
            result["message_id"] = gmail_id
        return result

    def _lookup_sent_id(self, rfc_id: str) -> str | None:
        stripped = rfc_id.strip("<>")
        for delay in (0.0, 1.5):
            if delay:
                time.sleep(delay)
            try:

                def work(connection: imaplib.IMAP4_SSL) -> list[int]:
                    return self._search_uids(
                        connection,
                        f"X-GM-RAW {_quote_imap(f'rfc822msgid:{stripped}')}",
                    )

                uids = self._run("lookup", ALL_MAIL, work)
                if uids:
                    def fetch(connection: imaplib.IMAP4_SSL) -> list[EmailMessage]:
                        return self._fetch_summaries(connection, uids[-1:])

                    found = self._run("lookup", ALL_MAIL, fetch)
                    if found:
                        return found[0].message_id
            except GmailError as exc:
                logger.debug("Gmail: sent-id lookup failed (%s)", exc)
                return None
        return None

    # ── public async surface ──────────────────────────────────────────

    async def _call(self, fn: Any, *args: Any) -> GmailResult:
        """Run a blocking operation off the event loop, typed errors mapped."""
        try:
            data = await asyncio.to_thread(fn, *args)
            return GmailResult(success=True, data=data)
        except GmailError as exc:
            logger.info("Gmail: %s — %s", exc.kind, exc)
            return GmailResult(success=False, error=str(exc), error_kind=exc.kind)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Gmail: unexpected failure")
            return GmailResult(
                success=False, error=str(exc), error_kind="gmail_error"
            )

    async def list_messages(
        self,
        query: str | None = None,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
        mailbox: str = "INBOX",
    ) -> GmailResult:
        """Recent messages, newest first, optionally filtered by a query."""
        return await self._call(self._list_blocking, query, limit, offset, mailbox)

    async def search_messages(
        self, query: str, limit: int = DEFAULT_LIMIT, offset: int = 0
    ) -> GmailResult:
        """Gmail search across the whole account, not just the inbox."""
        return await self._call(
            self._list_blocking, query, limit, offset, ALL_MAIL
        )

    async def get_message(
        self, message_id: str, mailbox: str = ALL_MAIL
    ) -> GmailResult:
        """One message with its body. Does not mark it as read."""
        return await self._call(self._get_blocking, message_id, mailbox)

    async def send_message(
        self,
        to: list[str],
        subject: str,
        body: str,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
    ) -> GmailResult:
        return await self._call(
            self._send_blocking, to, subject, body, cc or [], bcc or []
        )

    async def mark_read(self, message_id: str) -> GmailResult:
        return await self._call(
            self._modify_blocking,
            message_id,
            "+FLAGS",
            "(\\Seen)",
            ALL_MAIL,
            "mark_read",
        )

    async def mark_unread(self, message_id: str) -> GmailResult:
        return await self._call(
            self._modify_blocking,
            message_id,
            "-FLAGS",
            "(\\Seen)",
            ALL_MAIL,
            "mark_unread",
        )

    async def star(self, message_id: str) -> GmailResult:
        return await self._call(
            self._modify_blocking,
            message_id,
            "+FLAGS",
            "(\\Flagged)",
            ALL_MAIL,
            "star",
        )

    async def unstar(self, message_id: str) -> GmailResult:
        return await self._call(
            self._modify_blocking,
            message_id,
            "-FLAGS",
            "(\\Flagged)",
            ALL_MAIL,
            "unstar",
        )

    async def archive(self, message_id: str) -> GmailResult:
        """Remove the Inbox label. The message stays in All Mail."""
        return await self._call(
            self._modify_blocking,
            message_id,
            "-X-GM-LABELS",
            "(\\Inbox)",
            "INBOX",
            "archive",
        )

    async def trash(self, message_id: str) -> GmailResult:
        """Move to Trash, where Gmail deletes it after 30 days."""
        return await self._call(
            self._modify_blocking,
            message_id,
            "+X-GM-LABELS",
            "(\\Trash)",
            ALL_MAIL,
            "trash",
        )


# ── Shared instance ───────────────────────────────────────────────────────

_shared_client: GmailClient | None = None


def get_shared_gmail_client() -> GmailClient:
    """The process-wide client, created on first use."""
    global _shared_client
    if _shared_client is None:
        _shared_client = GmailClient()
    return _shared_client


def set_shared_gmail_client(client: GmailClient | None) -> None:
    """Swap the shared client. For tests and for wiring at startup."""
    global _shared_client
    _shared_client = client
