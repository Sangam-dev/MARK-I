"""OpenCode client — transport and server lifecycle. No LLM awareness.

This module speaks the OpenCode HTTP API and nothing else. It does not
know what a task is, which session belongs to which conversation, or that
an LLM exists anywhere in the process. :mod:`agent.tool` owns all of
that. The split is the same one :mod:`actions.gmail_client` and
:mod:`actions.gmail_tool` use, and for the same reason: the thing that
talks to a remote service should be testable without any notion of
prompts, turns, or delegation.

Which API
---------
The documented HTTP API, over ``httpx`` — already a project dependency —
rather than the ``opencode-ai`` SDK, which is at ``0.1.0a36`` and would
add an alpha dependency for what is four REST calls. The three endpoints
that matter, verified against a live 1.18.16 server:

* ``POST /session?directory=<cwd>`` → ``{"id": "ses_…"}``
* ``POST /session/{id}/message?directory=<cwd>`` → ``{"info": …, "parts": […]}``
  — synchronous: it returns once the agent has finished the turn.
* ``GET /api/health`` → ``{"healthy": true}``

Errors do not arrive as HTTP errors
-----------------------------------
This is the one surprising thing about the API and the reason
:meth:`OpenCodeClient.prompt` inspects the body before trusting the
status code. A model that has no credit, a context overflow, a provider
outage — all of these come back as **HTTP 200** with the failure sitting
in ``info.error``::

    {"info": {"error": {"name": "APIError",
                        "data": {"message": "No payment method…"}}},
     "parts": []}

Checking ``response.status_code`` alone would report those as successful
delegations with an empty result, which is exactly the failure mode the
"return failures cleanly" requirement is about.

Owning the server
-----------------
With no ``KANCHA_OPENCODE_URL`` configured the client spawns
``opencode serve`` itself and reads the real base URL back from the
server's own startup line. That readback is what makes port 0 (the
default) safe: OpenCode tries 4096 and falls back to a free port when it
is taken, so a second assistant on the same machine is never a collision.

The subprocess is a child of this process and inherits its environment,
which is how a non-default provider finds its credentials — OpenCode
discovers ``GEMINI_API_KEY`` and friends on its own. The default model
needs none of that: it runs on OpenCode's own free tier. Either way, no
key is ever read or sent by this module.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from agent.config import OpenCodeConfig

logger = logging.getLogger("kancha.agent.client")

# "opencode server listening on http://127.0.0.1:4599"
_LISTENING_RE = re.compile(r"listening on\s+(https?://\S+)", re.IGNORECASE)

# Connecting to a local server is instant or never; only the agent's own
# work is slow, and that is bounded by config.request_timeout_s.
_CONNECT_TIMEOUT_S = 5.0


class OpenCodeError(Exception):
    """Raised only inside this module; never crosses the client boundary."""


@dataclass(slots=True)
class OpenCodeResult:
    """Structured outcome. Never raises past the client boundary."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    #: Machine-readable failure class, so the tool layer can phrase a
    #: useful hint instead of echoing a transport message.
    error_kind: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "error_kind": self.error_kind,
        }


def _failure(kind: str, message: str) -> OpenCodeResult:
    return OpenCodeResult(success=False, error=message, error_kind=kind)


def _extract_error(info: dict[str, Any]) -> str:
    """Pull a human-readable message out of an ``info.error`` block."""
    error = info.get("error")
    if not isinstance(error, dict):
        return str(error) if error else ""
    data = error.get("data")
    message = ""
    if isinstance(data, dict):
        message = str(data.get("message") or "").strip()
    name = str(error.get("name") or "").strip()
    if message and name:
        return f"{name}: {message}"
    return message or name or "OpenCode reported an unspecified error."


def _http_error_message(body: Any, status: int) -> str:
    """OpenCode's error bodies are ``{"name": …, "data": {"message": …}}``."""
    if isinstance(body, dict):
        data = body.get("data")
        if isinstance(data, dict) and data.get("message"):
            return str(data["message"])
        if body.get("message"):
            return str(body["message"])
    return f"OpenCode returned HTTP {status}."


class OpenCodeClient:
    """Async transport to an OpenCode server, plus its lifecycle."""

    def __init__(self, config: OpenCodeConfig | None = None) -> None:
        self._config = config or OpenCodeConfig.from_env()
        self._base_url: str = self._config.server_url.rstrip("/")
        self._http: Any = None  # httpx.AsyncClient, imported lazily
        self._process: asyncio.subprocess.Process | None = None
        self._log_pump: asyncio.Task[None] | None = None
        # One lock so concurrent delegations cannot race into spawning two
        # servers, and so the second caller waits for the first's probe.
        self._ready_lock = asyncio.Lock()
        self._ready = False

    @property
    def config(self) -> OpenCodeConfig:
        return self._config

    @property
    def base_url(self) -> str:
        """The resolved server URL — empty until :meth:`ensure_ready`."""
        return self._base_url

    @property
    def owns_server(self) -> bool:
        """True when a server subprocess is ours to shut down."""
        return self._process is not None

    # ── Readiness ─────────────────────────────────────────────────────

    async def ensure_ready(self) -> OpenCodeResult:
        """Connect to (or start) a server. Idempotent, never raises."""
        if self._ready:
            return OpenCodeResult(True, {"base_url": self._base_url})

        async with self._ready_lock:
            if self._ready:  # settled while we waited for the lock
                return OpenCodeResult(True, {"base_url": self._base_url})

            if not self._config.enabled:
                return _failure(
                    "disabled",
                    "The OpenCode layer is disabled (KANCHA_OPENCODE_ENABLED=0).",
                )

            try:
                if self._config.spawns_server:
                    await self._spawn_server()
                elif not await self._healthy():
                    return _failure(
                        "unavailable",
                        f"No OpenCode server answered at {self._base_url}.",
                    )
            except OpenCodeError as exc:
                await self._stop_server()
                return _failure(str(exc.args[1]) if len(exc.args) > 1 else "unavailable", str(exc.args[0]))
            except Exception as exc:  # noqa: BLE001
                logger.exception("OpenCode startup failed")
                await self._stop_server()
                return _failure("unavailable", f"Could not start OpenCode: {exc}")

            self._ready = True
            logger.info("OpenCode ready at %s (%s)", self._base_url, self._config.describe())
            return OpenCodeResult(True, {"base_url": self._base_url})

    async def _spawn_server(self) -> None:
        """Start ``opencode serve`` and read its real URL back."""
        binary = shutil.which(self._config.binary)
        if binary is None:
            raise OpenCodeError(
                f"The '{self._config.binary}' command is not installed or not on PATH.",
                "not_installed",
            )

        self._config.workspace.mkdir(parents=True, exist_ok=True)

        # argv list, never a shell string — the workspace path is
        # user-configurable and must not be word-split or interpreted.
        argv = [
            binary,
            "serve",
            "--hostname",
            self._config.host,
            "--port",
            str(self._config.port),
        ]
        logger.info("Starting OpenCode server: %s", " ".join(argv))
        self._process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(self._config.workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        try:
            url = await asyncio.wait_for(
                self._read_listening_url(), timeout=self._config.startup_timeout_s
            )
        except asyncio.TimeoutError:
            raise OpenCodeError(
                f"OpenCode did not report a listening address within "
                f"{self._config.startup_timeout_s:.0f}s.",
                "timeout",
            ) from None

        self._base_url = url.rstrip("/")
        # Drain the rest of the server's output for the lifetime of the
        # process. Without this the pipe fills and the server blocks on
        # its own logging — a hang that looks like a hung agent.
        self._log_pump = asyncio.create_task(self._pump_logs(), name="opencode_logs")

        if not await self._healthy():
            raise OpenCodeError(
                f"OpenCode started at {self._base_url} but failed its health check.",
                "unavailable",
            )

    async def _read_listening_url(self) -> str:
        """Consume startup output until the server announces its address."""
        assert self._process is not None and self._process.stdout is not None
        while True:
            line = await self._process.stdout.readline()
            if not line:
                code = self._process.returncode
                raise OpenCodeError(
                    f"The OpenCode server exited during startup (code {code}).",
                    "unavailable",
                )
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                logger.debug("opencode: %s", text)
            match = _LISTENING_RE.search(text)
            if match:
                return match.group(1)

    async def _pump_logs(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    return
                logger.debug("opencode: %s", line.decode("utf-8", errors="replace").rstrip())
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            return

    async def _healthy(self) -> bool:
        try:
            response = await self._client().get(
                f"{self._base_url}/api/health", timeout=_CONNECT_TIMEOUT_S
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("OpenCode health check failed: %s", exc)
            return False
        return response.status_code == 200

    def _client(self) -> Any:
        """Lazily built httpx client (imported like the ollama embedder)."""
        if self._http is None:
            try:
                import httpx  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - httpx is a dep
                raise OpenCodeError(
                    "httpx is required for the OpenCode integration.", "not_configured"
                ) from exc
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self._config.request_timeout_s, connect=_CONNECT_TIMEOUT_S
                )
            )
        return self._http

    # ── Sessions ──────────────────────────────────────────────────────

    async def create_session(
        self, title: str = "", directory: str = ""
    ) -> OpenCodeResult:
        """Create a session. ``data['session_id']`` on success.

        *directory* is where the agent will work. It defaults to the
        configured workspace; a caller passes one when the user named a
        location ("build it in ~/projects"), because OpenCode resolves
        every relative path in the task against this directory and
        nothing downstream can correct a wrong one.
        """
        ready = await self.ensure_ready()
        if not ready.success:
            return ready

        payload: dict[str, Any] = {}
        if title.strip():
            payload["title"] = title.strip()[:200]

        result = await self._post("/session", payload, directory=directory)
        if not result.success:
            return result

        session_id = str(result.data.get("id") or "")
        if not session_id:
            return _failure("unavailable", "OpenCode created a session with no id.")
        logger.info(
            "OpenCode session created: %s (%s) in %s",
            session_id,
            title or "untitled",
            directory or self._config.workspace,
        )
        return OpenCodeResult(
            True,
            {
                "session_id": session_id,
                "title": title,
                "directory": directory or str(self._config.workspace),
            },
        )

    async def prompt(
        self, session_id: str, text: str, directory: str = ""
    ) -> OpenCodeResult:
        """Send *text* to *session_id* and wait for the agent to finish.

        Tries the configured model first and then, when it returns an
        ``agent_error`` (a provider outage), each :attr:`fallback_models
        <agent.config.OpenCodeConfig.fallback_models>` in turn — the
        first attempt that neither succeeds nor errors this way wins.

        On success ``data`` carries ``text`` (the agent's reply),
        ``tools`` (the tools it used), ``tokens`` and ``cost``.

        This call is the whole run: it returns when the agent stops, and
        that is minutes. Callers who need the user to stay conversational
        run it as a background task and watch :meth:`stream_events` —
        see :class:`agent.tool.OpenCodeTool`.
        """
        ready = await self.ensure_ready()
        if not ready.success:
            return ready

        if not session_id.strip():
            return _failure("invalid_argument", "A session id is required.")
        if not text.strip():
            return _failure("invalid_argument", "There is nothing to send.")

        # The primary model, then any configured fallbacks, tried in order
        # when the previous one fails with an agent_error — the class of
        # failure a provider outage arrives as. A delegation is one prompt
        # on a fresh session, and a failed attempt produces no agent
        # output, so re-sending the same task under the next model is safe.
        models = (self._config.model, *self._config.fallback_models)
        last: OpenCodeResult | None = None
        for index, model_id in enumerate(models):
            result = await self._prompt_once(session_id, text, directory, model_id)
            if result.success or result.error_kind != "agent_error":
                return result
            last = result
            remaining = models[index + 1 :]
            if remaining:
                logger.warning(
                    "OpenCode model %s/%s failed in %s (%s) — trying %s/%s",
                    self._config.provider,
                    model_id,
                    session_id,
                    result.error,
                    self._config.provider,
                    remaining[0],
                )
            else:
                logger.warning(
                    "OpenCode model %s/%s failed in %s (%s) — no fallbacks left",
                    self._config.provider,
                    model_id,
                    session_id,
                    result.error,
                )
        assert last is not None
        return last

    async def _prompt_once(
        self, session_id: str, text: str, directory: str, model_id: str
    ) -> OpenCodeResult:
        """One model attempt: send *text* to *session_id* and wait.

        On success ``data`` carries ``text`` (the agent's reply),
        ``tools`` (the tools it used), ``tokens`` and ``cost``.

        This call is the whole run: it returns when the agent stops, and
        that is minutes. Callers who need the user to stay conversational
        run it as a background task and watch :meth:`stream_events` —
        see :class:`agent.tool.OpenCodeTool`.
        """
        payload = {
            "model": {
                "providerID": self._config.provider,
                "modelID": model_id,
            },
            "agent": self._config.agent,
            "parts": [{"type": "text", "text": text}],
        }
        result = await self._post(
            f"/session/{session_id}/message", payload, directory=directory
        )
        if not result.success:
            return result

        body = result.data
        info = body.get("info") if isinstance(body.get("info"), dict) else {}

        # HTTP 200 with an error inside — see the module docstring.
        if info.get("error"):
            message = _extract_error(info)
            logger.warning("OpenCode agent error in %s: %s", session_id, message)
            return _failure("agent_error", message)

        # Only the final assistant message comes back here, not the
        # intermediate steps: a run that wrote five files and executed a
        # test suite still returns just ``step-start / text / step-finish``.
        # The agent's own summary is the result — and a better one than a
        # list of tool names would be.
        parts = body.get("parts") if isinstance(body.get("parts"), list) else []
        chunks = [
            str(part["text"]).strip()
            for part in parts
            if isinstance(part, dict)
            and part.get("type") == "text"
            and str(part.get("text") or "").strip()
        ]

        return OpenCodeResult(
            True,
            {
                "session_id": session_id,
                "text": "\n\n".join(chunks),
                "tokens": info.get("tokens") or {},
                "cost": info.get("cost") or 0,
            },
        )

    async def abort(self, session_id: str, directory: str = "") -> OpenCodeResult:
        """Ask the agent to stop working on a session."""
        if not self._ready or not session_id.strip():
            return OpenCodeResult(True, {})
        return await self._post(
            f"/session/{session_id}/abort", {}, directory=directory, expect_json=False
        )

    # ── Permissions ───────────────────────────────────────────────────
    #
    # The agent stops and asks before touching anything outside its
    # working directory. Nothing answers by default, so the run simply
    # stops — the tool call sits in "running" and the prompt call never
    # returns. That is the whole of "it did nothing for a long time":
    # not a crash, not a slow model, a question no one heard.

    async def pending_permissions(self, directory: str = "") -> OpenCodeResult:
        """Requests currently waiting for an answer. ``data['requests']``."""
        return await self._pending("/permission", directory, "permissions")

    async def pending_questions(self, directory: str = "") -> OpenCodeResult:
        """Clarifying questions waiting for an answer. ``data['requests']``.

        A second way a run stops dead. The agent decides it needs to know
        something before it can continue, asks, and waits — the tool call
        stays "running" and the prompt call never returns, exactly as an
        unanswered permission does.
        """
        return await self._pending("/question", directory, "questions")

    async def _pending(
        self, path: str, directory: str, what: str
    ) -> OpenCodeResult:
        ready = await self.ensure_ready()
        if not ready.success:
            return ready

        url = f"{self._base_url}{path}"
        params = {"directory": directory or str(self._config.workspace)}
        try:
            response = await self._client().get(
                url, params=params, timeout=_CONNECT_TIMEOUT_S
            )
        except Exception as exc:  # noqa: BLE001
            return _failure("network_error", f"Could not read {what}: {exc}")
        if response.status_code >= 400:
            return _failure("unavailable", f"OpenCode returned HTTP {response.status_code}.")
        try:
            body = response.json()
        except Exception:  # noqa: BLE001
            return _failure("unavailable", "OpenCode returned an unreadable response.")
        requests = body if isinstance(body, list) else []
        return OpenCodeResult(True, {"requests": requests})

    async def answer_question(
        self, request_id: str, answers: list[list[str]], directory: str = ""
    ) -> OpenCodeResult:
        """Answer a clarifying question.

        *answers* is one entry per question asked, each a list of the
        option labels chosen — the server's own shape, not a
        convenience: a request can carry several questions and each can
        allow several selections.
        """
        if not request_id.strip():
            return _failure("invalid_argument", "A question id is required.")
        if not answers:
            return _failure("invalid_argument", "There is nothing to answer with.")
        return await self._post(
            f"/question/{request_id}/reply",
            {"answers": answers},
            directory=directory,
            expect_json=False,
        )

    async def reject_question(
        self, request_id: str, directory: str = ""
    ) -> OpenCodeResult:
        """Decline to answer, letting the agent decide for itself."""
        if not request_id.strip():
            return _failure("invalid_argument", "A question id is required.")
        return await self._post(
            f"/question/{request_id}/reject",
            {},
            directory=directory,
            expect_json=False,
        )

    async def reply_permission(
        self, request_id: str, reply: str = "once", directory: str = ""
    ) -> OpenCodeResult:
        """Answer one request: ``once``, ``always`` or ``reject``."""
        if reply not in {"once", "always", "reject"}:
            return _failure(
                "invalid_argument", f"reply must be once, always or reject; got {reply!r}"
            )
        if not request_id.strip():
            return _failure("invalid_argument", "A permission request id is required.")
        return await self._post(
            f"/permission/{request_id}/reply",
            {"reply": reply},
            directory=directory,
            expect_json=False,
        )

    # ── Events ────────────────────────────────────────────────────────

    async def stream_events(self, directory: str = "") -> AsyncIterator[dict[str, Any]]:
        """Yield events for work happening in *directory*, reconnecting.

        This is the only window into a run in flight. The blocking
        ``/session/{id}/message`` call says nothing until the agent
        stops, so without this stream "how is it going" has no answer
        but "still going".

        **The directory matters.** ``/event`` is scoped to it: a stream
        opened on the default workspace sees nothing at all from a
        session running in ``~/projects`` — verified against a live
        1.18.16 server, and silently, since the connection succeeds and
        simply stays empty. Callers pass the directory their session was
        created in.

        Each event names its ``properties.sessionID``, so one stream
        still covers every session sharing a directory. Ends only when
        the consumer stops (or its task is cancelled).
        """
        ready = await self.ensure_ready()
        if not ready.success:
            return

        backoff = 1.0
        while True:
            try:
                async for event in self._read_event_stream(directory):
                    backoff = 1.0
                    yield event
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug("OpenCode event stream dropped: %s", exc)

            # A dropped stream is not a failed run — the agent keeps
            # working server-side — so reconnect rather than give up,
            # backing off so a dead server is not hammered.
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    async def _read_event_stream(
        self, directory: str = ""
    ) -> AsyncIterator[dict[str, Any]]:
        """One SSE connection's worth of events."""
        import httpx  # noqa: PLC0415

        url = f"{self._base_url}/event"
        params = {"directory": directory or str(self._config.workspace)}
        # No read timeout: a quiet stream is normal — the agent may think
        # for minutes between tool calls — and a timeout here would look
        # like a dropped connection every time it did.
        timeout = httpx.Timeout(None, connect=_CONNECT_TIMEOUT_S)
        async with self._client().stream(
            "GET", url, params=params, timeout=timeout
        ) as response:
            if response.status_code != 200:
                raise OpenCodeError(
                    f"event stream returned HTTP {response.status_code}", "unavailable"
                )
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    yield event

    # ── Transport ─────────────────────────────────────────────────────

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
        directory: str = "",
        expect_json: bool = True,
    ) -> OpenCodeResult:
        """One POST, with every failure mode mapped to an error_kind."""
        url = f"{self._base_url}{path}"
        params = {"directory": directory or str(self._config.workspace)}
        try:
            response = await self._client().post(url, params=params, json=payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # httpx.TimeoutException subclasses TransportError; naming it
            # explicitly would mean importing httpx at module scope just
            # for an isinstance check.
            kind = "timeout" if "timeout" in type(exc).__name__.lower() else "network_error"
            if kind == "timeout":
                message = (
                    f"OpenCode did not finish within "
                    f"{self._config.request_timeout_s:.0f}s."
                )
            else:
                message = f"Could not reach the OpenCode server: {exc}"
            logger.warning("OpenCode %s on %s: %s", kind, path, exc)
            return _failure(kind, message)

        try:
            body = response.json()
        except Exception:  # noqa: BLE001
            body = None

        if response.status_code == 404:
            return _failure("not_found", _http_error_message(body, 404))
        if response.status_code >= 400:
            return _failure("unavailable", _http_error_message(body, response.status_code))
        if not isinstance(body, dict):
            if not expect_json:
                # Some endpoints answer 200 with `true` or 204 with
                # nothing. Insisting on an object reported a permission
                # reply that had in fact been accepted as a failure.
                return OpenCodeResult(True, {})
            return _failure("unavailable", "OpenCode returned an unreadable response.")
        return OpenCodeResult(True, body)

    # ── Shutdown ──────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the HTTP client and stop a server we started."""
        if self._http is not None:
            with contextlib.suppress(Exception):
                await self._http.aclose()
            self._http = None
        await self._stop_server()
        self._ready = False

    async def _stop_server(self) -> None:
        if self._log_pump is not None:
            self._log_pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._log_pump
            self._log_pump = None

        process = self._process
        self._process = None
        if process is None or process.returncode is not None:
            return

        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(Exception):
                await process.wait()
        logger.info("OpenCode server stopped")


# ── Shared instance ───────────────────────────────────────────────────

_shared_client: OpenCodeClient | None = None


def get_shared_opencode_client() -> OpenCodeClient:
    """The process-wide client.

    Shared because it owns a server subprocess: a per-request instance
    would spawn one OpenCode server per delegation.
    """
    global _shared_client
    if _shared_client is None:
        _shared_client = OpenCodeClient()
    return _shared_client


def set_shared_opencode_client(client: OpenCodeClient | None) -> None:
    """Swap the shared client. For tests and for wiring at startup."""
    global _shared_client
    _shared_client = client


async def close_shared_opencode() -> None:
    """Shut down the shared client if one was ever built.

    Deliberately does not construct one: calling
    :func:`get_shared_opencode_client` from a shutdown path would create
    a client — and a server subprocess to stop — that never existed.
    """
    global _shared_client
    if _shared_client is None:
        return
    client = _shared_client
    _shared_client = None
    await client.close()
