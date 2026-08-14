"""OpenCode tool — the delegation surface the assistant's LLM can call.

The main LLM does not reach :mod:`agent.client` directly. It emits a
structured parameter dict — ``{"action": "delegate", "task": "…"}`` — and
this module decides whether that is a permitted operation, validates the
arguments, resolves *which* OpenCode session the request belongs to, and
only then calls the client. Same two-layer split as
:mod:`actions.gmail_tool` over :mod:`actions.gmail_client`: the client
speaks the protocol, this speaks policy.

Isolation
---------
Nothing here imports from ``actions/``. OpenCode is a separate execution
layer, not another entry in the action system: it gets no access to the
assistant's tools — no Gmail, no system control, no file controller, no
YouTube — and the assistant does not reach into OpenCode's. The one
deliberate seam is a single ``agent_task`` entry in
:mod:`tasks.registry`, because the registry is the only route by which
either LLM can be told a capability exists.

Sessions
--------
An OpenCode session is a working context: files it has created, what it
already knows about the project, what it was last asked. Delegation is
therefore stateful in a way the assistant's other tools are not, and
"add JWT authentication" is only meaningful against the session that
built the API.

* ``delegate`` always opens a **new** session and makes it active.
* ``follow_up`` continues one — the active session by default, or a
  named one via ``label``.

That split is what keeps two independent tasks from bleeding into each
other: a second ``delegate`` never inherits the first one's context, and
a follow-up never lands in a fresh session that has no idea what
"that same file" refers to.

Whose work is it
----------------
The assistant delegates; it does not do the work and it does not become
OpenCode's conversational front end. What comes back is the agent's own
summary of what it did, which the main LLM then reports to the user.

Delegation does not block the conversation
------------------------------------------
``delegate`` and ``follow_up`` **return as soon as the work is under
way**, not when it finishes. A real delegation — "build me a website" —
is minutes of work, and the earlier design awaited it inside the turn:
the assistant went silent, the user could not ask anything, and a run
that was proceeding perfectly well was indistinguishable from a hang.

So each run is an :class:`asyncio.Task`, and a single subscription to
OpenCode's event stream folds every step it takes into a
:class:`agent.progress.RunProgress` on the session. That buys three
things the blocking design could not have:

* the user keeps talking while the agent works;
* "check the progress" is answered from memory, with real counts and
  the current tool call, at any moment;
* when the run ends, :meth:`OpenCodeTool.set_notifier` lets the
  assistant say so unprompted instead of waiting to be asked.

The event pump is started lazily by the first delegation and shared by
all of them — one stream, every session, dispatched by ``sessionID``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from agent.client import (
    OpenCodeClient,
    OpenCodeResult,
    get_shared_opencode_client,
)
from agent.progress import RunProgress, summarise_permission, summarise_question

logger = logging.getLogger("kancha.agent.tool")

#: Called with (label, RunProgress) when a run reaches a terminal state.
#: The assistant installs one in core.pipeline so a finished build is
#: announced instead of sitting unmentioned until someone asks.
Notifier = Callable[[str, RunProgress], Awaitable[None]]

#: A delegated task is a paragraph of intent, not an essay. Long enough
#: for "analyse this project, find the performance problems, fix them and
#: run the tests"; short enough that a runaway prompt cannot be smuggled
#: through.
MAX_TASK_CHARS = 8000

#: Labels name a session for later follow-ups. Kept to a slug so they can
#: be spoken back to the user without reading punctuation aloud.
_LABEL_RE = re.compile(r"[^a-z0-9]+")
MAX_LABEL_CHARS = 48

_WORD_RE = re.compile(r"[A-Za-z0-9]+")

#: Prepended to every delegated objective.
#:
#: The agent's default posture is a terminal session with a developer
#: sitting in front of it, so when an objective is loose it asks — and a
#: vague one ("build me a website to commercialize my project") produced
#: six questions, then four more after those were answered, then more
#: again. Over voice that is not a conversation, it is a wall: each round
#: stops the run completely until the user happens to be told and
#: answers.
#:
#: This does not remove the ability to ask, and the assistant handles it
#: properly when it does (see ``_handle_answer``). It just tells the
#: agent the truth about its situation, so it saves the questions for
#: things that genuinely cannot be guessed.
DELEGATION_PREAMBLE = (
    "You are working unattended, for someone talking to a voice assistant "
    "rather than sitting at a terminal. Every question you ask halts you "
    "completely until they are told and answer, so prefer sensible "
    "defaults: choose the stack, the structure and the details yourself, "
    "and list the assumptions you made in your final summary so they can "
    "be changed afterwards. Ask only when proceeding is genuinely "
    "impossible — a missing credential, or a choice that would waste the "
    "entire task if guessed wrong. Work in the current directory.\n\n"
    "The task:\n"
)


def _match_option(text: str, options: list[str]) -> str | None:
    """Find the option label *text* means, or None.

    The server accepts its own labels, not paraphrases, so a spoken
    answer ("the code one") has to be mapped back. Exact match first,
    then containment either way, then the best word overlap — and
    nothing at all rather than a wrong guess, because a wrong answer to
    "who is this for" is worse than asking again.
    """
    cleaned = " ".join(text.lower().split()).strip(" .!?")
    if not cleaned or not options:
        return None

    for option in options:
        if option.lower() == cleaned:
            return option
    for option in options:
        lowered = option.lower()
        if cleaned in lowered or lowered in cleaned:
            return option

    words = {w for w in re.findall(r"[a-z0-9]+", cleaned) if len(w) > 2}
    best: tuple[int, str] | None = None
    for option in options:
        option_words = {w for w in re.findall(r"[a-z0-9]+", option.lower()) if len(w) > 2}
        overlap = len(words & option_words)
        if overlap and (best is None or overlap > best[0]):
            best = (overlap, option)
    return best[1] if best else None


def _match_answers(
    questions: list[dict[str, Any]], raw: str
) -> tuple[list[list[str]], str]:
    """Turn what the user said into one answer per question.

    Returns ``(answers, problem)``; *problem* is a message to send back
    when the answer cannot be mapped, rather than a guess the agent will
    then build a whole website on.
    """
    if not questions:
        return [], "There is no question waiting."

    parts = [p.strip() for p in raw.split(";") if p.strip()]
    if len(questions) > 1 and len(parts) != len(questions):
        listed = " ".join(
            f"{i}. {q.get('question') or q.get('header')}"
            for i, q in enumerate(questions, start=1)
        )
        return [], (
            f"The agent asked {len(questions)} questions and got "
            f"{len(parts)} answer(s). Answer all of them, separated by "
            f"semicolons: {listed}"
        )
    if not parts:
        return [], "There is nothing to answer with."

    answers: list[list[str]] = []
    for question, part in zip(questions, parts):
        options = [str(o) for o in question.get("options") or []]
        if not options:
            answers.append([part])
            continue

        # "A and B" only means two choices where the question allows it.
        pieces = (
            [p.strip() for p in re.split(r",| and ", part) if p.strip()]
            if question.get("multiple")
            else [part]
        )
        chosen: list[str] = []
        for piece in pieces:
            match = _match_option(piece, options)
            if match is None:
                if question.get("custom"):
                    chosen.append(piece)
                    continue
                return [], (
                    f"'{piece}' is not one of the choices for "
                    f"\"{question.get('question') or question.get('header')}\". "
                    f"It has to be one of: {', '.join(options)}."
                )
            if match not in chosen:
                chosen.append(match)
        answers.append(chosen)

    return answers, ""


@dataclass(frozen=True, slots=True)
class AgentAction:
    """One permitted operation."""

    name: str
    summary: str


ACTIONS: dict[str, AgentAction] = {
    "delegate": AgentAction(
        name="delegate",
        summary=(
            "Hand a complete coding, research or multi-step task to the "
            "OpenCode agent (task=the full objective in the user's own "
            "terms; optional directory=where to work; optional label to "
            "name the session). Starts the work in the background and "
            "returns immediately — the agent is still running when this "
            "returns."
        ),
    ),
    "follow_up": AgentAction(
        name="follow_up",
        summary=(
            "Continue the work already in progress with a further "
            "instruction (instruction=what to change or add; optional "
            "label to pick a specific session). Keeps the agent's "
            "context, and also returns while the agent works."
        ),
    ),
    "status": AgentAction(
        name="status",
        summary="List the delegated tasks, which is active, and how each is doing.",
    ),
    "progress": AgentAction(
        name="progress",
        summary=(
            "Report in detail how a running task is going — elapsed time, "
            "steps taken, files written and what the agent is doing right "
            "now (optional label; defaults to the active task)."
        ),
    ),
    "approve": AgentAction(
        name="approve",
        summary=(
            "Let a stopped task do the thing it asked permission for — "
            "reaching outside its working directory, usually (optional "
            "label; scope='always' to stop it asking again for the same "
            "thing). Only after the user has agreed."
        ),
    ),
    "answer": AgentAction(
        name="answer",
        summary=(
            "Give a stopped task the answer to the question it asked "
            "(answer=what the user said; separate answers with ';' if it "
            "asked more than one thing; optional label)."
        ),
    ),
    "deny": AgentAction(
        name="deny",
        summary=(
            "Refuse a stopped task: declines the permission it asked for, "
            "or tells it to decide a question for itself (optional label)."
        ),
    ),
    "end_session": AgentAction(
        name="end_session",
        summary="Stop a delegated task and forget it (optional label).",
    ),
}


@dataclass(slots=True)
class DelegatedSession:
    """One OpenCode session and what the assistant knows about it."""

    label: str
    session_id: str
    objective: str
    directory: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    turns: int = 1
    #: Live state of the run, folded from OpenCode's event stream.
    progress: RunProgress = field(default_factory=RunProgress)
    #: The background task running the current instruction, if any.
    task: Any = field(default=None, repr=False)

    @property
    def running(self) -> bool:
        return self.progress.running

    def describe(self) -> str:
        objective = self.objective if len(self.objective) <= 90 else f"{self.objective[:87]}…"
        return f"{self.label}: {objective} ({self.turns} instruction(s))"

    def describe_with_progress(self) -> str:
        """The listing line, with where the run actually stands."""
        return f"{self.describe()} — {self.progress.headline()}"


@dataclass(slots=True)
class AgentToolResult:
    """Structured result handed back to the task layer."""

    success: bool
    output: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "data": self.data,
            "error": self.error,
            "action": self.action,
        }


def describe_actions() -> str:
    """Render the allowlist for a prompt. The catalog has one source."""
    return "\n".join(f"  {spec.name}: {spec.summary}" for spec in ACTIONS.values())


def make_label(text: str, taken: set[str] | None = None) -> str:
    """Derive a stable, speakable label from a task description."""
    words = _WORD_RE.findall(text.lower())[:4]
    base = "-".join(words)[:MAX_LABEL_CHARS] or "task"
    taken = taken or set()
    if base not in taken:
        return base
    for suffix in range(2, 100):
        candidate = f"{base}-{suffix}"
        if candidate not in taken:
            return candidate
    return f"{base}-{int(time.time())}"


def normalise_label(value: Any) -> str:
    """Accept whatever the LLM produced and reduce it to a slug."""
    slug = _LABEL_RE.sub("-", str(value or "").strip().lower()).strip("-")
    return slug[:MAX_LABEL_CHARS]


class OpenCodeTool:
    """Validates, routes and tracks structured delegations to OpenCode."""

    def __init__(self, client: OpenCodeClient | None = None) -> None:
        self._client_override = client
        self._sessions: dict[str, DelegatedSession] = {}
        self._active: str = ""
        self._notifier: Notifier | None = None
        #: working directory -> the task subscribed to its event stream.
        self._pumps: dict[str, asyncio.Task[None]] = {}
        #: session_id -> label, so an event can find its run in O(1).
        self._by_session_id: dict[str, str] = {}

    @property
    def client(self) -> OpenCodeClient:
        return self._client_override or get_shared_opencode_client()

    @property
    def sessions(self) -> dict[str, DelegatedSession]:
        return self._sessions

    @property
    def active_label(self) -> str:
        return self._active

    def set_notifier(self, notifier: Notifier | None) -> None:
        """Install the callback fired when a run finishes.

        Kept as a plain callable rather than an event emit so this
        package stays free of the assistant's bus — the pipeline wires
        it to ``ResponseReady`` at startup.
        """
        self._notifier = notifier

    # ── background machinery ──────────────────────────────────────────

    def _ensure_pump(self, directory: str) -> None:
        """Start the event subscription for *directory*, once.

        One pump per working directory rather than one overall:
        OpenCode's event stream is scoped to a directory (see
        :meth:`agent.client.OpenCodeClient.stream_events`), so a single
        subscription would report progress for runs in the default
        workspace and silently nothing for any other.
        """
        existing = self._pumps.get(directory)
        if existing is not None and not existing.done():
            return
        if getattr(self.client, "stream_events", None) is None:
            return  # a client that does not publish events
        self._pumps[directory] = asyncio.create_task(
            self._pump_events(directory), name=f"opencode_events{directory or ''}"
        )

    async def _pump_events(self, directory: str) -> None:
        """Fold every server event into the run it belongs to."""
        try:
            async for event in self.client.stream_events(directory):
                properties = event.get("properties")
                session_id = ""
                if isinstance(properties, dict):
                    session_id = str(properties.get("sessionID") or "")
                label = self._by_session_id.get(session_id)
                if label is None:
                    continue
                session = self._sessions.get(label)
                if session is None:
                    continue
                was_blocked = bool(session.progress.blocked_on)
                try:
                    session.progress.apply_event(event)
                except Exception:  # noqa: BLE001 — one odd event, not the pump
                    logger.debug("Could not fold event %s", event.get("type"))
                    continue
                if session.progress.blocked_on and not was_blocked:
                    # Do not wait for the next heartbeat: the run is
                    # stopped until someone answers, and every second
                    # spent not asking is a second of nothing happening.
                    logger.info(
                        "Delegated run '%s' is waiting for permission to %s",
                        session.label,
                        session.progress.permission_request(),
                    )
                    await self._fire_notifier(session)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            # Losing progress reporting must not lose the delegation:
            # the run itself is a separate task and keeps going.
            logger.exception("OpenCode event pump stopped (%s)", directory or "default")

    async def _heartbeat(self, session: DelegatedSession) -> None:
        """Volunteer progress while a run is in flight.

        "Show me progress continuously" is not the same as "answer when
        asked": a build that says nothing for ten minutes reads as a
        crash even when it is going fine. This speaks up on an interval —
        but only when the counters have actually moved, so an agent
        thinking hard between tool calls does not produce a run of
        identical updates.
        """
        interval = getattr(self.client.config, "progress_interval_s", 0.0)
        if not interval or interval <= 0:
            return
        last: tuple[int, int, int] | None = None
        while True:
            await asyncio.sleep(interval)
            progress = session.progress
            if not progress.running:
                return
            if progress.blocked_on:
                # It has already asked, and that was announced the moment
                # it happened. Repeating "still working" over the top of
                # an unanswered question helps nobody.
                continue
            signature = (progress.steps, progress.tool_calls, progress.file_count)
            if signature == last:
                continue
            last = signature
            await self._fire_notifier(session)

    async def _run(self, session: DelegatedSession, text: str) -> None:
        """Drive one instruction to completion, in the background."""
        heartbeat = asyncio.create_task(
            self._heartbeat(session), name=f"opencode_heartbeat_{session.label}"
        )
        try:
            try:
                reply = await self.client.prompt(
                    session.session_id, text, directory=session.directory
                )
            except asyncio.CancelledError:
                session.progress.cancel()
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("Delegated run '%s' crashed", session.label)
                session.progress.fail(str(exc))
                await self._fire_notifier(session)
                return

            session.updated_at = time.time()
            if reply.success:
                summary = str(reply.data.get("text") or "").strip()
                session.progress.tokens = (
                    reply.data.get("tokens") or session.progress.tokens
                )
                with contextlib.suppress(TypeError, ValueError):
                    session.progress.cost = float(
                        reply.data.get("cost") or session.progress.cost
                    )
                session.progress.finish(
                    summary
                    or (
                        f"The agent finished working on '{session.label}' but did not "
                        "report what it did."
                    )
                )
                logger.info(
                    "Delegated run '%s' finished in %.0fs (%d steps, %d files)",
                    session.label,
                    session.progress.elapsed,
                    session.progress.steps,
                    session.progress.file_count,
                )
            else:
                failure = self._failure(reply, "delegate", label=session.label)
                session.progress.fail(failure.error or "The delegation failed.")
                logger.warning(
                    "Delegated run '%s' failed: %s", session.label, session.progress.error
                )

            await self._fire_notifier(session)
        finally:
            # However this ended, stop volunteering updates about it.
            heartbeat.cancel()

    async def _fire_notifier(self, session: DelegatedSession) -> None:
        notifier = self._notifier
        if notifier is None:
            return
        try:
            await notifier(session.label, session.progress)
        except Exception:  # noqa: BLE001 — the announcement, not the run
            logger.exception("Agent completion notifier failed")

    async def join(self, label: str = "", timeout: float | None = None) -> bool:
        """Await a run's background task. For tests and shutdown.

        Returns False if there was nothing to wait for or it timed out.
        """
        session = self._sessions.get(label or self._active)
        if session is None or session.task is None:
            return False
        try:
            await asyncio.wait_for(asyncio.shield(session.task), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return False
        except Exception:  # noqa: BLE001 — _run reports failures on progress
            return True
        return True

    async def aclose(self) -> None:
        """Stop the pump and every in-flight run."""
        for session in list(self._sessions.values()):
            task = session.task
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await task
        for pump in list(self._pumps.values()):
            pump.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await pump
        self._pumps.clear()

    # ── entry point ───────────────────────────────────────────────────

    async def execute(self, params: dict[str, Any]) -> AgentToolResult:
        """Run one structured delegation action. Never raises."""
        action = str(params.get("action") or "").strip().lower()

        if action not in ACTIONS:
            return AgentToolResult(
                success=False,
                error=(
                    f"Unknown agent action '{action or '(missing)'}'. "
                    f"Supported: {', '.join(sorted(ACTIONS))}."
                ),
                action=action,
            )

        handler = getattr(self, f"_handle_{action}")
        try:
            return await handler(params)
        except Exception as exc:  # noqa: BLE001
            # A crash in the delegation layer must cost the delegation,
            # never the assistant.
            logger.exception("OpenCodeTool: %s crashed", action)
            return AgentToolResult(success=False, error=str(exc), action=action)

    # ── delegate ──────────────────────────────────────────────────────

    async def _handle_delegate(self, params: dict[str, Any]) -> AgentToolResult:
        task = self._read_text(params, "task", "objective", "request", "instruction")
        if not task:
            return AgentToolResult(
                success=False,
                action="delegate",
                error="Describe the task to delegate (task='…').",
            )
        if len(task) > MAX_TASK_CHARS:
            return AgentToolResult(
                success=False,
                action="delegate",
                error=f"That task is too long ({len(task)} chars, max {MAX_TASK_CHARS}).",
            )

        requested = normalise_label(params.get("label"))
        label = (
            requested
            if requested and requested not in self._sessions
            else make_label(requested or task, set(self._sessions))
        )

        # Where the agent will work. The executor has already resolved
        # whatever the user said into an absolute path; anything else is
        # ignored rather than guessed at, because a wrong working
        # directory means the work lands somewhere nobody looks.
        directory = str(params.get("directory") or "").strip()

        created = await self.client.create_session(
            title=task[:120], directory=directory
        )
        if not created.success:
            return self._failure(created, "delegate")

        session_id = str(created.data.get("session_id"))
        resolved_directory = str(created.data.get("directory") or directory)
        session = DelegatedSession(
            label=label,
            session_id=session_id,
            objective=task,
            directory=resolved_directory,
            progress=RunProgress(
                label=label, objective=task, directory=resolved_directory
            ),
        )
        self._sessions[label] = session
        self._by_session_id[session_id] = label
        self._active = label
        self._evict_oldest()

        logger.info(
            "Delegating to OpenCode [label=%s session=%s dir=%s]: %.120s",
            label,
            session_id,
            resolved_directory or "(default workspace)",
            task,
        )
        return self._start(session, DELEGATION_PREAMBLE + task, "delegate")

    def _start(
        self, session: DelegatedSession, text: str, action: str
    ) -> AgentToolResult:
        """Launch the run in the background and report that it started.

        The reply is deliberately not a result: there is no result yet.
        Saying so plainly is what stops the main LLM announcing a
        finished website thirty seconds into building one.
        """
        self._ensure_pump(session.directory)
        session.progress.state = "starting"
        session.task = asyncio.create_task(
            self._run(session, text), name=f"opencode_run_{session.label}"
        )
        where = f" in {session.directory}" if session.directory else ""
        return AgentToolResult(
            success=True,
            action=action,
            output=(
                f"The agent has started working on '{session.label}'{where}. "
                "It is running in the background — it is NOT finished. Tell the "
                "user it is under way and that they can ask how it is going at "
                "any time; I will report back when it completes."
            ),
            data={
                "label": session.label,
                "session_id": session.session_id,
                "directory": session.directory,
                "state": session.progress.state,
                "started": True,
            },
        )

    # ── follow_up ─────────────────────────────────────────────────────

    async def _handle_follow_up(self, params: dict[str, Any]) -> AgentToolResult:
        instruction = self._read_text(
            params, "instruction", "task", "request", "objective"
        )
        if not instruction:
            return AgentToolResult(
                success=False,
                action="follow_up",
                error="Say what to change or add (instruction='…').",
            )
        if len(instruction) > MAX_TASK_CHARS:
            return AgentToolResult(
                success=False,
                action="follow_up",
                error=(
                    f"That instruction is too long ({len(instruction)} chars, "
                    f"max {MAX_TASK_CHARS})."
                ),
            )

        session, error = self._resolve_session(params)
        if session is None:
            return AgentToolResult(success=False, action="follow_up", error=error)

        if session.running and session.task is not None and not session.task.done():
            # OpenCode would queue it, but the user almost certainly does
            # not mean "and another thing" while the first is mid-flight.
            return AgentToolResult(
                success=False,
                action="follow_up",
                error=(
                    f"'{session.label}' is still working — "
                    f"{session.progress.headline()}. Wait for it to finish, or "
                    "stop it first."
                ),
                data={"label": session.label, "state": session.progress.state},
            )

        self._active = session.label
        session.turns += 1
        session.updated_at = time.time()
        # A follow-up is a new run against the same context: reset the
        # counters so "how's it going" describes this instruction, not
        # the sum of everything the session has ever done.
        session.progress = RunProgress(
            label=session.label,
            objective=instruction,
            directory=session.directory,
        )
        logger.info(
            "Continuing OpenCode [label=%s session=%s]: %.120s",
            session.label,
            session.session_id,
            instruction,
        )
        return self._start(session, instruction, "follow_up")

    # ── status ────────────────────────────────────────────────────────

    async def _handle_status(self, params: dict[str, Any]) -> AgentToolResult:
        if not self._sessions:
            return AgentToolResult(
                success=True,
                action="status",
                output="Nothing has been delegated to the agent yet.",
                data={"sessions": [], "active": ""},
            )

        lines = []
        for label, session in self._sessions.items():
            marker = " (active)" if label == self._active else ""
            lines.append(f"{session.describe_with_progress()}{marker}")
        return AgentToolResult(
            success=True,
            action="status",
            output="\n".join(lines),
            data={
                "active": self._active,
                "running": [s.label for s in self._sessions.values() if s.running],
                "sessions": [
                    {
                        "label": s.label,
                        "objective": s.objective,
                        "turns": s.turns,
                        "session_id": s.session_id,
                        "directory": s.directory,
                        **s.progress.snapshot(),
                    }
                    for s in self._sessions.values()
                ],
            },
        )

    # ── progress ──────────────────────────────────────────────────────

    async def _handle_progress(self, params: dict[str, Any]) -> AgentToolResult:
        """Answer "how's it going" from what the event pump has seen.

        No round trip: the run is being watched continuously, so this is
        a read of local state and is as fast as any other reply.
        """
        if not self._sessions:
            return AgentToolResult(
                success=True,
                action="progress",
                output="Nothing has been delegated to the agent yet.",
                data={"sessions": []},
            )

        session, error = self._resolve_session(params)
        if session is None:
            return AgentToolResult(success=False, action="progress", error=error)

        return AgentToolResult(
            success=True,
            action="progress",
            output=session.progress.describe(),
            data={"label": session.label, **session.progress.snapshot()},
        )

    # ── permissions ───────────────────────────────────────────────────

    async def _handle_approve(self, params: dict[str, Any]) -> AgentToolResult:
        scope = str(params.get("scope") or "").strip().lower()
        return await self._answer_permission(
            params, "always" if scope == "always" else "once", "approve"
        )

    async def _handle_deny(self, params: dict[str, Any]) -> AgentToolResult:
        session, error = self._resolve_session(params)
        if session is not None and session.progress.blocked_kind == "question":
            return await self._reject_question(session)
        return await self._answer_permission(params, "reject", "deny")

    async def _handle_answer(self, params: dict[str, Any]) -> AgentToolResult:
        """Answer the clarifying question a run stopped on.

        The agent asks through its own ``question`` tool and then waits —
        the same dead stop as an unanswered permission, and the more
        likely one: a vague objective ("build me a website to sell my
        project") is exactly what makes it ask.
        """
        session, error = self._resolve_session(params)
        if session is None:
            return AgentToolResult(success=False, action="answer", error=error)

        if session.progress.blocked_kind != "question":
            await self._lookup_pending(session)
        if session.progress.blocked_kind != "question":
            return AgentToolResult(
                success=False,
                action="answer",
                error=f"'{session.label}' has not asked anything.",
                data={"label": session.label},
            )

        questions = session.progress.pending_questions()
        request_id = str(session.progress.blocked_on.get("id") or "")
        raw = self._read_text(params, "answer", "reply", "response", "instruction")
        if not raw:
            return AgentToolResult(
                success=False,
                action="answer",
                error=(
                    "Say what the answer is. The agent asked: "
                    f"{session.progress.question_prompt()}"
                ),
                data={"label": session.label, "questions": questions},
            )

        answers, problem = _match_answers(questions, raw)
        if problem:
            return AgentToolResult(
                success=False, action="answer", error=problem,
                data={"label": session.label, "questions": questions},
            )

        result = await self.client.answer_question(
            request_id, answers, directory=session.directory
        )
        if not result.success:
            return self._failure(result, "answer", label=session.label)

        session.progress.blocked_on = {}
        session.progress.updated_at = time.time()
        logger.info("Answered '%s': %s", session.label, answers)
        return AgentToolResult(
            success=True,
            action="answer",
            output=f"Answered. '{session.label}' is working again.",
            data={"label": session.label, "answers": answers},
        )

    async def _reject_question(self, session: DelegatedSession) -> AgentToolResult:
        request_id = str(session.progress.blocked_on.get("id") or "")
        result = await self.client.reject_question(
            request_id, directory=session.directory
        )
        if not result.success:
            return self._failure(result, "deny", label=session.label)
        session.progress.blocked_on = {}
        session.progress.updated_at = time.time()
        return AgentToolResult(
            success=True,
            action="deny",
            output=(
                f"Told '{session.label}' to decide for itself and carry on."
            ),
            data={"label": session.label},
        )

    async def _answer_permission(
        self, params: dict[str, Any], reply: str, action: str
    ) -> AgentToolResult:
        """Unblock a run that stopped to ask for something.

        Answering is the *only* way out of this state: the agent waits
        indefinitely, so an unanswered request is a run that is finished
        in every sense except that it never reports.
        """
        session, error = self._resolve_session(params)
        if session is None:
            return AgentToolResult(success=False, action=action, error=error)

        if session.progress.blocked_kind == "question":
            # Approving a question is meaningless — it wants an answer.
            return AgentToolResult(
                success=False,
                action=action,
                error=(
                    f"'{session.label}' is waiting for an answer, not approval. "
                    f"It asked: {session.progress.question_prompt()}"
                ),
                data={
                    "label": session.label,
                    "questions": session.progress.pending_questions(),
                },
            )

        request = dict(session.progress.blocked_on)
        request_id = str(request.get("id") or "")
        if not request_id:
            # The event may have been missed (a dropped stream); ask the
            # server rather than telling the user there is nothing to do.
            request_id = await self._lookup_pending(session)
        if not request_id:
            return AgentToolResult(
                success=False,
                action=action,
                error=f"'{session.label}' is not waiting for permission to anything.",
                data={"label": session.label},
            )

        result = await self.client.reply_permission(
            request_id, reply, directory=session.directory
        )
        if not result.success:
            return self._failure(result, action, label=session.label)

        session.progress.blocked_on = {}
        session.progress.updated_at = time.time()
        what = request.get("resource") or "that"
        if reply == "reject":
            message = f"Refused. '{session.label}' will carry on without {what}."
        else:
            message = (
                f"Approved{' for good' if reply == 'always' else ''}. "
                f"'{session.label}' is working again."
            )
        logger.info("Permission %s for '%s': %s", reply, session.label, what)
        return AgentToolResult(
            success=True,
            action=action,
            output=message,
            data={"label": session.label, "reply": reply, "request": request},
        )

    async def _lookup_pending(self, session: DelegatedSession) -> str:
        """Ask the server what this session is stopped on.

        The event may have been missed — a dropped stream, or a pump
        started after the fact. Saying "nothing is pending" while the run
        sits blocked would be the worst possible answer, so check.
        """
        for attribute, summarise in (
            ("pending_permissions", summarise_permission),
            ("pending_questions", summarise_question),
        ):
            lookup = getattr(self.client, attribute, None)
            if lookup is None:
                continue
            result = await lookup(session.directory)
            if not result.success:
                continue
            for request in result.data.get("requests") or []:
                if not isinstance(request, dict):
                    continue
                if str(request.get("sessionID") or "") == session.session_id:
                    session.progress.blocked_on = summarise(request)
                    return str(request.get("id") or "")
        return ""

    # ── end_session ───────────────────────────────────────────────────

    async def _handle_end_session(self, params: dict[str, Any]) -> AgentToolResult:
        session, error = self._resolve_session(params)
        if session is None:
            return AgentToolResult(success=False, action="end_session", error=error)

        # Best-effort: a session that is still working should stop before
        # we forget it exists, or it keeps burning tokens unobserved.
        await self.client.abort(session.session_id, directory=session.directory)

        task = session.task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await task
        session.progress.cancel()

        self._sessions.pop(session.label, None)
        self._by_session_id.pop(session.session_id, None)
        if self._active == session.label:
            self._active = next(reversed(self._sessions), "") if self._sessions else ""
        return AgentToolResult(
            success=True,
            action="end_session",
            output=f"Stopped '{session.label}' and forgot it.",
            data={"label": session.label, "active": self._active},
        )

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _read_text(params: dict[str, Any], *names: str) -> str:
        """First non-empty of *names*.

        The alias list is not politeness — the planner fills parameters
        from a natural-language plan step, and a task worth delegating
        arriving under ``request`` instead of ``task`` should not fail as
        "describe the task".
        """
        for name in names:
            value = str(params.get(name) or "").strip()
            if value:
                return value
        return ""

    def _resolve_session(
        self, params: dict[str, Any]
    ) -> tuple[DelegatedSession | None, str]:
        """Pick the session a follow-up refers to."""
        label = normalise_label(params.get("label"))
        if label:
            session = self._sessions.get(label)
            if session is not None:
                return session, ""
            known = ", ".join(self._sessions) or "none"
            return None, f"No delegated task called '{label}'. Known: {known}."

        session_id = str(params.get("session_id") or "").strip()
        if session_id:
            for session in self._sessions.values():
                if session.session_id == session_id:
                    return session, ""
            return None, f"No delegated task with session id '{session_id}'."

        if self._active and self._active in self._sessions:
            return self._sessions[self._active], ""
        return None, "Nothing has been delegated yet, so there is nothing to continue."

    def _evict_oldest(self) -> None:
        """Keep the tracked set bounded without touching live work."""
        limit = self.client.config.max_sessions
        while len(self._sessions) > limit:
            for label in list(self._sessions):
                # Never evict the active one, and never evict a run still
                # in flight — forgetting it would leave a background task
                # writing files with nothing tracking or reporting it.
                if label == self._active or self._sessions[label].running:
                    continue
                dropped = self._sessions.pop(label)
                self._by_session_id.pop(dropped.session_id, None)
                logger.info(
                    "Forgetting OpenCode session %s (%s) — tracking limit %d",
                    dropped.label,
                    dropped.session_id,
                    limit,
                )
                break
            else:  # nothing safe to drop
                return

    @staticmethod
    def _failure(
        result: OpenCodeResult, action: str, label: str = ""
    ) -> AgentToolResult:
        """Turn a client error into something the LLM can act on."""
        hints = {
            "disabled": "The agent is turned off (KANCHA_OPENCODE_ENABLED=0).",
            "not_installed": "OpenCode is not installed on this machine.",
            "unavailable": "The OpenCode server could not be reached.",
            "timeout": "The agent ran out of time.",
            "agent_error": "The agent could not complete the task.",
            "not_found": "That delegated session no longer exists on the server.",
            "network_error": "The connection to the agent failed."
        }
        prefix = hints.get(result.error_kind or "", "")
        message = f"{prefix} {result.error}".strip() if prefix else (result.error or "")
        data: dict[str, Any] = {}
        if result.error_kind:
            data["error_kind"] = result.error_kind
        if label:
            data["label"] = label
        return AgentToolResult(
            success=False,
            action=action,
            error=message or "The delegation failed.",
            data=data,
        )


# ── Shared instance ───────────────────────────────────────────────────

_shared_tool: OpenCodeTool | None = None


def get_shared_opencode_tool() -> OpenCodeTool:
    """The process-wide tool, so sessions survive between turns.

    A per-request instance would forget every session, and "add JWT
    authentication" would open a fresh agent with no idea what API the
    user is talking about.
    """
    global _shared_tool
    if _shared_tool is None:
        _shared_tool = OpenCodeTool()
    return _shared_tool


def set_shared_opencode_tool(tool: OpenCodeTool | None) -> None:
    """Swap the shared tool. For tests and for wiring at startup."""
    global _shared_tool
    _shared_tool = tool
