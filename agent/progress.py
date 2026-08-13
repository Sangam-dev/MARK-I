"""What a delegated run is doing right now.

A delegation is minutes of work behind a single HTTP call that says
nothing until it finishes. That is fine for a machine and useless for a
person: "build me a website" looked identical to a crash for eight
minutes, and the only honest answer to "how's it going?" was "I don't
know".

OpenCode does publish the missing detail — its server emits an SSE
stream of the agent's every step, and its own TUI is drawn from it. This
module folds that stream into one small mutable record per run, so the
assistant can answer "check the progress" from memory, instantly, with
no round trip to anything.

What is deliberately *not* here
-------------------------------
No transport (:mod:`agent.client` owns that), no sessions
(:mod:`agent.tool`), and no bus. A ``RunProgress`` is a value you can
build from a list of dicts in a test, which is exactly how the progress
tests drive it.

The vocabulary
--------------
Folded from a live 1.18.16 server (see ``apply_event``):

``message.part.updated`` with ``part.type``
    ``step-start`` — the agent began a reasoning step.
    ``tool`` — a tool call, with ``state.status`` moving
    pending → running → completed/error and ``state.title`` carrying
    the human-readable form ("mkdir -p demo", "src/App.tsx").
    ``step-finish`` — token and cost accounting for the step.
``message.part.delta``
    Streaming assistant text, one fragment per event.
``file.edited``
    A file the run actually changed. The most concrete progress signal
    there is, and the one a user most wants read back.
``session.status``
    ``busy`` / ``idle`` — liveness, independent of what we infer.
``session.idle``
    The agent loop stopped.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Tool calls kept for the narrative. Enough to say what has been
#: happening lately; bounded so a two-hour run cannot grow without limit.
MAX_TOOL_HISTORY = 40

#: Same reasoning for files. The count stays exact either way.
MAX_FILES = 200

#: The tail of the agent's own prose, for "what is it saying".
MAX_TEXT_CHARS = 600

#: Runs that have stopped for one reason or another.
TERMINAL_STATES = frozenset({"done", "failed", "cancelled"})

#: Tools whose completion means a file now exists or changed.
_FILE_TOOLS = frozenset({"write", "edit", "patch", "multiedit", "apply_patch"})


def _short(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


#: Markdown the agent writes for a terminal and nobody wants read aloud.
#: Underscores are deliberately absent — they are markup far less often
#: than they are part of a path, and "jarvis_website" must not come back
#: as "jarviswebsite".
_MARKDOWN_NOISE_RE = re.compile(r"[`*#>]+")

#: "Done." / "Finished:" — the announcement already says as much.
_REDUNDANT_OPENER_RE = re.compile(
    r"^(?:done|finished|completed|all done)\b[\s.!:—-]*", re.IGNORECASE
)


def headline_of(text: str, limit: int = 200) -> str:
    """The gist of the agent's summary, in a sentence or two.

    Its summaries are written for a code editor: several hundred words
    of markdown with bullet lists of every file. Announced verbatim that
    is a wall of text, and unusable over speech — so take the opening
    prose, drop the markup, and stop at the first list.
    """
    head = str(text or "").strip()
    if not head:
        return ""

    # The *earliest* of these, not the first one in the list: a summary
    # that opens a section ("\n\n**What's there**") before its bullets
    # would otherwise keep the heading and read it out as prose.
    cuts = [
        head.find(marker)
        for marker in ("\n-", "\n*", "\n•", "\n1.", "\n\n")
        if head.find(marker) > 0
    ]
    if cuts:
        head = head[: min(cuts)]

    head = " ".join(_MARKDOWN_NOISE_RE.sub("", head).split())
    head = _REDUNDANT_OPENER_RE.sub("", head).strip()

    sentences = re.split(r"(?<=[.!?])\s+", head)
    out = ""
    for sentence in sentences:
        candidate = f"{out} {sentence}".strip()
        if out and len(candidate) > limit:
            break
        out = candidate
        if len(out) >= limit:
            break
    return _short(out, limit)


def humanise_duration(seconds: float) -> str:
    """A duration a person would say out loud."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        if minutes < 5 and secs:
            return f"{minutes} minute{'s' if minutes != 1 else ''} {secs} seconds"
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours, minutes = divmod(minutes, 60)
    if minutes:
        return f"{hours} hour{'s' if hours != 1 else ''} {minutes} minutes"
    return f"{hours} hour{'s' if hours != 1 else ''}"


#: What each permission actually means, said the way a person would.
#: The raw names ("external_directory") are for the wire, not for speech.
_PERMISSION_PHRASES: dict[str, str] = {
    "external_directory": "look outside its own folder",
    "bash": "run a shell command",
    "edit": "edit a file",
    "write": "write a file",
    "webfetch": "fetch a web page",
    "network": "use the network",
}


def _speakable_resource(raw: str, limit: int = 70) -> str:
    """The target of a request, short enough to hear.

    A permission's target can be a whole shell pipeline — three commands
    joined by semicolons with redirects — and reading it out is the
    definition of annoying. Take the first command, shorten the home
    directory, and cut it well before it becomes a recital.
    """
    text = " ".join(str(raw or "").split())
    if not text:
        return ""
    # The first command of a pipeline is the one that says what this is.
    for separator in (" && ", " ; ", "; ", " | "):
        index = text.find(separator)
        if index > 0:
            text = text[:index]
            break
    text = re.sub(r"\s*2>/dev/null|\s*>/dev/null|\s*2>&1", "", text)
    home = str(Path.home())
    if home and home in text:
        text = text.replace(home, "~")
    return _short(text.strip(), limit)


def summarise_permission(props: dict[str, Any]) -> dict[str, Any]:
    """Reduce a ``permission.asked`` payload to id, action and target."""
    action = str(props.get("permission") or props.get("action") or "access")
    metadata = props.get("metadata") if isinstance(props.get("metadata"), dict) else {}
    resource = str(
        metadata.get("filepath")
        or metadata.get("parentDir")
        or metadata.get("command")
        or ""
    )
    if not resource:
        patterns = props.get("patterns") or props.get("resources")
        if isinstance(patterns, list) and patterns:
            resource = str(patterns[0])
    return {
        "kind": "permission",
        "id": str(props.get("id") or ""),
        "permission": _PERMISSION_PHRASES.get(action, action.replace("_", " ")),
        "resource": _speakable_resource(resource),
    }


def summarise_question(props: dict[str, Any]) -> dict[str, Any]:
    """Reduce a ``question.asked`` payload to something answerable.

    ``questions`` keeps every question the request carries, each with
    its options, because the reply has to supply an answer per question
    in order — and because the point of surfacing this at all is to read
    the actual question to the user, not to say "it asked something".
    """
    questions: list[dict[str, Any]] = []
    raw = props.get("questions")
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            options = []
            for option in item.get("options") or []:
                if isinstance(option, dict) and option.get("label"):
                    options.append(str(option["label"]))
                elif isinstance(option, str):
                    options.append(option)
            questions.append(
                {
                    "question": str(item.get("question") or item.get("header") or ""),
                    "header": str(item.get("header") or ""),
                    "options": options,
                    "custom": bool(item.get("custom", False)),
                    "multiple": bool(item.get("multiple", False)),
                }
            )
    return {
        "kind": "question",
        "id": str(props.get("id") or ""),
        "questions": questions,
    }


@dataclass(slots=True)
class ToolStep:
    """One tool call the agent made."""

    tool: str
    title: str = ""
    status: str = "pending"
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0

    def describe(self) -> str:
        label = self.tool or "tool"
        return f"{label} ({_short(self.title, 70)})" if self.title else label


@dataclass(slots=True)
class RunProgress:
    """Live state of one delegated run. Mutated by :meth:`apply_event`."""

    label: str = ""
    objective: str = ""
    directory: str = ""
    state: str = "starting"  # starting | working | done | failed | cancelled
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    steps: int = 0
    tool_calls: int = 0
    tools: list[ToolStep] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    file_count: int = 0

    #: Streaming prose from the agent, kept to a tail.
    text: str = ""
    #: Server-reported liveness, separate from what we infer.
    server_status: str = ""

    #: What the run has stopped to ask, if anything. Either a permission
    #: request (``kind="permission"``) or a clarifying question
    #: (``kind="question"``). A run in this state is not slow — it is
    #: waiting for an answer and will wait forever.
    blocked_on: dict[str, Any] = field(default_factory=dict)

    tokens: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0

    #: Set once the run stops.
    result: str = ""
    error: str = ""

    # Bookkeeping for in-flight tool calls, keyed by the part id.
    _open_tools: dict[str, ToolStep] = field(default_factory=dict, repr=False)

    # ── derived ───────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self.state not in TERMINAL_STATES

    @property
    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return max(0.0, end - self.started_at)

    @property
    def idle_for(self) -> float:
        """Seconds since anything last happened."""
        return max(0.0, time.time() - self.updated_at)

    @property
    def current_tool(self) -> ToolStep | None:
        for step in reversed(self.tools):
            if step.status in {"pending", "running"}:
                return step
        return None

    # ── folding ───────────────────────────────────────────────────────

    def apply_event(self, event: dict[str, Any]) -> None:
        """Fold one server event in. Unknown events are ignored."""
        kind = str(event.get("type") or "")
        props = event.get("properties")
        if not isinstance(props, dict):
            props = {}

        handled = True
        if kind == "message.part.updated":
            self._apply_part(props.get("part"))
        elif kind == "message.part.delta":
            if props.get("field") == "text":
                self._append_text(str(props.get("delta") or ""))
        elif kind == "file.edited":
            self._add_file(str(props.get("file") or ""))
        elif kind == "session.status":
            status = props.get("status")
            if isinstance(status, dict):
                self.server_status = str(status.get("type") or "")
        elif kind == "session.idle":
            self.server_status = "idle"
        elif kind in {"permission.asked", "permission.v2.asked"}:
            self.blocked_on = summarise_permission(props)
        elif kind in {"question.asked", "question.v2.asked"}:
            self.blocked_on = summarise_question(props)
        elif kind in {
            "permission.replied",
            "permission.v2.replied",
            "question.replied",
            "question.rejected",
            "question.v2.replied",
            "question.v2.rejected",
        }:
            self.blocked_on = {}
        else:
            handled = False

        if handled:
            # Only real activity counts as activity: a heartbeat that
            # bumped the clock would make a wedged run look healthy.
            self.updated_at = time.time()
            if self.state == "starting":
                self.state = "working"

    def _apply_part(self, part: Any) -> None:
        if not isinstance(part, dict):
            return
        part_type = str(part.get("type") or "")

        if part_type == "step-start":
            self.steps += 1
            return

        if part_type == "step-finish":
            tokens = part.get("tokens")
            if isinstance(tokens, dict):
                self.tokens = tokens
            try:
                self.cost = float(part.get("cost") or self.cost)
            except (TypeError, ValueError):
                pass
            return

        if part_type != "tool":
            return

        key = str(part.get("id") or part.get("callID") or "")
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        status = str(state.get("status") or "pending")
        title = str(state.get("title") or "")
        if not title:
            # Before the tool reports a title, its input is the best
            # description available — "npm install" beats "bash".
            inputs = state.get("input")
            if isinstance(inputs, dict):
                for candidate in ("command", "filePath", "path", "pattern", "query"):
                    if inputs.get(candidate):
                        title = str(inputs[candidate])
                        break

        step = self._open_tools.get(key)
        if step is None:
            step = ToolStep(tool=str(part.get("tool") or "tool"), title=title, status=status)
            self.tools.append(step)
            if key:
                self._open_tools[key] = step
            if len(self.tools) > MAX_TOOL_HISTORY:
                dropped = self.tools.pop(0)
                for open_key, open_step in list(self._open_tools.items()):
                    if open_step is dropped:
                        self._open_tools.pop(open_key, None)
        else:
            if title:
                step.title = title
            step.status = status

        if status in {"completed", "error"} and step.ended_at == 0.0:
            step.ended_at = time.time()
            self.tool_calls += 1
            self._open_tools.pop(key, None)
            if status == "completed":
                self._record_written_file(step, state)

    def _record_written_file(self, step: ToolStep, state: dict[str, Any]) -> None:
        """Count a file the agent wrote through a tool.

        ``file.edited`` covers files changed on disk by a shell command,
        but the agent's own ``write``/``edit`` tools do not emit it — a
        run that built three files that way reported none, which is
        precisely the detail the user wants read back.
        """
        if step.tool not in _FILE_TOOLS:
            return
        path = ""
        inputs = state.get("input")
        if isinstance(inputs, dict):
            path = str(inputs.get("filePath") or inputs.get("path") or "")
        if not path and "/" in step.title:
            path = step.title
        self._add_file(path.strip())

    def _append_text(self, delta: str) -> None:
        if not delta:
            return
        self.text = (self.text + delta)[-MAX_TEXT_CHARS:]

    def _add_file(self, path: str) -> None:
        if not path:
            return
        if path in self.files:
            return
        self.files.append(path)
        self.file_count += 1
        if len(self.files) > MAX_FILES:
            self.files.pop(0)

    # ── transitions ───────────────────────────────────────────────────

    def finish(self, result: str) -> None:
        self.state = "done"
        self.result = result.strip()
        self.finished_at = time.time()
        self.updated_at = self.finished_at

    def fail(self, error: str) -> None:
        self.state = "failed"
        self.error = error.strip()
        self.finished_at = time.time()
        self.updated_at = self.finished_at

    def cancel(self) -> None:
        self.state = "cancelled"
        self.finished_at = time.time()
        self.updated_at = self.finished_at

    # ── reporting ─────────────────────────────────────────────────────

    def headline(self) -> str:
        """One line: what this run is and where it stands."""
        if self.state == "done":
            return f"'{self.label}' finished after {humanise_duration(self.elapsed)}"
        if self.state == "failed":
            return f"'{self.label}' failed after {humanise_duration(self.elapsed)}"
        if self.state == "cancelled":
            return f"'{self.label}' was stopped after {humanise_duration(self.elapsed)}"
        if self.blocked_kind == "question":
            return f"'{self.label}' is stopped, waiting for an answer to a question"
        if self.blocked_on:
            return (
                f"'{self.label}' is stopped, waiting for permission to "
                f"{self.permission_request()}"
            )
        if self.state == "starting":
            return f"'{self.label}' is starting up"
        return f"'{self.label}' has been working for {humanise_duration(self.elapsed)}"

    @property
    def blocked_kind(self) -> str:
        """"permission", "question", or "" when the run is not stopped."""
        return str(self.blocked_on.get("kind") or "") if self.blocked_on else ""

    def permission_request(self) -> str:
        """The pending request as a phrase — "look outside its own folder"."""
        if self.blocked_kind != "permission":
            return ""
        action = str(self.blocked_on.get("permission") or "access")
        resource = str(self.blocked_on.get("resource") or "")
        return f"{action} ({resource})" if resource else action

    def pending_questions(self) -> list[dict[str, Any]]:
        if self.blocked_kind != "question":
            return []
        questions = self.blocked_on.get("questions")
        return questions if isinstance(questions, list) else []

    def question_prompt(self, first_only: bool = False) -> str:
        """The questions as something a person can answer out loud.

        The options matter as much as the question: the agent will only
        accept one of its own labels back unless it marked the question
        as accepting free text.

        *first_only* is for the unprompted announcement. The agent asks
        in batches — six questions with four options each is normal —
        and reciting all of that at someone who did not ask is the sort
        of thing that makes people stop listening.
        """
        questions = self.pending_questions()
        if not questions:
            return ""
        if first_only and len(questions) > 1:
            rest = len(questions) - 1
            headers = ", ".join(
                str(q.get("header") or "").strip().lower()
                for q in questions[1:]
                if q.get("header")
            )
            about = f" — about {headers}" if headers else ""
            return (
                f"{self._render_question(questions[0], '')} "
                f"There {'are' if rest > 1 else 'is'} {rest} more question"
                f"{'s' if rest > 1 else ''}{about}. Ask me for the details "
                "to hear them."
            )
        return " ".join(
            self._render_question(item, f"{index}. " if len(questions) > 1 else "")
            for index, item in enumerate(questions, start=1)
        )

    @staticmethod
    def _render_question(item: dict[str, Any], prefix: str) -> str:
        text = str(item.get("question") or item.get("header") or "").strip()
        options = [str(o) for o in item.get("options") or []]
        if not options:
            return f"{prefix}{text}"
        suffix = " (or say something else)" if item.get("custom") else ""
        return f"{prefix}{text} Options: {', '.join(options)}.{suffix}"

    def brief(self) -> str:
        """One sentence, for a message the user did not ask for.

        :meth:`describe` answers "how is it going" — someone asked, so
        detail is the point. This is the opposite case: an announcement
        arriving unprompted, where the full report is a wall of text and
        what is wanted is "it's done, here's what it made". Never
        includes the label; the caller has already said it.
        """
        if self.state == "done":
            scale = (
                f"{self.file_count} file{'s' if self.file_count != 1 else ''}"
                if self.file_count
                else f"{self.steps} step{'s' if self.steps != 1 else ''}"
            )
            gist = headline_of(self.result)
            done = f"{scale} in {humanise_duration(self.elapsed)}."
            return f"{done} {gist}".strip() if gist else done

        if self.state == "failed":
            return (
                f"It stopped after {humanise_duration(self.elapsed)}. "
                f"{_short(self.error, 160)}"
            ).strip()

        if self.state == "cancelled":
            return f"Stopped after {humanise_duration(self.elapsed)}."

        if self.blocked_kind == "question":
            return f"It is asking: {self.question_prompt(first_only=True)}"
        if self.blocked_on:
            return f"It needs permission to {self.permission_request()}."

        counters = [f"{self.steps} step{'s' if self.steps != 1 else ''}"]
        if self.file_count:
            counters.append(
                f"{self.file_count} file{'s' if self.file_count != 1 else ''}"
            )
        current = self.current_tool
        where = f" Right now: {current.describe()}." if current is not None else ""
        return (
            f"{humanise_duration(self.elapsed)} in, "
            f"{' and '.join(counters)} so far.{where}"
        )

    def describe(self) -> str:
        """A paragraph a person can hear and act on.

        Written to be read aloud: counts first because they are what
        "how far along is it" actually means, then the current activity,
        then the most recent files.
        """
        parts = [self.headline()]

        if self.objective:
            parts.append(f"Task: {_short(self.objective, 120).rstrip('.')}.")

        counters = []
        if self.steps:
            counters.append(f"{self.steps} step{'s' if self.steps != 1 else ''}")
        if self.tool_calls:
            counters.append(
                f"{self.tool_calls} tool call{'s' if self.tool_calls != 1 else ''}"
            )
        if self.file_count:
            counters.append(
                f"{self.file_count} file{'s' if self.file_count != 1 else ''} written"
            )
        if counters:
            parts.append("So far: " + ", ".join(counters) + ".")

        if self.blocked_kind == "question":
            # Read the question out. Reporting only that one exists puts
            # the user one round trip away from the thing they need.
            parts.append(
                f"It is asking: {self.question_prompt()} It will not continue "
                "until it gets an answer."
            )
        elif self.blocked_on:
            # Say what to do about it. This state does not resolve on its
            # own — the agent waits for an answer indefinitely.
            parts.append(
                f"It needs permission to {self.permission_request()} and will not "
                "continue until it is approved or refused."
            )
        elif self.running:
            current = self.current_tool
            if current is not None:
                parts.append(f"Right now: {current.describe()}.")
            elif self.tools:
                parts.append(f"Last action: {self.tools[-1].describe()}.")
            # A long silence is the single most useful thing to surface:
            # it is the difference between "working" and "wedged".
            if self.idle_for > 120 and self.steps:
                parts.append(
                    f"Nothing has happened for {humanise_duration(self.idle_for)}."
                )
        elif self.tools:
            recent = ", ".join(step.describe() for step in self.tools[-3:])
            parts.append(f"Last actions: {recent}.")

        if self.files:
            shown = [p.rsplit("/", 1)[-1] for p in self.files[-5:]]
            more = f" and {self.file_count - len(shown)} more" if self.file_count > len(shown) else ""
            parts.append(f"Files: {', '.join(shown)}{more}.")

        if self.state == "done" and self.result:
            # Even when asked, the raw summary is pages of markdown.
            parts.append(f"Result: {headline_of(self.result, 320)}")
        if self.state == "failed" and self.error:
            parts.append(f"Error: {_short(self.error, 300)}")

        return " ".join(parts)

    def snapshot(self) -> dict[str, Any]:
        """Machine-readable form, for the API and for tests."""
        current = self.current_tool
        return {
            "label": self.label,
            "state": self.state,
            "objective": self.objective,
            "directory": self.directory,
            "elapsed_s": round(self.elapsed, 1),
            "idle_s": round(self.idle_for, 1) if self.running else 0.0,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "files": list(self.files),
            "file_count": self.file_count,
            "current_tool": current.describe() if current else "",
            "recent_tools": [step.describe() for step in self.tools[-5:]],
            "text": self.text.strip(),
            "server_status": self.server_status,
            "blocked_on": dict(self.blocked_on),
            "blocked_kind": self.blocked_kind,
            "awaiting_permission": self.blocked_kind == "permission",
            "awaiting_answer": self.blocked_kind == "question",
            "question": self.question_prompt(),
            "tokens": dict(self.tokens),
            "cost": self.cost,
            "result": self.result,
            "error": self.error,
        }
