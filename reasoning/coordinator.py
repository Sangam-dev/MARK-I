"""The Conversation LLM — KANCHA's controller.

Every user utterance enters here and nowhere else. This class decides
whether a turn is ordinary conversation or needs real work done, and
when it needs work done it delegates *explicitly* via
:class:`~core.events.TaskRequested` to the Task LLM
(:mod:`planning.planner`). The Task LLM never sees raw user input.

    USER ─► ReasoningCoordinator ─┬─► ResponseReady            (talk)
                                  └─► TaskRequested            (act)
                                          │
                                       Task LLM ─► tools ─► TaskResultReady
                                          │
                                  ReasoningCoordinator ─► ResponseReady

Because this class owns the conversation, it — and only it — can
resolve references like "do it", "send that", "use the second one".
Resolution happens *before* delegation: the ``instruction`` carried by
:class:`~core.events.TaskRequested` is always self-contained.

Conversation history ownership
--------------------------------
The coordinator owns the short-term in-memory buffer directly.
It adds user/assistant turns synchronously so the context is
always up-to-date before the LLM call.  Durable memory (SQL facts
and RAG entries) is carried inline by the primary Gemini response
JSON envelope and persisted here via MemoryManager.save_sql /
append_rag — there is no separate extraction step anymore.

RAG ownership
-------------
This class is the **Conversation Manager** in the RAG architecture.
It never touches the vector database: retrieval goes through
:class:`~memory.rag.manager.RAGManager`, gated by a
:class:`~memory.rag.router.MemoryRouter`. Conversely the RAG Manager
never builds prompts — turning retrieved chunks into prompt text is
:meth:`_format_rag_context`, and it lives here on purpose.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from typing import Any

from core.audio_state import audio_state
from core.bus import EventBus
from core.clock import current_datetime_block
from core.events import (
    IntentIdentified,
    MemoryRetrieved,
    PartialResponse,
    PartialTranscriptReady,
    ResponseReady,
    TaskRequested,
    TaskResultReady,
    TextInputReceived,
    TranscriptReady,
    UserInterrupted,
)
from memory.activity_memory import ActivityMemory
from memory.manager import MemoryManager
from memory.rag import MemoryRouter, RAGManager, RetrievedChunk
from memory.token_log import TokenLog
from planning.orchestrator import looks_like_assent, looks_like_refusal
from planning.prompts import format_tool_catalog
from reasoning.llm_client import (
    GeminiClient,
    extract_streamed_message,
    looks_like_envelope,
    parse_memory_response,
)
from reasoning.naturalize import naturalize_plan_response
from reasoning.prompt_builder import JARVIS_PERSONA

logger = logging.getLogger("kancha.reasoning.coordinator")


# Marker prefix for the synthetic history turn that carries a task result
# back into the conversation on the chaining path. It is explained to the
# model in CONTROLLER_INSTRUCTIONS so it never mistakes one for the user
# speaking.
TASK_RESULT_TURN_PREFIX = "[task result]"

# How many times one user turn may chain into another task before we stop
# feeding results back to the controller. Chaining is a real requirement
# ("find the email, then reply to it"), but a controller that keeps
# setting follow_up would otherwise run tools in a loop with nobody
# watching. Past the cap the result is simply reported to the user.
MAX_TASK_CHAIN_DEPTH = 4

# Task statuses that are a question rather than an outcome. On these the
# controller asks the user and the task stays open in the Orchestrator.
WAITING_STATUSES = frozenset({"waiting_for_input", "waiting_for_confirmation"})

# Modes that continue the task already open in this session, rather than
# starting a new one. Mirrors planning/orchestrator.py, which validates
# whatever we propose against the real task state.
RESUME_MODES = frozenset({"answer", "confirm", "reject", "cancel", "modify"})

# How long a delegated task may run before its acknowledgement ("Opening
# Firefox…") is spoken. Under this, the user hears one response — the
# outcome — instead of a confirmation followed by a near-identical
# confirmation. Over it, the acknowledgement goes out so a slow tool
# (a cold-start web search, a desktop automation) isn't dead air.
#
# 3s is chosen from measured round trips: weather, app launches, alarms
# and file operations all land inside ~2.5s, so they answer once. Only
# work that is genuinely going to keep the user waiting announces itself.
#
# Override with KANCHA_TASK_ACK_DELAY (seconds); 0 speaks it immediately,
# which restores the always-two-responses behaviour.
DEFAULT_TASK_ACK_DELAY_S = 3.0


# How much raw tool output to carry forward as reference context. Enough
# for a listing of ten emails with their ids; short of pasting an entire
# inbox into every subsequent prompt.
MAX_REFERENCE_CHARS = 1800


def _compact_raw(results: list[dict[str, Any]] | None) -> str:
    """Render tool output for the controller's eyes only."""
    lines: list[str] = []
    for entry in results or []:
        text = str(entry.get("result", "")).strip()
        if not text:
            continue
        tool = str(entry.get("tool", "")).strip() or "tool"
        lines.append(f"{tool}: {text}")
    joined = "\n".join(lines)
    if len(joined) > MAX_REFERENCE_CHARS:
        joined = joined[:MAX_REFERENCE_CHARS].rstrip() + "\n… (truncated)"
    return joined


def _ack_delay_from_env() -> float:
    raw = os.getenv("KANCHA_TASK_ACK_DELAY")
    if not raw:
        return DEFAULT_TASK_ACK_DELAY_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_TASK_ACK_DELAY_S


# ── Preemptive generation ─────────────────────────────────────────────────────
# A speculative reply is generated on the *interim* transcript (raced ahead
# by input/stt.py — fired mid-speech on a snapshot) while the authoritative
# Whisper transcription still runs. The guess is only ever *reused* when the
# authoritative transcript validates it; everything about it is best-effort
# and discardable. Every knob below is tuned to keep the speculation cheap
# and safe — one guess per turn, never on task-like turns, never delegated,
# never retrieved.
PREEMPTIVE_MIN_WORDS = int(os.getenv("KANCHA_PREEMPTIVE_MIN_WORDS", "4"))
# A parked guess older than this is stale (the user has moved on) and is
# discarded rather than trusted.
PREEMPTIVE_MAX_AGE_S = float(os.getenv("KANCHA_PREEMPTIVE_MAX_AGE_S", "20.0"))
# How long to wait on an in-flight guess before giving up and running the
# normal turn. Kept tiny — the guess already had the rest of the utterance
# plus the endpoint plus the final Whisper round-trip as runway, and a slow
# one is not worth delaying the real answer for.
PREEMPTIVE_REUSE_WAIT_S = float(os.getenv("KANCHA_PREEMPTIVE_REUSE_WAIT_S", "0.2"))
# The interim ASR runs mid-speech on a snapshot, so its transcript is a
# *prefix* of the final one. The guess is reused only when the final begins
# with the partial (in order) AND the partial covers at least this fraction
# of the final words — the user did not add materially more after the
# snapshot. Lower = more reuse but answers more often from an incomplete
# request; higher = safer but less reuse.
PREEMPTIVE_MIN_COVERAGE = float(os.getenv("KANCHA_PREEMPTIVE_MIN_COVERAGE", "0.7"))
# Re-armed preemption: when a later snapshot of the SAME utterance adds at
# least this many new words over the partial we last speculated on, the
# coordinator supersedes its in-flight (or parked) guess with a fresh one
# generated from the fuller text. Word-set difference, so ASR rewording of
# already-seen words does not trigger a pointless re-arm.
PREEMPTIVE_REARM_WORDS = int(os.getenv("KANCHA_PREEMPTIVE_REARM_WORDS", "2"))


class _PreemptiveGuess:
    """A parked speculative generation awaiting authoritative validation."""

    __slots__ = ("partial", "text", "payload", "created_at", "gen")

    def __init__(
        self,
        partial: str,
        text: str,
        payload: dict[str, Any],
        created_at: float,
        gen: int,
    ) -> None:
        self.partial = partial
        self.text = text
        self.payload = payload
        self.created_at = created_at
        self.gen = gen


# LiveKit's approach: the preemptive transcript and the final transcript
# are compared as sets of words, casefolded and punctuation-stripped, so
# ASR rewording or a re-ordered clause doesn't defeat the match. On top of
# that we normalise common contractions both ways ("what's" == "what is"),
# since the two ASR passes often expand or contract them differently.
# Because the interim ASR runs mid-speech on a snapshot, its transcript is
# usually a *prefix* of the final one — handled by :func:`_is_partial_prefix`
# (order-preserving prefix match + coverage) alongside word-set equality.
_WORD_SPLIT_RE = re.compile(r"[^\w\s]")

_CONTRACTION_EXPANSIONS = {
    "whats": "what is",
    "what's": "what is",
    "dont": "do not",
    "don't": "do not",
    "doesnt": "does not",
    "doesn't": "does not",
    "didnt": "did not",
    "didn't": "did not",
    "isnt": "is not",
    "isn't": "is not",
    "cant": "cannot",
    "can't": "cannot",
    "wont": "will not",
    "won't": "will not",
    "wouldnt": "would not",
    "wouldn't": "would not",
    "couldnt": "could not",
    "couldn't": "could not",
    "shouldnt": "should not",
    "shouldn't": "should not",
    "youre": "you are",
    "you're": "you are",
    "theyre": "they are",
    "they're": "they are",
    "im": "i am",
    "i'm": "i am",
    "we're": "we are",
    "theres": "there is",
    "there's": "there is",
    "thats": "that is",
    "that's": "that is",
}


def _normalise_word_list(text: str) -> list[str]:
    """Casefold, expand common contractions and strip punctuation down to
    an ordered word list. Shared by the two transcript validators below."""
    tokens = text.casefold().split()
    expanded: list[str] = []
    for tok in tokens:
        expanded.extend(_CONTRACTION_EXPANSIONS.get(tok, tok).split())
    return [
        w
        for w in _WORD_SPLIT_RE.sub(" ", " ".join(expanded)).split()
        if w
    ]


def _transcripts_equivalent(a: str, b: str) -> bool:
    """True when two transcripts carry the same words, ignoring case,
    punctuation and contraction spelling (word-set equality)."""
    wa = set(_normalise_word_list(a))
    wb = set(_normalise_word_list(b))
    if not wa or not wb:
        return False
    return wa == wb


def _is_partial_prefix(partial: str, final: str, min_coverage: float) -> bool:
    """True when *final* begins with *partial* — in order, after the same
    normalisation as :func:`_transcripts_equivalent` — and *partial* covers
    at least ``min_coverage`` of the final words.

    The interim ASR runs mid-speech on a snapshot, so its transcript is a
    prefix of the authoritative one. Word-set equality (above) handles the
    exact-match case; this handles the snapshot-captured-early case while
    keeping the bar high enough that a substantially-extended utterance is
    discarded rather than answered from a stale prefix.
    """
    pa = _normalise_word_list(partial)
    fa = _normalise_word_list(final)
    if not pa or not fa:
        return False
    if len(pa) > len(fa):
        return False
    if fa[: len(pa)] != pa:
        return False
    return len(pa) / len(fa) >= min_coverage


def _partial_grew(prev: str, current: str, min_new_words: int) -> bool:
    """True when *current* adds at least ``min_new_words`` words over *prev*.

    Uses the word-set *difference* (after the same normalisation as the
    transcript validators) rather than a raw length delta, so an ASR that
    reworded a clause without adding content does not spuriously re-arm a
    preemptive guess.
    """
    wp = set(_normalise_word_list(prev))
    wc = set(_normalise_word_list(current))
    return len(wc - wp) >= min_new_words


def _controller_instructions() -> str:
    """Build the controller half of the system prompt.

    Rendered from :data:`tasks.registry.TASK_REGISTRY` (via
    :func:`planning.prompts.format_tool_catalog`) so the catalog the
    Conversation LLM reasons about is exactly the one the Executor
    enforces.
    """
    return f"""
CRITICAL SYSTEM INSTRUCTION — you are the controller of this assistant.

You decide, for every message, whether it is ordinary conversation or
whether it needs real work done. Nothing else in the system makes that
decision. When work is needed you delegate it by including a "task"
object in your JSON response; an execution layer you cannot see picks it
up, runs the tools, and reports back to you.

# Work the execution layer can do

{format_tool_catalog()}

# When the request needs one of those

1. Set "message" to a SHORT acknowledgement in the present continuous
   tense — "Checking the weather in Kathmandu now.", "Opening Firefox…",
   "Searching for that…". You are announcing that work has started.
2. Include a "task" object describing the work.
3. NEVER claim the work is done. No "Done.", "Completed.", "Message
   sent.", "Deleted.". You will be told the outcome afterwards and you
   will report it then. Equally, never say you are unable to do these
   things — you can, through the task layer.

# When it does not

Answer normally and omit "task" entirely. Greetings, opinions,
explanations, questions about yourself, and anything you can answer from
memory or the conversation are all ordinary conversation.

# Resolving references — this is your job alone

The execution layer never sees this conversation. It only sees the
"instruction" string you write. So a message like "do it", "go ahead",
"send it", "check that", "use the second one", "try again", "yes,
proceed" or "cancel that" must be resolved by YOU, from the conversation
above, into a complete instruction.

    User: Should I send this email to Dr. Rana?
    You:  The draft looks fine. I can send it whenever you like.
    User: Yeah, do that.
    task.instruction -> "Send the drafted email to Dr. Rana."
    task.context     -> {{"referenced_object": "previous_email"}}

If you genuinely cannot tell what "it" refers to, do NOT delegate — ask
the user which one they mean and omit "task".

# Writing the task object

- "instruction": one imperative sentence, self-contained. No pronouns
  referring to earlier turns, no "the one I mentioned".
- "task_type": the catalog name you believe fits, or omit it if unsure —
  the execution layer will choose and may override you.
- "parameters": concrete values you already know (city, app name, path,
  query). Omit what you do not know rather than inventing it.
- "expected_result": what you expect to get back, in a few words.
- "context": optional notes about what you resolved, e.g.
  {{"referenced_object": "previous_email"}}.
- "follow_up": true ONLY when you will need to look at the result and
  decide a further action yourself — e.g. "find the latest email from my
  professor and reply saying I'll attend", where the reply depends on
  what the search returns. Leave it false when the result just needs to
  be reported to the user.
- "mode": what this message does to the task already in progress. See
  below. Omit it (or use "new") when the user is starting something.

# Continuing a task that is already waiting

The execution layer can come back with a QUESTION instead of a result —
it may need a value it does not have, or approval before it changes
something. When that happens you ask the user, in your own words, and
the task stays open. Any block titled "Task currently in progress"
above tells you what is open and what it is waiting for.

The user's next message is then usually about THAT task, not a new one.
Set "mode" so the execution layer resumes it instead of starting over:

- "answer"  — they supplied the value that was asked for. Put it in
              "parameters" under the field name from the question.
              ("Tell him I'll submit tomorrow" → parameters {{"body": "..."}})
- "confirm" — they approved it. "yes", "go ahead", "do it", "send it".
- "reject"  — they declined it. "no", "don't", "not now".
- "cancel"  — they called the whole thing off. "cancel that", "forget it".
- "modify"  — they changed the task. "actually send it to bob@x.com" →
              mode "modify" with parameters {{"to": "bob@x.com"}}.
- "new"     — genuinely unrelated to the open task.

Rules that matter:

- When resuming, "instruction" may repeat the original request; the
  parameters and the mode are what carry the new information.
- NEVER answer a question the execution layer asked by inventing the
  value yourself. If the user has not said what the email should say,
  ask them — do not write it for them.
- NEVER set "mode" to "confirm" unless the user, in their own message,
  actually agreed. Your own judgement that something is fine is not the
  user's approval, and claiming it is will be rejected.
- If nothing is in progress, "mode" is "new" whatever the user says.

One "task" object per response. If a request needs several tool calls
that you can specify up-front, describe them all in a single
"instruction" — the execution layer decomposes and orders them itself.

# Reading results back

A turn beginning with "{TASK_RESULT_TURN_PREFIX}" is not the user
speaking. It is the execution layer reporting to you. Turn it into a
natural reply for the user, in your own voice, and — if the original
request needs another step — delegate that next step with a new "task".
Never read raw tool output or status codes aloud.

When such a turn says the layer is WAITING — for a value or for
approval — your reply is a question to the user and nothing else. Ask it
naturally ("What would you like the email to say?", "That'll send the
email to John — want me to go ahead?") and OMIT "task" entirely. The
task is already open; delegating again would start a second one.
"""


class ReasoningCoordinator:
    """Conversation LLM: intent, reference resolution, delegation, response."""

    def __init__(
        self,
        bus: EventBus,
        gemini_client: GeminiClient,
        memory_manager: MemoryManager,
        token_log: TokenLog | None = None,
        rag_manager: RAGManager | None = None,
        rag_router: MemoryRouter | None = None,
        activity_memory: ActivityMemory | None = None,
        ack_delay_s: float | None = None,
    ) -> None:
        self.bus = bus
        self.gemini_client = gemini_client
        self.memory_manager = memory_manager
        self.token_log = token_log
        # Both None when RAG is disabled or failed to start. Every use
        # site null-checks, so the conversational path is unchanged in
        # that case.
        self.rag_manager = rag_manager
        self.rag_router = rag_router
        # The isolated project-activity store. Queried separately from
        # rag_manager and rendered in its own prompt section, so project
        # history can never crowd conversation memory (and vice versa).
        self.activity_memory = activity_memory
        # task_id -> the TaskRequested we delegated. Popped when its
        # result comes back; tells us whether the result should be
        # chained (follow_up) or simply reported.
        self._in_flight: dict[str, TaskRequested] = {}
        # session_id -> a short record of the last delegation and how it
        # turned out. Injected into the prompt so "try again" and
        # "cancel that" resolve against something concrete.
        self._last_task: dict[str, dict[str, Any]] = {}
        # session_id -> how many tasks the current user turn has chained
        # into. Reset on every new user utterance.
        self._chain_depth: dict[str, int] = {}
        # task_id -> the timer that will speak the withheld acknowledgement
        # if the task is still running when it expires.
        self._pending_acks: dict[str, asyncio.Task] = {}
        self._ack_delay_s = (
            _ack_delay_from_env() if ack_delay_s is None else max(0.0, ack_delay_s)
        )
        self._response_stream_finished = asyncio.Event()
        self._response_stream_finished.set()
        # ── Preemptive generation (see on_partial_transcript) ──────────
        # session_id -> a parked _PreemptiveGuess, or None while its guess
        # task is still in flight. Popped when the authoritative turn lands.
        self._preemptive: dict[str, _PreemptiveGuess | None] = {}
        # session_id -> the in-flight guess task, so the authoritative turn
        # can await it briefly or cancel it.
        self._preemptive_tasks: dict[str, asyncio.Task] = {}
        # session_id -> (gen, last partial we spawned a guess for). Enables
        # re-armed preemption: a later snapshot of the SAME utterance that
        # grew enough supersedes the earlier guess (see on_partial_transcript).
        self._preemptive_partials: dict[str, tuple[int, str]] = {}
        # session_id -> monotonic utterance counter. Bumped on every user
        # turn; each guess is tagged with the generation it fired for, so a
        # late-arriving guess can never be misattributed to a later turn.
        self._utterance_gen: dict[str, int] = {}
        # Fire-and-forget tasks we must keep referenced (memory persist
        # moved off the critical path).
        self._bg_tasks: set[asyncio.Task] = set()
        # The task running the current user turn. Captured in on_user_input
        # so a barge-in (UserInterrupted) can cancel the in-flight LLM
        # generation and stop its partials from being spoken over the user.
        self._turn_task: asyncio.Task | None = None

    def register(self) -> None:
        """Subscribe the controller to user input and to task results.

        These are the only two inputs the system has. Note what is NOT
        here: no ``IntentIdentified``, no ``ReasoningRequested``. User
        input reaches exactly one LLM — this one.
        """
        self.bus.subscribe(TextInputReceived, self.on_user_input)
        self.bus.subscribe(TranscriptReady, self.on_user_input)
        self.bus.subscribe(PartialTranscriptReady, self.on_partial_transcript)
        self.bus.subscribe(TaskResultReady, self.on_task_result)
        self.bus.subscribe(UserInterrupted, self.on_user_interrupted)

    # ── Entry point: the user says something ──────────────────────────────────

    async def on_user_input(self, event: TextInputReceived | TranscriptReady) -> None:
        """Run one conversational turn, and delegate a task if one is needed."""
        text = (getattr(event, "text", "") or "").strip()
        if not text:
            return

        session_id = event.session_id
        self._turn_task = asyncio.current_task()

        # This is now the authoritative turn. Bump the utterance generation
        # and lift any preemptive state for it — a parked (or still
        # arriving) guess may be reused below if it validates.
        gen = self._utterance_gen.get(session_id, 0) + 1
        self._utterance_gen[session_id] = gen
        guess_task = self._preemptive_tasks.pop(session_id, None)
        parked = self._preemptive.pop(session_id, None)
        self._preemptive_partials.pop(session_id, None)

        self._response_stream_finished.clear()
        # A new utterance starts a fresh chain budget.
        self._chain_depth[session_id] = 0

        # Start thinking gate
        audio_state.thinking_started()

        reused = False
        try:
            self.memory_manager.short_term.add("user", text)
            if parked is not None or guess_task is not None:
                # There was (or is) a speculative guess for this utterance.
                # Await any in-flight generation briefly, then reuse it if
                # the authoritative transcript validates it.
                reused = await self._try_reuse_preemptive(
                    text,
                    session_id,
                    gen,
                    parked=parked,
                    task=guess_task,
                )
            if not reused:
                payload = await self._run_conversation_turn(
                    session_id=session_id,
                    retrieval_query=text,
                )
        finally:
            self._response_stream_finished.set()
            audio_state.thinking_finished()

        if reused:
            return

        # Delegation happens after the acknowledgement is out of the door,
        # so a long-running task never delays the immediate response.
        if not self._maybe_delegate(
            payload, session_id=session_id, user_request=text
        ):
            self._maybe_resume_open_task(text, session_id)

    async def on_user_interrupted(self, event: UserInterrupted) -> None:
        """The user barged in while the assistant was speaking.

        Cancel the in-flight generation of the interrupted turn. Its
        ``finally`` still releases the stream/thinking gates, and stopping
        it means no further ``PartialResponse``/``ResponseReady`` for that
        turn can be spoken over the user's new utterance. The barge-in's
        own transcript becomes the next authoritative turn as usual.
        """
        task = self._turn_task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    # ── Preemptive generation ───────────────────────────────────────────────

    async def on_partial_transcript(
        self, event: PartialTranscriptReady
    ) -> None:
        """Start a speculative reply on an interim transcript.

        Runs the *same* controller on the partial text — but with every
        side effect disabled. The result is parked, never streamed and
        never delegated; :meth:`_try_reuse_preemptive` decides whether the
        authoritative turn can use it. Any rejection here simply means the
        normal turn runs as if no speculation happened.
        """
        text = (event.text or "").strip()
        if not text:
            return
        session_id = event.session_id
        if not await self._should_preempt(session_id, text):
            return

        gen = self._utterance_gen.get(session_id, 0)

        # Re-armed preemption: within the SAME utterance, a later (fuller)
        # partial supersedes the previous guess only when the interim ASR
        # caught at least PREEMPTIVE_REARM_WORDS new words. Anything less
        # is a reword or truncation of what we already guessed on.
        last = self._preemptive_partials.get(session_id)
        if last is not None and last[0] == gen:
            if not _partial_grew(last[1], text, PREEMPTIVE_REARM_WORDS):
                return
        self._preemptive_partials[session_id] = (gen, text)

        # Supersede any in-flight guess — whether from an earlier utterance
        # or (re-arm) from this one's previous snapshot.
        prior = self._preemptive_tasks.pop(session_id, None)
        if prior is not None and not prior.done():
            prior.cancel()

        self._preemptive[session_id] = None  # in-flight marker
        task = asyncio.create_task(
            self._generate_preemptive(session_id, text, gen),
            name=f"preemptive:{session_id}",
        )
        self._preemptive_tasks[session_id] = task

    async def _should_preempt(self, session_id: str, text: str) -> bool:
        """Decide whether an interim transcript is worth speculating on.

        Every gate is conservative — a rejection simply falls through to
        the normal turn, so these checks only ever *skip* work, never
        degrade it.
        """
        if not self._response_stream_finished.is_set():
            return False  # a turn is already being processed
        # Imported lazily: nlu.classifier imports reasoning.llm_client,
        # which triggers reasoning/__init__ -> this module. A module-level
        # import here would form an import cycle.
        from nlu.classifier import classify_tool_request

        if classify_tool_request(text) is not None:
            return False  # task-like — the controller must decide this
        record = self._last_task.get(session_id) or {}
        if record.get("status") in WAITING_STATUSES:
            return False  # user is answering a task question — resume path
        if len(text.split()) < PREEMPTIVE_MIN_WORDS:
            return False
        if not self.gemini_client.has_available_key():
            return False  # never queue a guess behind a cooling key
        if self.rag_router is not None:
            # Never speculate when the turn would retrieve long-term
            # memory — a context-free guess could degrade recall if reused.
            try:
                decision = await self.rag_router.decide(text)
            except Exception:  # noqa: BLE001
                decision = None
            if decision is not None and decision.retrieve:
                return False
        return True

    async def _generate_preemptive(
        self, session_id: str, partial: str, gen: int
    ) -> None:
        """Generate a speculative reply on the interim transcript.

        Deliberately side-effect free: nothing is added to short-term
        memory, nothing is persisted, nothing is streamed to the user and
        nothing is ever delegated. The result is parked and only reused by
        :meth:`_try_reuse_preemptive` once the authoritative transcript
        validates it.
        """
        try:
            facts = await self.memory_manager.get_all_facts()
            memory_event = MemoryRetrieved(
                session_id=session_id,
                query=partial,
                structured_context=facts,
                episodic_context=[],
            )
            system = self._build_system_prompt(
                memory_event,
                retrieved=None,
                session_id=session_id,
                activity_chunks=None,
            )
            history = [*self._get_history(), {"role": "user", "content": partial}]

            response_text, payload, _ = await self._stream_once(
                history=history,
                system=system,
                session_id=session_id,
                call_site="preemptive",
                stream_to_user=False,
            )
            if not response_text or payload.get("task"):
                # A blank guess, or one that wanted to delegate — never
                # usable. A preemptive reply must never start work.
                self._preemptive[session_id] = None
                return
            self._preemptive[session_id] = _PreemptiveGuess(
                partial=partial,
                text=response_text,
                payload=payload,
                created_at=time.monotonic(),
                gen=gen,
            )
            logger.debug(
                "Preemptive reply parked (session=%s, %d chars, gen=%d)",
                session_id,
                len(response_text),
                gen,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Preemptive generation failed (non-fatal): %s", exc)
            self._preemptive[session_id] = None

    async def _try_reuse_preemptive(
        self,
        final_text: str,
        session_id: str,
        gen: int,
        parked: _PreemptiveGuess | None,
        task: asyncio.Task | None,
    ) -> bool:
        """Validate and reuse a preemptive guess, or fall through to the
        normal turn.

        A guess is trusted only when it is fresh, was generated for this
        utterance's generation, its transcript matches the authoritative
        one (word-set equality, or an order-preserving prefix covering most
        of it — the interim ASR runs mid-speech on a snapshot), and it
        never delegated a task (a preemptive reply must
        never start work).
        """
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(task), timeout=PREEMPTIVE_REUSE_WAIT_S
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()
                return False

        if parked is None:
            parked = self._preemptive.pop(session_id, None)
        if parked is None:
            return False

        if parked.gen + 1 != gen:
            return False
        if time.monotonic() - parked.created_at > PREEMPTIVE_MAX_AGE_S:
            return False
        if not (
            _transcripts_equivalent(parked.partial, final_text)
            or _is_partial_prefix(parked.partial, final_text, PREEMPTIVE_MIN_COVERAGE)
        ):
            logger.debug(
                "Preemptive guess discarded — transcript mismatch (session=%s)",
                session_id,
            )
            return False
        if parked.payload.get("task"):
            logger.info(
                "Preemptive guess discarded — it delegated a task (session=%s)",
                session_id,
            )
            return False
        # The final transcript is authoritative. If the completed utterance
        # turns out to be a task or would retrieve long-term memory, the
        # preemptive guess — generated from the incomplete partial without
        # any task/retrieval context — must not stand in for the real turn.
        from nlu.classifier import classify_tool_request

        if classify_tool_request(final_text) is not None:
            logger.debug(
                "Preemptive guess discarded — final is a task (session=%s)",
                session_id,
            )
            return False
        if self.rag_router is not None:
            try:
                decision = await self.rag_router.decide(final_text)
            except Exception:  # noqa: BLE001
                decision = None
            if decision is not None and decision.retrieve:
                logger.debug(
                    "Preemptive guess discarded — final would retrieve "
                    "(session=%s)",
                    session_id,
                )
                return False

        self.memory_manager.short_term.add("assistant", parked.text)
        self._spawn_persist(parked.payload, session_id=session_id)
        self.bus.emit(ResponseReady(text=parked.text, session_id=session_id))
        logger.info(
            "Preemptive reply reused (session=%s, lead_time=%.2fs)",
            session_id,
            time.monotonic() - parked.created_at,
        )
        return True

    def _maybe_resume_open_task(self, text: str, session_id: str) -> None:
        """Route a bare yes/no to the open task the controller forgot.

        The controller is *supposed* to answer "yes" with a task carrying
        ``mode: "confirm"``. When it instead replies conversationally —
        which models do, because "yes" reads like small talk — the user's
        approval would evaporate and the task would sit waiting forever.
        This is the net: an unambiguous yes or no, and a task actually
        waiting on one, is enough to resume it.

        Deliberately narrow. Anything with more in it than assent or
        refusal is left to the controller, which can see what the extra
        words changed.
        """
        record = self._last_task.get(session_id) or {}
        if record.get("status") not in WAITING_STATUSES:
            return
        task_id = str(record.get("task_id") or "")
        if not task_id:
            return

        if looks_like_assent(text):
            mode = "confirm"
        elif looks_like_refusal(text):
            mode = "reject"
        else:
            return

        logger.info(
            "Controller attached no task to a bare %r — resuming task %s as %s",
            text,
            task_id,
            mode,
        )
        record["status"] = "running"
        self.bus.emit(
            TaskRequested(
                task_id=task_id,
                task_type=str(record.get("task_type") or ""),
                instruction=str(record.get("instruction") or ""),
                user_request=text,
                mode=mode,
                resume_task_id=task_id,
                session_id=session_id,
            )
        )

    # ── Entry point: the Task LLM reports back ────────────────────────────────

    async def on_task_result(self, event: TaskResultReady) -> None:
        """Turn a structured Task Result into the final user-facing response.

        The Task LLM returns execution data only — status plus per-tool
        output. Deciding what the user hears, including how a failure is
        explained, happens here.
        """
        session_id = event.session_id
        request = self._in_flight.pop(event.task_id, None) if event.task_id else None
        if event.task_id:
            # The outcome is here, so the "starting now" line is redundant.
            self._cancel_ack(event.task_id)

        record = self._last_task.get(session_id)
        if record is not None and record.get("task_id") == event.task_id:
            record["status"] = event.status
            record["error"] = event.error
            record["question"] = event.question
            record["missing_fields"] = list(event.missing_fields)
            record["description"] = event.description
            # Keep the unabridged tool output. The user hears a short
            # spoken summary with no ids in it, but "trash the second
            # one" still has to resolve to a real message id — so the
            # data is remembered even though it is never said.
            record["raw"] = _compact_raw(event.results)

        # The acknowledgement may still be streaming; never talk over it.
        await self._response_stream_finished.wait()

        # The orchestrator is waiting on the user. Ask them — this is the
        # only route a question from the execution layer has to a person.
        if event.status in WAITING_STATUSES:
            await self._ask_on_behalf_of_task(event)
            return

        audio_state.thinking_started()

        try:
            depth = self._chain_depth.get(session_id, 0)
            if request is not None and request.follow_up:
                if depth < MAX_TASK_CHAIN_DEPTH:
                    # Chaining: hand the result back to the conversation
                    # so the controller can decide the next action itself.
                    self._chain_depth[session_id] = depth + 1
                    await self._continue_after_result(event, request)
                    return
                logger.warning(
                    "Task chain for session %s hit the depth cap (%d) — "
                    "reporting the result instead of chaining again",
                    session_id,
                    MAX_TASK_CHAIN_DEPTH,
                )

            response_text = await naturalize_plan_response(
                llm=self.gemini_client,
                user_request=event.user_request or event.instruction,
                task_results=list(event.results or []),
                status=event.status,
                token_log=self.token_log,
            )

            self.memory_manager.short_term.add("assistant", response_text)
            self.bus.emit(ResponseReady(text=response_text, session_id=session_id))
        finally:
            audio_state.thinking_finished()

    async def _continue_after_result(
        self, event: TaskResultReady, request: TaskRequested
    ) -> None:
        """Chained turn: feed a task result back through the controller.

        Used when the Conversation LLM flagged the delegation as
        ``follow_up`` — i.e. it needs to *see* the outcome before it can
        decide what to do next ("find the email, then reply to it").
        The response it generates may carry another ``task``, which is
        delegated exactly like a first-turn one.
        """
        self._response_stream_finished.clear()
        try:
            self.memory_manager.short_term.add(
                "user", self._format_result_turn(event)
            )
            payload = await self._run_conversation_turn(
                session_id=event.session_id,
                retrieval_query=request.instruction or event.instruction,
            )
        finally:
            self._response_stream_finished.set()

        self._maybe_delegate(
            payload,
            session_id=event.session_id,
            user_request=request.user_request or event.user_request,
        )

    async def _ask_on_behalf_of_task(self, event: TaskResultReady) -> None:
        """Put the execution layer's question to the user, in our voice.

        The Task LLM is not allowed to address the user, so a question it
        raises arrives here as ``waiting_for_input`` /
        ``waiting_for_confirmation`` and is asked as an ordinary
        conversational turn. The task stays open in the Orchestrator;
        the user's reply comes back through :meth:`on_user_input` with a
        ``mode`` that resumes it.

        Any ``task`` the model attaches to this turn is dropped. The task
        is already open — delegating here would start a second one, which
        is precisely the "two unrelated tasks" bug this layer exists to
        prevent.
        """
        session_id = event.session_id
        self._last_task[session_id] = {
            "task_id": event.task_id,
            "task_type": event.task_type,
            "instruction": event.instruction,
            "status": event.status,
            "error": "",
            "question": event.question,
            "missing_fields": list(event.missing_fields),
            "description": event.description,
        }

        audio_state.thinking_started()
        self._response_stream_finished.clear()
        try:
            self.memory_manager.short_term.add("user", self._format_result_turn(event))
            payload = await self._run_conversation_turn(
                session_id=session_id,
                retrieval_query=event.instruction or event.user_request,
            )
        finally:
            self._response_stream_finished.set()
            audio_state.thinking_finished()

        if payload.get("task"):
            logger.info(
                "Ignoring a delegation on the question turn for task %s — "
                "that task is already open and awaiting the user",
                event.task_id,
            )
            # The controller broke the question-turn contract: it
            # re-delegated instead of asking. Its message is an
            # acknowledgement, not a question — speaking it would tell
            # the user the work is starting while the task is actually
            # parked awaiting approval. Re-ask deterministically.
            self._reask_on_behalf_of_task(session_id)
            return

        # A well-behaved question turn carries no task, so
        # _run_conversation_turn already delivered its message (the
        # question) to the user. Nothing more to say here.

    def _reask_on_behalf_of_task(self, session_id: str) -> None:
        """Repeat the pending question, deterministically — no LLM call.

        The controller is not allowed to re-delegate on a question turn,
        but it sometimes does anyway. When it does, its message is an
        acknowledgement rather than the question, so this asks the exact
        question the execution layer already recorded instead of speaking
        a false "starting now" while the task waits.
        """
        record = self._last_task.get(session_id) or {}
        question = str(record.get("question") or "").strip()
        description = str(record.get("description") or "").strip()
        if question:
            text = f"Sorry — {question}"
        elif description:
            text = f"Sorry, should I go ahead and {description}, or not?"
        else:
            text = "Sorry, should I go ahead with that, or not?"
        self.memory_manager.short_term.add("assistant", text)
        self._emit_ack(session_id, text)

    def _format_result_turn(self, event: TaskResultReady) -> str:
        """Render a Task Result as an internal history turn.

        Prefixed with :data:`TASK_RESULT_TURN_PREFIX`, which
        :func:`_controller_instructions` teaches the model to read as
        "the execution layer is reporting", never as user speech.
        """
        lines = [f"{TASK_RESULT_TURN_PREFIX} status={event.status}"]
        if event.instruction:
            lines.append(f"requested: {event.instruction}")

        if event.status == "waiting_for_input":
            lines.append(
                "The execution layer needs more information before it can "
                "continue. Ask the user this, in your own words, and do not "
                "answer it yourself:"
            )
            lines.append(f"question: {event.question}")
            if event.missing_fields:
                lines.append(f"missing: {', '.join(event.missing_fields)}")
            return "\n".join(lines)

        if event.status == "waiting_for_confirmation":
            lines.append(
                "This changes something, so it needs the user's approval "
                "before it runs. Tell them plainly what will happen and ask "
                "whether to go ahead:"
            )
            lines.append(f"about to: {event.description}")
            return "\n".join(lines)

        for entry in event.results or []:
            tool = str(entry.get("tool", "")).strip() or "tool"
            output = str(entry.get("result", "")).strip()
            if output:
                lines.append(f"{tool}: {output}")
        if event.error:
            lines.append(f"error: {event.error}")
        return "\n".join(lines)

    # ── Delegation ────────────────────────────────────────────────────────────

    def _maybe_delegate(
        self, payload: dict[str, Any], session_id: str, user_request: str
    ) -> bool:
        """Emit :class:`TaskRequested` if the controller asked for work.

        Returns True when a task was delegated. The Task LLM is only ever
        reached through this method.
        """
        task = payload.get("task")
        if not isinstance(task, dict):
            return False

        instruction = str(task.get("instruction") or "").strip()
        task_type = str(task.get("task_type") or "").strip()
        if not instruction and not task_type:
            return False
        if not instruction:
            # A bare task_type is still actionable; the raw request is the
            # best instruction we have for it.
            instruction = user_request

        parameters = task.get("parameters")
        context = task.get("context")

        # Resuming reuses the open task's id, so the Orchestrator's state,
        # this class's in-flight map and any pending acknowledgement all
        # keep talking about the same task across turns.
        mode = str(task.get("mode") or "new").strip().lower()
        record = self._last_task.get(session_id) or {}
        open_task_id = str(record.get("task_id") or "")
        resuming = mode in RESUME_MODES and bool(open_task_id)
        task_id = open_task_id if resuming else f"task-{uuid.uuid4().hex[:8]}"

        request = TaskRequested(
            task_id=task_id,
            task_type=task_type,
            instruction=instruction,
            parameters=dict(parameters) if isinstance(parameters, dict) else {},
            expected_result=str(task.get("expected_result") or "").strip(),
            context=dict(context) if isinstance(context, dict) else {},
            follow_up=bool(task.get("follow_up", False)),
            user_request=user_request,
            mode=mode if resuming else "new",
            resume_task_id=open_task_id if resuming else "",
            session_id=session_id,
        )

        self._in_flight[request.task_id] = request
        self._last_task[session_id] = {
            "task_id": request.task_id,
            "task_type": request.task_type,
            "instruction": request.instruction,
            "status": "running",
            "error": "",
        }

        logger.info(
            "Delegating to the Task LLM | task_id=%s type=%s follow_up=%s | %r",
            request.task_id,
            request.task_type or "(unspecified)",
            request.follow_up,
            request.instruction,
        )
        self._schedule_ack(request, str(payload.get("_ack") or "").strip())
        self.bus.emit(request)
        return True

    def _schedule_ack(self, request: TaskRequested, text: str) -> None:
        """Speak the acknowledgement only if the task turns out to be slow.

        "Opening Firefox…" followed a beat later by "Firefox is open, sir."
        is two confirmations of one action. So a withheld acknowledgement
        waits out :attr:`_ack_delay_s`: if the result lands first the user
        hears the outcome alone, and if it doesn't, the acknowledgement
        goes out so a long tool call isn't silence.
        """
        if not text:
            return

        if self._ack_delay_s <= 0:
            self._emit_ack(request.session_id, text)
            return

        async def _wait_then_ack() -> None:
            try:
                await asyncio.sleep(self._ack_delay_s)
            except asyncio.CancelledError:
                return
            # Re-check under no intervening await: on_task_result pops the
            # request the moment the result arrives, so this is race-free.
            if request.task_id not in self._in_flight:
                return
            self._pending_acks.pop(request.task_id, None)
            logger.debug(
                "Task %s still running after %.1fs — speaking the acknowledgement",
                request.task_id,
                self._ack_delay_s,
            )
            self._emit_ack(request.session_id, text)

        self._pending_acks[request.task_id] = asyncio.create_task(
            _wait_then_ack(), name=f"task_ack:{request.task_id}"
        )

    def _emit_ack(self, session_id: str, text: str) -> None:
        self.bus.emit(PartialResponse(text="", done=True, session_id=session_id))
        self.bus.emit(ResponseReady(text=text, session_id=session_id))

    def _cancel_ack(self, task_id: str) -> None:
        """Drop a withheld acknowledgement — its result got here first."""
        pending = self._pending_acks.pop(task_id, None)
        if pending is not None and not pending.done():
            pending.cancel()

    # ── Conversational turn ───────────────────────────────────────────────────

    async def _run_conversation_turn(
        self, session_id: str, retrieval_query: str
    ) -> dict[str, Any]:
        """Stream one controller response and return its parsed envelope.

        The caller is responsible for having added the triggering turn to
        short-term memory first. Emits ``PartialResponse`` while the text
        streams and ``ResponseReady`` at the end, then returns the parsed
        ``{message, task?, sql?, rag?}`` payload so the caller can act on
        any delegation.

        A turn that delegates is the exception: its message is only an
        acknowledgement, so it is returned under ``_ack`` **unsent** and
        :meth:`_schedule_ack` decides whether the user ever hears it.
        """
        # 1. Retrieve SQLite facts
        facts = await self.memory_manager.get_all_facts()
        memory_event = MemoryRetrieved(
            session_id=session_id,
            query=retrieval_query,
            structured_context=facts,
            episodic_context=[],
        )

        # 2. Retrieve RAG chunks
        retrieved = await self._retrieve_rag_context(retrieval_query, intent_event=None)

        # 2b. Retrieve project activity from its own store (same router
        # gate, separate index — see _retrieve_activity_context).
        activity = await self._retrieve_activity_context(retrieval_query)

        # 3. Build the system prompt: persona + memory + controller rules
        system = self._build_system_prompt(
            memory_event, retrieved, session_id, activity_chunks=activity
        )

        # 4. Stream the reply (history already ends with the current turn)
        history = self._get_history()

        logger.debug(
            "Calling Conversation LLM (streaming) | history_turns=%d | system_len=%d | rag_chunks=%d",
            len(history),
            len(system),
            len(retrieved),
        )

        response_text, payload, _ = await self._stream_once(
            history=history,
            system=system,
            session_id=session_id,
            call_site="conversational",
        )

        # If the stream returned nothing usable — first chunk won the race
        # but no real text followed (Gemini occasionally streams just the
        # opening JSON and then stalls), or the call raised before any
        # chunk arrived — retry once before giving up. Without this, the
        # mic opens into dead silence and the user sees the assistant
        # "ignore" their question.
        if not response_text:
            logger.info("Empty LLM stream (session=%s); retrying once", session_id)
            response_text, payload, _ = await self._stream_once(
                history=history,
                system=system,
                session_id=session_id,
                call_site="conversational-retry",
            )

        # Last resort: surface a user-visible fallback so the mic doesn't
        # open into dead silence. TTS will speak this short apology and
        # the mic reopens for the user's next attempt.
        if not response_text:
            logger.error(
                "LLM produced no usable text after retry (session=%s); surfacing fallback",
                session_id,
            )
            response_text = (
                "Sorry, sir — my response came back blank. Could you say that again?"
            )

        # 5. Add assistant turn to short_term SYNCHRONOUSLY
        self.memory_manager.short_term.add("assistant", response_text)

        # 6. Persist memory carried by the response envelope, off the
        # critical path — the user is waiting to hear the reply, and these
        # writes are best-effort by construction. `sql` lands in SQLite,
        # `rag` in the vector store (+ the rag.txt audit log).
        self._spawn_persist(payload, session_id=session_id)

        # 7. Deliver — unless this turn is an acknowledgement of work that
        # is about to start. In that case the text is handed to
        # _schedule_ack, which speaks it only if the task outlives the
        # grace period. This must hold even when the model streams the
        # message before the "task" key (violating the format rules): an
        # acknowledgement is never spoken directly, or a misbehaving model
        # could claim the work is starting while the task is actually
        # parked awaiting confirmation.
        if payload.get("task"):
            payload["_ack"] = response_text
            return payload

        self.bus.emit(PartialResponse(text="", done=True, session_id=session_id))
        self.bus.emit(ResponseReady(text=response_text, session_id=session_id))

        return payload

    async def _stream_once(
        self,
        history: list[dict[str, str]],
        system: str,
        session_id: str,
        call_site: str,
        stream_to_user: bool = True,
    ) -> tuple[str, dict[str, Any], bool]:
        """Stream the JSON envelope once.

        Returns ``(visible text, payload, streamed)`` where ``streamed``
        says whether the text was forwarded to the user as it generated.

        The envelope streams token-by-token. While it arrives we strip the
        scaffolding and forward ONLY the visible ``message`` text as
        ``PartialResponse`` events so the UI renders words as they
        generate — unless ``stream_to_user`` is False (preemptive
        generation), in which case the text is collected but never
        surfaced: a speculative reply must not be spoken before it is
        validated. The full buffer is parsed once at the end for the
        authoritative message plus the ``task``/``sql``/``rag`` fields.

        The exception is a turn that opens with a ``"task"`` key. That
        message is an acknowledgement of work about to start, and we may
        end up never showing it (see :meth:`_schedule_ack`) — so it is
        held back rather than streamed. The format instructions require
        ``task`` to come first precisely so this decision can be made
        before the first visible character is committed to; if a model
        ignores that and streams the message first, we simply stream it
        and the acknowledgement is shown as usual.
        """
        raw_parts: list[str] = []
        buffer = ""
        prev_visible = ""
        # None until the envelope reveals which key came first.
        delegating: bool | None = None
        try:
            async for chunk in self.gemini_client.generate_with_history_stream(
                history=history,
                system=system,
                hedge_width=1,
                call_site=call_site,
            ):
                raw_parts.append(chunk)
                buffer += chunk

                if delegating is None:
                    if '"message"' in buffer:
                        delegating = False
                    elif '"task"' in buffer:
                        delegating = True

                if delegating:
                    continue

                visible = extract_streamed_message(buffer)
                if len(visible) > len(prev_visible):
                    if stream_to_user:
                        self.bus.emit(
                            PartialResponse(
                                text=visible[len(prev_visible) :],
                                session_id=session_id,
                            )
                        )
                    prev_visible = visible
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Streaming Conversation LLM call failed (session=%s): %s",
                session_id,
                exc,
            )

        raw_response = "".join(raw_parts)
        payload = parse_memory_response(raw_response)
        response_text = payload.get("message", "").strip()

        # Decide whether the parsed message is *actually* the message, or
        # scaffolding that leaked through because the envelope did not
        # parse. Reading `{"task": {"task_type": ...` aloud is the worst
        # outcome this method has, so the check is deliberately broad:
        # anything that still smells like the envelope is discarded, even
        # at the cost of falling back to a blank turn.
        #
        # We deliberately do NOT discard plain English — the streaming
        # client yields a single prose chunk ("I'm having trouble thinking
        # right now…") on an upstream exception, and that should be said.
        if looks_like_envelope(response_text):
            logger.warning(
                "Discarding envelope scaffolding that reached the response "
                "text (session=%s, %d chars)",
                session_id,
                len(response_text),
            )
            response_text = ""

        if not response_text:
            # Two recoveries before giving up: whatever was already shown
            # to the user, then a direct scrape of the message field out
            # of the raw buffer — which works even when the surrounding
            # JSON is too broken to parse.
            response_text = prev_visible or extract_streamed_message(raw_response)
            if looks_like_envelope(response_text):
                response_text = ""

        return response_text, payload, bool(prev_visible)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_system_prompt(
        self,
        memory_event: MemoryRetrieved | None,
        retrieved: list[RetrievedChunk] | None = None,
        session_id: str | None = None,
        activity_chunks: list[RetrievedChunk] | None = None,
    ) -> str:
        """
        Build the system prompt: persona + facts + memory + controller rules.

        Layers, deliberately kept apart:

        * **persona** — who the assistant is.
        * **User facts** — structured key/value memory from SQLite.
        * **Relevant long-term memory** — semantic chunks from the vector
          store, present only when the Memory Router asked for retrieval
          and something scored above the similarity threshold.
        * **Recent project activity** — a separate section fed from the
          isolated activity store; only when the same router gate fires.
        * **Recent action** — what was last delegated and how it went, so
          "try again" / "cancel that" resolve against something concrete.
        * **Controller instructions** — the tool catalog and the rules for
          delegating work.

        Conversation history is passed separately as a message list —
        it must NEVER appear here as "structured facts", which confuses
        the LLM into anchoring on old turns.
        """
        parts = [JARVIS_PERSONA, current_datetime_block()]

        if memory_event:
            # Only inject items that have a key+value structure — these are
            # explicit user facts (e.g. name, language preference).
            # Recent interaction rows from SQLite (which have only "content")
            # are intentionally excluded: they belong in the message list.
            fact_lines = [
                f"- {item['key']}: {item['value']}"
                for item in memory_event.structured_context
                if "key" in item and "value" in item
            ]
            if fact_lines:
                parts.append("User facts:\n" + "\n".join(fact_lines))

        rag_block = self._format_rag_context(retrieved)
        if rag_block:
            parts.append(rag_block)

        activity_block = self._format_activity_context(activity_chunks)
        if activity_block:
            parts.append(activity_block)

        action_block = self._format_last_task(session_id)
        if action_block:
            parts.append(action_block)

        parts.append(_controller_instructions().strip())

        return "\n\n".join(parts)

    def _format_last_task(self, session_id: str | None) -> str:
        """Render the last delegation so follow-ups have something to bind to.

        This is the state that makes "try again", "did that work?" and
        "cancel that" resolvable *here* rather than by handing the Task
        LLM the whole conversation and hoping.
        """
        if not session_id:
            return ""
        record = self._last_task.get(session_id)
        if not record:
            return ""
        status = record.get("status", "unknown")
        if status in WAITING_STATUSES:
            # An open task changes what the next message probably means,
            # so it gets a heading the model cannot skim past.
            lines = [
                "Task currently in progress — the user's next message is "
                "most likely about THIS, so set \"mode\" accordingly:",
                f"- instruction: {record.get('instruction', '')}",
                f"- waiting for: "
                + (
                    "the user's approval"
                    if status == "waiting_for_confirmation"
                    else "information from the user"
                ),
            ]
            if record.get("question"):
                lines.append(f"- you asked: {record['question']}")
            if record.get("missing_fields"):
                lines.append(
                    f"- fields still needed: {', '.join(record['missing_fields'])}"
                )
            if record.get("description"):
                lines.append(f"- about to: {record['description']}")
            return "\n".join(lines)

        lines = [
            "Most recent action you delegated (for resolving follow-ups "
            "like \"try again\" or \"cancel that\"):",
            f"- instruction: {record.get('instruction', '')}",
            f"- status: {status}",
        ]
        task_type = record.get("task_type")
        if task_type:
            lines.insert(1, f"- type: {task_type}")
        error = record.get("error")
        if error:
            lines.append(f"- error: {error}")

        raw = record.get("raw")
        if raw:
            lines.append("")
            lines.append(
                "Full data it returned — REFERENCE ONLY. The user has "
                "already been given a short spoken summary of this. Never "
                "read it back, never recite ids, timestamps or paths from "
                "it. Use it only to resolve what the user refers to next "
                "(\"the second one\", \"the GitHub one\") into concrete "
                "parameters for your next task:"
            )
            lines.append(raw)
        return "\n".join(lines)

    def _format_rag_context(self, retrieved: list[RetrievedChunk] | None) -> str:
        """Render retrieved chunks into a prompt block.

        This is the Conversation Manager's job, not the RAG Manager's —
        which is why it takes structured :class:`RetrievedChunk` objects
        and produces the string, rather than receiving a pre-baked
        prompt fragment.

        The block is budgeted by ``KANCHA_RAG_MAX_CONTEXT_CHARS``.
        Chunks arrive sorted by score, so truncation always drops the
        least relevant material first.
        """
        if not retrieved:
            return ""

        budget = 4000
        if self.rag_manager is not None:
            budget = self.rag_manager.config.max_context_chars

        lines: list[str] = []
        used = 0
        included = 0

        for index, chunk in enumerate(retrieved, start=1):
            source = f", source: {chunk.source}" if chunk.source else ""
            page = chunk.metadata.get("page")
            page_note = f", page {page}" if page else ""
            header = (
                f"[{index}] {chunk.title} "
                f"({chunk.doc_type}, relevance {chunk.score:.2f}{source}{page_note})"
            )
            body = chunk.content.strip()
            entry = f"{header}\n{body}"

            if used + len(entry) > budget:
                if included == 0:
                    # Always include at least one chunk, truncated to fit,
                    # rather than retrieving something and then showing
                    # the model nothing.
                    entry = entry[:budget].rstrip() + " […]"
                    lines.append(entry)
                    included += 1
                break

            lines.append(entry)
            used += len(entry)
            included += 1

        if not lines:
            return ""

        if included < len(retrieved):
            logger.debug(
                "RAG context truncated to %d/%d chunk(s) by the %d-char budget",
                included,
                len(retrieved),
                budget,
            )

        return (
            "Relevant long-term memory (retrieved for this message):\n\n"
            + "\n\n".join(lines)
            + "\n\nUse this material only where it genuinely answers the user. "
            "Ignore anything irrelevant, and never mention that you looked "
            "anything up or refer to these numbered entries."
        )

    def _format_activity_context(
        self, retrieved: list[RetrievedChunk] | None
    ) -> str:
        """Render project-activity chunks into their own prompt section.

        Kept separate from :meth:`_format_rag_context` on purpose:
        activity records are project-work history, not conversational
        memory, and the model should treat them as such. The section has
        its own small budget (activity summaries are short by
        construction), so project history never eats into the
        conversation-memory budget either.
        """
        if not retrieved:
            return ""

        budget = 1500
        lines: list[str] = []
        used = 0
        for index, chunk in enumerate(retrieved, start=1):
            header = (
                f"[{index}] {chunk.title} "
                f"(project memory, relevance {chunk.score:.2f})"
            )
            entry = f"{header}\n{chunk.content.strip()}"
            if used + len(entry) > budget:
                if not lines:
                    entry = entry[:budget].rstrip() + " […]"
                else:
                    break
            lines.append(entry)
            used += len(entry)

        if not lines:
            return ""
        return (
            "Recent project activity (from project memory):\n\n"
            + "\n\n".join(lines)
        )

    async def _retrieve_activity_context(
        self, user_input: str
    ) -> list[RetrievedChunk]:
        """Retrieve project-activity hits from the isolated store.

        Uses the *same* router gate as the main retrieval, so nothing new
        is ever queried for messages the router would skip. The two
        stores are searched independently — activity hits are additive
        and can never alter what the main store returns.
        """
        if self.activity_memory is None or self.rag_router is None:
            return []
        try:
            decision = await self.rag_router.decide(user_input)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory router failed (%s) — skipping project memory", exc)
            return []
        if not decision.retrieve:
            return []
        try:
            return await self.activity_memory.search(decision.query or user_input)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Project memory retrieval failed (%s) — answering without it", exc
            )
            return []

    async def _retrieve_rag_context(
        self, user_input: str, intent_event: IntentIdentified | None
    ) -> list[RetrievedChunk]:
        """Ask the router whether to retrieve, then retrieve if so.

        Fully guarded: retrieval enriches a reply, so any failure here
        must degrade the answer rather than break the turn.
        """
        if self.rag_manager is None or self.rag_router is None:
            return []

        try:
            decision = await self.rag_router.decide(
                user_input,
                intent=intent_event.intent.value if intent_event else None,
                is_task=bool(intent_event.requires_task) if intent_event else False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory router failed (%s) — skipping retrieval", exc)
            return []

        if not decision.retrieve:
            logger.debug("RAG skipped: %s", decision.reason)
            return []

        logger.info("RAG retrieval triggered: %s", decision.reason)
        try:
            return await self.rag_manager.retrieve(decision.query or user_input)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG retrieval failed (%s) — answering without it", exc)
            return []

    def _get_history(self) -> list[dict[str, str]]:
        """Return short_term buffer as a clean [{role, content}] list."""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.memory_manager.short_term.get_recent()
        ]

    def _spawn_persist(self, payload: dict[str, Any], session_id: str) -> None:
        """Persist response-carried memory off the critical path.

        Writes the ``sql``/``rag`` memory the primary Gemini response
        carried, in the background: the user is waiting to hear the reply,
        and these writes are best-effort by construction. Errors are
        logged, never surfaced.
        """
        if not payload:
            return

        async def _safe() -> None:
            try:
                await self._persist_response_memory(payload, session_id=session_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Background memory persist failed (non-fatal): %s", exc
                )

        task = asyncio.create_task(_safe(), name="memory_persist")
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _persist_response_memory(
        self, payload: dict[str, Any], session_id: str = "default"
    ) -> None:
        """Persist sql/rag memory carried by the primary Gemini response.

        Best-effort: a failure to persist must never break the reply.
        Absent keys ("sql"/"rag") are simply skipped.

        ``rag`` entries go to two places with different standing:

        * the **vector database** — the authoritative semantic store,
          written through the RAG Manager so chunking, embedding and
          duplicate detection all apply;
        * ``memory/rag.txt`` — a human-readable audit log, for
          inspection and debugging only. Nothing ever reads it back.
        """
        sql = payload.get("sql")
        if sql:
            try:
                saved = await self.memory_manager.save_sql(sql)
                if saved:
                    logger.info("Persisted %d SQL fact(s) from response", saved)
            except Exception as exc:
                logger.warning("SQL memory persist failed (non-fatal): %s", exc)

        rag = payload.get("rag")
        if not rag:
            return

        # 1. Authoritative store.
        if self.rag_manager is not None:
            try:
                results = await self.rag_manager.index_conversation_entries(
                    rag, session_id=session_id
                )
                indexed = sum(r.chunks_indexed for r in results)
                if indexed:
                    logger.info(
                        "Indexed %d RAG chunk(s) from response into the vector store",
                        indexed,
                    )
            except Exception as exc:
                logger.warning("RAG vector index failed (non-fatal): %s", exc)
        else:
            logger.debug("RAG disabled — response rag entries go to the audit log only")

        # 2. Audit log (unchanged behaviour).
        try:
            appended = await self.memory_manager.append_rag(rag)
            if appended:
                logger.info("Appended %d RAG entries to rag.txt (audit log)", appended)
        except Exception as exc:
            logger.warning("RAG audit append failed (non-fatal): %s", exc)
