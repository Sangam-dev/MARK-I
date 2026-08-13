"""Memory Router — decides *whether* a turn needs semantic retrieval.

Retrieving for every message is wrong on three counts: it wastes an
embedding call and a vector search on "hi", it pollutes the prompt with
irrelevant context that measurably degrades replies, and it adds latency
to the turns that are most sensitive to it (short commands).

So retrieval is gated. The router is a strategy object with one method,
:meth:`MemoryRouter.decide`, returning a :class:`RouteDecision`. Today
the default implementation is a weighted keyword heuristic; swapping it
for a small classifier model later means writing one new subclass and
changing ``KANCHA_RAG_ROUTER`` — no caller changes, because the
Conversation Manager only ever sees ``RouteDecision``.

Worked examples (see ``tests`` for the executable version)::

    "Hi"                                          -> no  (greeting)
    "Open Chrome"                                 -> no  (task intent)
    "Continue my transformer research."           -> yes (continuity + possessive topic)
    "What debugging solution did we use last week?" -> yes (past-reference + shared memory)
    "Summarize the uploaded networking PDF."       -> yes (document reference)
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .config import RAGConfig

logger = logging.getLogger("kancha.memory.rag.router")


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Outcome of a routing decision.

    ``query`` exists so a future router can rewrite the retrieval query
    (expand pronouns, strip filler) without the caller knowing. The
    heuristic router passes the text through unchanged.
    """

    retrieve: bool
    reason: str
    confidence: float = 0.0
    query: str = ""
    signals: tuple[str, ...] = field(default_factory=tuple)


class MemoryRouter(ABC):
    """Strategy interface for the retrieve/skip decision."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def decide(
        self,
        text: str,
        *,
        intent: str | None = None,
        is_task: bool = False,
    ) -> RouteDecision:
        """Return whether *text* warrants a vector-store lookup."""


# ── Signal patterns ──────────────────────────────────────────────────
#
# Grouped by what they indicate, with a weight. Weights are summed and
# compared against _RETRIEVE_THRESHOLD. Keeping them as data (rather than
# nested ifs) is what makes the heuristic auditable and tunable.

_GREETING_RE = re.compile(
    r"^\s*(?:hi|hey|hello|yo|sup|hiya|howdy|good\s+(?:morning|afternoon|evening|night)|"
    r"thanks|thank\s+you|thx|ok|okay|cool|nice|great|sure|yes|no|yep|nope|"
    r"bye|goodbye|see\s+you|good\s?night|how\s+are\s+you|what'?s\s+up)"
    r"[\s!.,?]*$",
    re.IGNORECASE,
)

# Explicit reference to previously stored knowledge or an uploaded file.
_DOCUMENT_RE = re.compile(
    r"\b(?:uploaded|attachment|attached|the\s+(?:pdf|document|doc|file|paper|report|"
    r"manual|spec|guide|book|article)|my\s+(?:notes?|documents?|files?|papers?)|"
    r"from\s+the\s+(?:pdf|document|doc|file|paper|report))\b",
    re.IGNORECASE,
)

# Reference to shared history: "we discussed", "last week", "you told me".
#
# The verb list is deliberately broad. A false positive costs one
# embedding call; a false negative makes the assistant look like it has
# amnesia about work the user knows it recorded, which is far worse.
_PAST_REFERENCE_RE = re.compile(
    r"\b(?:(?:we|i)\s+(?:discussed|talked|decided|used|did|solved|tried|agreed|"
    r"built|found|made|created|added|wrote|chose|picked|fixed|changed|planned|"
    r"implemented|integrated|designed|set\s+up|finished|completed|started)|"
    r"you\s+(?:told|said|mentioned|suggested|showed|explained)|"
    r"i\s+(?:told|said|mentioned|asked|showed)\s+you|"
    r"last\s+(?:week|month|time|night|session)|earlier|previously|before|"
    r"the\s+other\s+day|remember|recall|back\s+then|we'?d|we'?ve|i'?ve)\b",
    re.IGNORECASE,
)

# The natural way to ask about your own history is an inverted question:
# "what did we do…", "how did we integrate…", "did I finish…", "what have
# I been working on". Auxiliary inversion puts the verb *before* the
# pronoun, so :data:`_PAST_REFERENCE_RE`, which expects "we|i" followed by
# a verb, never matched any of them. That single gap silently blocked most
# episodic questions — including the retrieval prefetch, which is gated by
# the same decision.
_EPISODIC_QUESTION_RE = re.compile(
    r"\b(?:"
    r"(?:what|when|where|why|how|which|who)\s+(?:did|have|has|had|was|were)\s+"
    r"(?:i|we|you)\b"
    r"|(?:did|have|has|had)\s+(?:i|we|you)\s+\w+"
    r"|(?:i|we)\s+(?:have\s+|had\s+)?been\s+\w+ing"
    r"|what\s+(?:i|we|you)\s+(?:did|have|said|built|made)"
    r")",
    re.IGNORECASE,
)

# Continuity: picking a thread back up.
_CONTINUITY_RE = re.compile(
    r"\b(?:continue|resume|pick\s+up|carry\s+on|go\s+back\s+to|where\s+(?:we|i)\s+left|"
    r"keep\s+going\s+(?:on|with)|follow\s+up\s+on|update\s+(?:me\s+)?on)\b",
    re.IGNORECASE,
)

# Possessive knowledge: the user's own long-lived work.
_POSSESSIVE_TOPIC_RE = re.compile(
    r"\b(?:my|our)\s+(?:research|project|thesis|paper|study|notes?|work|code|"
    r"design|architecture|setup|config|stack|plan|roadmap|experiment|analysis|"
    r"implementation|progress|findings?|learnings?)\b",
    re.IGNORECASE,
)

# Knowledge-domain nouns that suggest a substantive, retrievable topic.
_KNOWLEDGE_NOUN_RE = re.compile(
    r"\b(?:research|architecture|algorithm|implementation|debugging|bug|error|"
    r"solution|approach|design|decision|documentation|summary|summarize|summarise|"
    r"explain|analysis|benchmark|experiment|finding|conclusion|tradeoff|trade-off|"
    r"transformer|embedding|model|dataset|pipeline|deployment|migration|refactor|"
    r"protocol|specification|requirement|technique|methodology)\b",
    re.IGNORECASE,
)

_QUESTION_RE = re.compile(
    r"^\s*(?:what|why|how|when|where|which|who|whose|whom|can|could|would|should|"
    r"did|do|does|is|are|was|were|tell\s+me|explain|describe|summarize|summarise|"
    r"give\s+me|show\s+me|list)\b",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[A-Za-z0-9'-]+")

# (pattern, weight, label)
_SIGNALS: tuple[tuple[re.Pattern[str], float, str], ...] = (
    (_DOCUMENT_RE, 1.0, "document-reference"),
    (_PAST_REFERENCE_RE, 1.0, "past-reference"),
    # Same weight as an explicit past reference: "what did we do about X"
    # is asking about shared history just as plainly as "we discussed X".
    (_EPISODIC_QUESTION_RE, 1.0, "episodic-question"),
    (_CONTINUITY_RE, 0.8, "continuity"),
    (_POSSESSIVE_TOPIC_RE, 0.8, "possessive-topic"),
    (_KNOWLEDGE_NOUN_RE, 0.4, "knowledge-noun"),
)

# Sum of matched weights needed to trigger retrieval. 0.8 means any one
# strong signal fires, while a lone knowledge noun (0.4) is not enough on
# its own — it needs a question or a second signal to reach the bar.
_RETRIEVE_THRESHOLD = 0.8

# Below this many words a message is treated as chit-chat or a command.
_MIN_WORDS = 3


class HeuristicRouter(MemoryRouter):
    """Weighted keyword router. Fast, deterministic, no model call."""

    @property
    def name(self) -> str:
        return "heuristic"

    async def decide(
        self,
        text: str,
        *,
        intent: str | None = None,
        is_task: bool = False,
    ) -> RouteDecision:
        cleaned = (text or "").strip()

        if not cleaned:
            return RouteDecision(False, "empty input", query=cleaned)

        # 1. Device actions never need semantic memory. "Open Chrome"
        #    is fully specified by its own words.
        if is_task or (intent or "").lower() == "task":
            return RouteDecision(False, "task intent", query=cleaned)

        # 2. Greetings and acknowledgements.
        if _GREETING_RE.match(cleaned):
            return RouteDecision(False, "greeting or acknowledgement", query=cleaned)

        words = _WORD_RE.findall(cleaned)
        if len(words) < _MIN_WORDS:
            return RouteDecision(
                False, f"too short ({len(words)} words)", query=cleaned
            )

        # 3. Weighted signal scan.
        score = 0.0
        matched: list[str] = []
        for pattern, weight, label in _SIGNALS:
            if pattern.search(cleaned):
                score += weight
                matched.append(label)

        is_question = bool(_QUESTION_RE.match(cleaned) or cleaned.endswith("?"))
        # A question shape amplifies weak topical signals without being
        # sufficient alone — "how are you?" must not trigger retrieval.
        if is_question and matched:
            score += 0.4
            matched.append("question")

        if score >= _RETRIEVE_THRESHOLD:
            confidence = min(1.0, score / 2.0)
            decision = RouteDecision(
                True,
                f"matched {', '.join(matched)}",
                confidence=confidence,
                query=cleaned,
                signals=tuple(matched),
            )
        else:
            decision = RouteDecision(
                False,
                f"no retrieval signal (score={score:.1f})",
                confidence=0.0,
                query=cleaned,
                signals=tuple(matched),
            )

        logger.debug(
            "Route %s: %s (%.60r)",
            "RETRIEVE" if decision.retrieve else "SKIP",
            decision.reason,
            cleaned,
        )
        return decision


class AlwaysRouter(MemoryRouter):
    """Always retrieve. Useful for evaluating recall during development."""

    @property
    def name(self) -> str:
        return "always"

    async def decide(
        self, text: str, *, intent: str | None = None, is_task: bool = False
    ) -> RouteDecision:
        cleaned = (text or "").strip()
        if not cleaned or is_task:
            return RouteDecision(False, "empty or task", query=cleaned)
        return RouteDecision(True, "router=always", confidence=1.0, query=cleaned)


class NeverRouter(MemoryRouter):
    """Never retrieve. Kill switch that leaves indexing intact."""

    @property
    def name(self) -> str:
        return "never"

    async def decide(
        self, text: str, *, intent: str | None = None, is_task: bool = False
    ) -> RouteDecision:
        return RouteDecision(False, "router=never", query=(text or "").strip())


def build_router(config: RAGConfig) -> MemoryRouter:
    """Instantiate the configured routing strategy."""
    if config.router_strategy == "always":
        return AlwaysRouter()
    if config.router_strategy == "never":
        return NeverRouter()
    return HeuristicRouter()
