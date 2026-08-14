"""Configuration for the OpenCode execution layer.

Every tunable lives here and is sourced from the environment with a
documented default — nothing about OpenCode is hardcoded at its call
site. The config object is built once in :func:`core.pipeline.build_pipeline`
and threaded into the client, exactly like :class:`memory.rag.config.RAGConfig`.

Environment variables (all optional, all prefixed ``KANCHA_OPENCODE_``)::

    KANCHA_OPENCODE_ENABLED          1|0    (default 1)
    KANCHA_OPENCODE_URL              url    (default "" — spawn a local server)
    KANCHA_OPENCODE_HOST             host   (default 127.0.0.1)
    KANCHA_OPENCODE_PORT             int    (default 0 — let the OS pick)
    KANCHA_OPENCODE_BINARY           path   (default opencode)
    KANCHA_OPENCODE_PROVIDER         str    (default opencode)
    KANCHA_OPENCODE_MODEL            str    (default deepseek-v4-flash-free)
    KANCHA_OPENCODE_AGENT            str    (default build)
    KANCHA_OPENCODE_WORKSPACE        path   (default ~/kancha-workspace)
    KANCHA_OPENCODE_TIMEOUT_S        s      (default 1800)
    KANCHA_OPENCODE_STARTUP_TIMEOUT_S s     (default 45)
    KANCHA_OPENCODE_MAX_SESSIONS     int    (default 8)
    KANCHA_OPENCODE_PROGRESS_INTERVAL_S s   (default 90, 0 disables)

Two things are worth knowing about the defaults.

**The workspace is not this repository.** OpenCode's ``build`` agent
edits files and runs commands for real, and "analyse this project and fix
the performance problems" is a request the user is expected to make. If
the default working directory were the assistant's own source tree, a
delegated task could rewrite the assistant mid-conversation. It defaults
to a separate directory instead; point it at a project deliberately.

**The default model costs nothing and needs no key.** Delegation runs on
OpenCode's own free tier (``opencode/deepseek-v4-flash-free``), so it does not spend
the assistant's Gemini quota and does not depend on any key in ``.env``.
Point ``KANCHA_OPENCODE_PROVIDER``/``_MODEL`` elsewhere and OpenCode will
discover that provider's credentials from the environment on its own
(``GEMINI_API_KEY``, ``GROQ_API_KEY``, ``OPENAI_API_KEY``, …) — the
server is a child process, so it inherits the same ``.env`` the assistant
loaded. That is why no key is named in this file and none is ever sent
over the wire by us.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("kancha.agent.config")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("%s=%r is not an integer — using %d", name, raw, default)
        return default


def _env_float(name: str, default: float, *, low: float, high: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return min(high, max(low, float(raw)))
    except ValueError:
        logger.warning("%s=%r is not a number — using %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return default


@dataclass(frozen=True, slots=True)
class OpenCodeConfig:
    """Immutable configuration for the OpenCode integration."""

    # ── Lifecycle ────────────────────────────────────────────────────
    enabled: bool = True

    # An explicit URL means "an OpenCode server is already running, use
    # it". Empty means "spawn and own one" — see OpenCodeClient.
    server_url: str = ""
    host: str = "127.0.0.1"
    #: 0 lets the OS assign a free port. The real one is read back from
    #: the server's own startup line, so a busy port is never a failure.
    port: int = 0
    binary: str = "opencode"

    # ── Model selection ──────────────────────────────────────────────
    # OpenCode's own default, on its hosted Zen gateway's free tier: no
    # API key, no cost, 200k context. DeepSeek V4 Flash Free scores higher
    # on SWE-bench than the previous default (big-pickle) at the same
    # context size. The assistant's Gemini keys are a finite shared pool
    # already split between the Conversation and Task LLMs (see
    # reasoning/llm_client_mulapi.py), and a delegated build is thousands
    # of tokens over many turns — putting that on the same pool would
    # starve the conversation to run the agent.
    #
    # Not every Zen model is free. The paid ones (claude-sonnet-4-6,
    # gpt-5.x, …) return "No payment method" unless one is on file, which
    # arrives as an agent_error, not a crash.
    provider: str = "opencode"
    model: str = "deepseek-v4-flash-free"
    #: OpenCode agent profile: "build" edits files and runs commands,
    #: "plan" is read-only. Research tasks work under either.
    agent: str = "build"

    # ── Where the work happens ───────────────────────────────────────
    workspace: Path = Path.home() / "kancha-workspace"

    # ── Budgets ──────────────────────────────────────────────────────
    # A delegated build is minutes of work, not seconds — this is not a
    # conversational latency budget and must not be tuned like one. It
    # exists so a wedged agent eventually returns control.
    #
    # Nobody waits on this any more: the run happens in a background task
    # and its progress is streamed (agent/tool.py), so a longer ceiling
    # costs nothing in responsiveness and a whole front-end build no
    # longer gets cut off at ten minutes with the work half done.
    request_timeout_s: float = 1800.0
    startup_timeout_s: float = 45.0
    #: Cap on tracked sessions, so a long-running assistant does not
    #: accumulate them without bound.
    max_sessions: int = 8

    #: How often to volunteer progress on a run in flight, in seconds.
    #: A build that says nothing for ten minutes is indistinguishable
    #: from a crash, so the assistant speaks up periodically instead of
    #: waiting to be asked — but only when something actually changed
    #: since the last one (see agent/tool.py), so a thinking agent does
    #: not produce a stream of identical updates. 0 disables it.
    progress_interval_s: float = 90.0

    @property
    def spawns_server(self) -> bool:
        """True when we are responsible for starting (and stopping) it."""
        return not self.server_url.strip()

    @classmethod
    def from_env(cls) -> "OpenCodeConfig":
        """Build a config from environment variables.

        Every fallback below must match the dataclass field default
        above. They are two separate defaults for the same knob, and
        when they drift, changing one has no effect at runtime — see
        ``tests/test_opencode.py::from_env_fallbacks_match_the_field_defaults``.
        """
        workspace = _env_str("KANCHA_OPENCODE_WORKSPACE", "")
        return cls(
            enabled=_env_bool("KANCHA_OPENCODE_ENABLED", True),
            server_url=_env_str("KANCHA_OPENCODE_URL", "").rstrip("/"),
            host=_env_str("KANCHA_OPENCODE_HOST", "127.0.0.1"),
            port=_env_int("KANCHA_OPENCODE_PORT", 0, minimum=0),
            binary=_env_str("KANCHA_OPENCODE_BINARY", "opencode"),
            provider=_env_str("KANCHA_OPENCODE_PROVIDER", "opencode"),
            model=_env_str("KANCHA_OPENCODE_MODEL", "deepseek-v4-flash-free"),
            agent=_env_str("KANCHA_OPENCODE_AGENT", "build"),
            workspace=(
                Path(workspace).expanduser()
                if workspace
                else Path.home() / "kancha-workspace"
            ),
            request_timeout_s=_env_float(
                "KANCHA_OPENCODE_TIMEOUT_S", 1800.0, low=10.0, high=7200.0
            ),
            startup_timeout_s=_env_float(
                "KANCHA_OPENCODE_STARTUP_TIMEOUT_S", 45.0, low=2.0, high=300.0
            ),
            max_sessions=_env_int("KANCHA_OPENCODE_MAX_SESSIONS", 8, minimum=1),
            progress_interval_s=_env_float(
                "KANCHA_OPENCODE_PROGRESS_INTERVAL_S", 90.0, low=0.0, high=3600.0
            ),
        )

    def describe(self) -> str:
        """One-line summary for startup logs."""
        where = self.server_url or f"spawn {self.binary} on {self.host}:{self.port or 'auto'}"
        return (
            f"model={self.provider}/{self.model} agent={self.agent} "
            f"workspace={self.workspace} server=({where}) "
            f"timeout={self.request_timeout_s:.0f}s"
        )
