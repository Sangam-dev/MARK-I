from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

logger = logging.getLogger("kancha.reasoning.groq_voice")

DEFAULT_MODEL = os.getenv("KANCHA_GROQ_VOICE_MODEL", "openai/gpt-oss-20b")

# Measured round trip on llama-3.1-8b-instant was 0.2–0.45s, so this is
# not a latency budget — it is a stall guard. It sits well above the
# normal case because the cost of tripping it is the thing the user
# complained about in the first place: the raw listing, ids and all,
# read out loud. A cold first call has been seen to take seconds.
DEFAULT_TIMEOUT_S = float(os.getenv("KANCHA_GROQ_VOICE_TIMEOUT", "6.0"))

# Long enough for a two-sentence answer plus the occasional list of three.
MAX_TOKENS = 220

# Tool output beyond this is summarised even if it looks clean — nobody
# wants a 40-line process table read out.
_LONG_OUTPUT_CHARS = 220

# Machine debris that should never be spoken: hex ids, pids, byte counts,
# absolute paths, device nodes, timestamps with timezones.
_MACHINE_MARKERS = re.compile(
    r"(?:\bid=|\bpid\b|\bmessage_id\b|\buid\b|0x[0-9a-f]+|\b[0-9a-f]{12,}\b|"
    r"\b\d{2}-[A-Za-z]{3}-\d{4}\b|[+-]\d{4}\b|"
    r"/(?:home|usr|var|etc|proc|dev|sys|mnt|opt|tmp)/)",
    re.IGNORECASE,
)

# Numbers dense enough to be a readout rather than a sentence.
# "Brightness set to 70%." has one and is fine to say as-is;
# "421.7 GB total, 268.3 GB used, 131.9 GB available (68% used)" has four
# and is a table read aloud.
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_MAX_BARE_NUMBERS = 2

SYSTEM_PROMPT = (
    "You are KANCHA, a British AI assistant in the style of JARVIS. You are "
    "speaking out loud to one person — your reply is read by a text-to-speech "
    "voice, so write what a person would say, never what a terminal would print.\n"
    "\n"
    "Rules:\n"
    "1. Lead with the answer. One or two sentences. Only go longer when the "
    "user explicitly asked to hear something in full, such as the contents of "
    "an email or a document.\n"
    "2. Never speak machine identifiers: message ids, pids, uids, hashes, "
    "hex addresses (0x...), byte counts, absolute file paths, raw timestamps "
    "or timezone offsets. They are handled elsewhere. Replace them with "
    "'the process', 'it', or 'the one you asked about' — never read them "
    "out loud.\n"
    "3. Never enumerate a long list. Give the count, then the one or two that "
    "matter. 'Fourteen unread — the newest is from GitHub about your token.'\n"
    "4. Round and humanise numbers: 'about 40 percent', 'just under 8 gigs', "
    "'a couple of minutes ago'.\n"
    "5. No markdown, no bullet points, no headings, no emoji, no status "
    "codes, no preamble like 'Here is'. Plain spoken prose.\n"
    "6. State outcomes plainly. If something failed, say what failed and why "
    "in ordinary words — do not read the error text back.\n"
    "7. Do not invent anything that is not in the tool output. If it is "
    "empty, say so briefly.\n"
    "\n"
    "Address the user as 'sir' occasionally — not in every sentence."
)


def needs_naturalizing(text: str) -> bool:
    """True if *text* would sound wrong read aloud.

    Short, already-conversational output ("Firefox is open, sir.") skips
    the model entirely — that is the common case and it should stay free
    and instant.
    """
    if not text or not text.strip():
        return False
    stripped = text.strip()
    if len(stripped) > _LONG_OUTPUT_CHARS:
        return True
    if "\n" in stripped:
        return True
    if _MACHINE_MARKERS.search(stripped):
        return True
    return len(_NUMBER_RE.findall(stripped)) > _MAX_BARE_NUMBERS


class GroqVoice:
    """Rewrites tool output as speech. Never raises, never blocks long."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT_S,
        client: Any = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self._client = client
        self._init_done = client is not None

    # ── client ────────────────────────────────────────────────────────

    def _ensure_client(self) -> Any:
        if self._init_done:
            return self._client

        self._init_done = True
        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            logger.info(
                "No GROQ_API_KEY — tool output will use the deterministic voice"
            )
            return None
        try:
            from groq import Groq  # noqa: PLC0415

            self._client = Groq(api_key=key)
            logger.debug("Groq voice ready (%s)", self.model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Groq voice init failed: %s", exc)
            self._client = None
        return self._client

    @property
    def available(self) -> bool:
        return self._ensure_client() is not None

    # ── the one public call ───────────────────────────────────────────

    async def speak(
        self,
        user_request: str,
        tool_output: str,
        status: str = "completed",
        fallback: str = "",
    ) -> str:
        """Return a spoken-shape reply, or *fallback* if anything fails."""
        fallback = (fallback or tool_output or "").strip()
        client = self._ensure_client()
        if client is None or not tool_output.strip():
            return fallback

        try:
            spoken = await asyncio.wait_for(
                asyncio.to_thread(
                    self._complete, user_request, tool_output, status
                ),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Groq voice timed out after %.1fs — using raw output", self.timeout)
            return fallback
        except Exception as exc:  # noqa: BLE001
            logger.warning("Groq voice failed: %s — using raw output", exc)
            return fallback

        spoken = (spoken or "").strip()
        if not spoken:
            return fallback
        # A model that echoes the machine text back has not done the job.
        if _MACHINE_MARKERS.search(spoken) and not _MACHINE_MARKERS.search(
            user_request
        ):
            logger.info("Groq voice returned machine text — using raw output")
            return fallback
        return spoken

    def _complete(self, user_request: str, tool_output: str, status: str) -> str:
        """Blocking Groq call. Runs in a worker thread."""
        note = "" if status == "completed" else f"\nOutcome: {status}"
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f'The user said: "{user_request.strip()}"\n\n'
                        f"What the tools returned:\n{tool_output.strip()}"
                        f"{note}\n\n"
                        "Say this back to them out loud."
                    ),
                },
            ],
            # gpt-oss models reason by default; "low" keeps the reasoning
            # tokens small so MAX_TOKENS is spent on the spoken answer, not
            # the thinking (medium/high can return empty content and trip
            # the raw-output fallback). "none" is qwen3-only.
            reasoning_effort="low",
            max_tokens=MAX_TOKENS,
            temperature=0.3,
        )
        return response.choices[0].message.content or ""


class NullVoice(GroqVoice):
    """A voice that never calls out. For tests and for GROQ-less setups."""

    def __init__(self) -> None:
        super().__init__(client=None)
        self._init_done = True

    def _ensure_client(self) -> Any:
        return None


# ── Shared instance ───────────────────────────────────────────────────

_shared_voice: GroqVoice | None = None


def get_shared_voice() -> GroqVoice:
    """The process-wide voice, created on first use.

    Set ``KANCHA_GROQ_VOICE=0`` to turn the phrasing pass off entirely
    and keep the deterministic tool voice.
    """
    global _shared_voice
    if _shared_voice is None:
        if os.getenv("KANCHA_GROQ_VOICE", "1").strip() in ("0", "false", "off"):
            logger.info("Groq voice disabled by KANCHA_GROQ_VOICE")
            _shared_voice = NullVoice()
        else:
            _shared_voice = GroqVoice()
    return _shared_voice


def set_shared_voice(voice: GroqVoice | None) -> None:
    """Swap the shared voice. For tests and for wiring at startup."""
    global _shared_voice
    _shared_voice = voice
