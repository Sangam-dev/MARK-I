"""Pydantic models for the WebSocket envelope and REST request/response bodies.

Keeping these in one place makes the wire format easy to audit against
answers/integration_plan.md and answers/guide.md.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ── WebSocket ────────────────────────────────────────────────────────────────


class WSIncoming(BaseModel):
    """Envelope for messages sent from the frontend to the backend over `/ws`.

    type:
        "user_text"   -> payload: {"text": str}. Injected onto the bus as a
                         TextInputReceived event (same as CLI --text mode).
        "retry_last"  -> payload: {} (re-runs the last task; coordinator's
                         retry-phrase regex already understands "retry it").
        "ping"        -> payload: {}. Server replies with "pong" (used by the
                         frontend for a crude WS round-trip latency reading).
    """

    type: Literal["user_text", "retry_last", "ping"]
    payload: dict[str, Any] = Field(default_factory=dict)
    session_id: str = "default"


# ── REST: settings ───────────────────────────────────────────────────────────


class SettingsModel(BaseModel):
    """Persisted user-facing settings (memory/data/settings.json).

    NOTE: `tts_enabled` here only controls *future* process starts today —
    TTSHandler is registered once at pipeline build time. Toggling this at
    runtime does not hot-swap the handler yet (see answers/guide.md,
    "known limitations").
    """

    tts_enabled: bool = True
    voice_mode: bool = False
    session_id: str = "default"


# ── REST: alarms ─────────────────────────────────────────────────────────────


class AlarmCreateRequest(BaseModel):
    command: str
    delay_seconds: int | None = None


# ── REST: generic action result envelope ────────────────────────────────────


class ActionResult(BaseModel):
    success: bool
    message: str
