"""FastAPI + WebSocket server exposing the KANCHA pipeline to jarvis_frontend.

Run with (from the `kancha/` directory):

    uv run uvicorn api.server:app --host 127.0.0.1 --port 8765 --reload

The Electron main process (electron/main.ts) spawns this automatically in the
packaged app — see answers/guide.md for the full picture and for how to wire
up a new backend feature end-to-end.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from api.bridge import attach_bridge
from api.routes import files, health, history, tasks, weather
from api.routes import memory as memory_routes
from api.routes import settings as settings_routes
from api.schemas import WSIncoming
from api.ws_manager import manager
from core.events import AssistantState, AssistantStateChanged, TextInputReceived
from core.pipeline import Pipeline, build_pipeline, shutdown_pipeline
from core.project_logging import setup_logging

setup_logging()
logger = logging.getLogger("kancha.api.server")

# Single-user desktop app: one process == one session. See
# answers/integration_plan.md section 6 ("single session assumption").
DEFAULT_SESSION_ID = os.getenv("KANCHA_SESSION_ID", "default")
DATA_DIR = Path(__file__).resolve().parent.parent / "memory" / "data"


def _tts_enabled_from_env() -> bool:
    return os.getenv("KANCHA_TTS_ENABLED", "true").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def _monitor_enabled_from_env() -> bool:
    """Background system monitor is on by default in the API server.

    On a desktop session the loop is cheap (~30s psutil polls against a
    300s alert cooldown) and the alerts surface through the same
    WebSocket that already pipes ``ResponseReady`` events to the front-end,
    so we don't expect this to be a noise concern. Set
    ``KANCHA_MONITOR_ENABLED=0`` to opt out for debugging.
    """
    return os.getenv("KANCHA_MONITOR_ENABLED", "true").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def _voice_input_enabled_from_env() -> bool:
    return os.getenv("KANCHA_VOICE_ENABLED", "true").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def _start_voice_input(pipeline: Pipeline) -> asyncio.Task | None:
    """Start continuous microphone listening automatically, if possible.

    Mirrors `main.py --voice` (continuous, no wake word) exactly — same
    MicrophoneListener class, same event flow — so "talk to jarvis" works
    the moment the backend (and therefore the Electron app) starts, with no
    manual toggle. Requires the `groq` package and a GROQ_API_KEY; if either
    is missing this logs a warning and the server keeps running fine with
    text-only chat.
    """
    if not _voice_input_enabled_from_env():
        logger.info("Voice input disabled via KANCHA_VOICE_ENABLED=0")
        return None

    if not os.getenv("GROQ_API_KEY"):
        logger.warning(
            "GROQ_API_KEY not set — voice input disabled; text chat still works. "
            "Set GROQ_API_KEY in kancha/.env to enable talking to JARVIS."
        )
        return None

    try:
        from input.stt import MicrophoneListener
    except ImportError as exc:
        logger.warning(
            "Voice input unavailable (groq package not installed): %s. "
            "Text chat still works. Install with: uv add groq",
            exc,
        )
        return None

    mic = MicrophoneListener(
        bus=pipeline.bus,
        session_id=pipeline.session_id,
        wake_word_gated=False,  # continuous listening, no wake word required
    )
    mic.register()
    task = asyncio.create_task(mic.run(), name="microphone")
    logger.info(
        "Continuous microphone listening started automatically (session=%s)",
        pipeline.session_id,
    )
    return task


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = await build_pipeline(
        session_id=DEFAULT_SESSION_ID,
        data_dir=DATA_DIR,
        enable_tts=_tts_enabled_from_env(),
        enable_console_formatter=True,
        enable_system_monitor=_monitor_enabled_from_env(),
    )
    attach_bridge(pipeline)

    mic_task = _start_voice_input(pipeline)

    app.state.pipeline = pipeline
    app.state.voice_available = mic_task is not None
    app.state.mic_task = mic_task

    logger.info(
        "KANCHA API server ready (session=%s, tts=%s, voice=%s)",
        DEFAULT_SESSION_ID,
        pipeline.tts_enabled,
        app.state.voice_available,
    )
    try:
        yield
    finally:
        if mic_task is not None and not mic_task.done():
            mic_task.cancel()
            try:
                await asyncio.wait_for(mic_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
        await shutdown_pipeline(app.state.pipeline)


app = FastAPI(title="KANCHA API", lifespan=lifespan)

# This server is only ever bound to 127.0.0.1 and talked to by the local
# Electron renderer (which sends Origin: null / file://, or http://localhost
# during `pnpm dev`) — wide-open CORS is safe here because it never leaves
# the local machine. See answers/guide.md, "Security posture".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(memory_routes.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(tasks.router, prefix="/api")
app.include_router(files.router, prefix="/api")
app.include_router(weather.router, prefix="/api")
app.include_router(settings_routes.router, prefix="/api")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    pipeline: Pipeline = websocket.app.state.pipeline
    await manager.connect(websocket)
    try:
        while True:
            raw = await websocket.receive_json()
            try:
                message = WSIncoming.model_validate(raw)
            except ValidationError as exc:
                await websocket.send_json(
                    {
                        "type": "error",
                        "payload": {"message": str(exc), "recoverable": True},
                        "session_id": DEFAULT_SESSION_ID,
                    }
                )
                continue

            session_id = message.session_id or pipeline.session_id

            if message.type == "ping":
                await websocket.send_json(
                    {"type": "pong", "payload": {}, "session_id": session_id}
                )
                continue

            if message.type == "user_text":
                text = str(message.payload.get("text", "")).strip()
                if not text:
                    continue
                # No "listening" phase for typed input — jump straight to
                # thinking. TTSHandler (or the bridge's fallback, if TTS is
                # disabled) will bring the state back to idle once the
                # response is ready.
                pipeline.bus.emit(
                    AssistantStateChanged(
                        state=AssistantState.THINKING, session_id=session_id
                    )
                )
                pipeline.bus.emit(TextInputReceived(text=text, session_id=session_id))
                continue

            if message.type == "retry_last":
                pipeline.bus.emit(
                    AssistantStateChanged(
                        state=AssistantState.THINKING, session_id=session_id
                    )
                )
                # ReasoningCoordinator's `_is_retry_request` regex recognizes
                # this phrase and re-runs the last concrete task.
                pipeline.bus.emit(
                    TextInputReceived(text="retry", session_id=session_id)
                )
                continue

    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket)
