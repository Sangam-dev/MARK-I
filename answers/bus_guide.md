# KANCHA `core/bus.py` — Complete Guide & AI Assistant Integration

> **File:** `core/bus.py`  
> **Pattern:** Async Pub/Sub Event Bus  
> **Role:** The central nervous system of KANCHA — every module talks to every other module exclusively through here.

---

## Table of Contents

1. [What is the Event Bus?](#1-what-is-the-event-bus)
2. [How `bus.py` Works — Line by Line](#2-how-buspy-works--line-by-line)
3. [The Event System (`core/events.py`)](#3-the-event-system-coreevents-py)
4. [Core Bus Methods — Reference](#4-core-bus-methods--reference)
5. [The Full AI Pipeline Flow](#5-the-full-ai-pipeline-flow)
6. [Complete Integration Example — Wiring It All Together](#6-complete-integration-example--wiring-it-all-together)
7. [Module-by-Module Integration Guide](#7-module-by-module-integration-guide)
8. [Error Handling Through the Bus](#8-error-handling-through-the-bus)
9. [Testing Your Integrations](#9-testing-your-integrations)
10. [Advanced Patterns](#10-advanced-patterns)

---

## 1. What is the Event Bus?

The `EventBus` is a **publish/subscribe (pub/sub) message broker** that runs entirely inside Python's `asyncio` event loop.

Instead of modules calling each other directly (tight coupling), every module:
- **Emits events** when something happens (e.g., "transcript is ready")
- **Subscribes to events** it cares about (e.g., "handle transcript ready")

This means the STT module never imports the NLU module, the NLU module never imports the LLM module — they all only know about **events**.

```
┌─────────┐    emit(TranscriptReady)    ┌─────────┐
│   STT   │ ─────────────────────────► │  Bus    │ ──► NLU handler
│ Module  │                             │         │ ──► Memory handler  
└─────────┘                             └─────────┘
```

---

## 2. How `bus.py` Works — Line by Line

```python
class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[type, list[Handler]] = defaultdict(list)
        self._tasks: set[asyncio.Task] = set()
```

- `_handlers` — maps an **event class** → list of **async functions** that handle it.
- `_tasks` — tracks all running `asyncio.Task`s so they can be awaited on shutdown (`drain()`).

---

### `subscribe(event_type, handler)`

```python
def subscribe(self, event_type: Type[E], handler: Handler) -> None:
    if not asyncio.iscoroutinefunction(handler):
        raise TypeError(...)
    self._handlers[event_type].append(handler)
```

Registers a handler for an event type. **Only async functions are accepted** — sync functions raise `TypeError` immediately. This is enforced up-front so you get a clear error at registration time, not silently at runtime.

---

### `emit(event)` — Fire and Forget

```python
def emit(self, event: BaseEvent) -> None:
    handlers = self._handlers.get(type(event), [])
    for handler in handlers:
        task = asyncio.create_task(
            self._run_handler(handler, event),
            name=f"{handler.__name__}:{event.event_id[:8]}"
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
```

- Creates an `asyncio.Task` for **each registered handler** — they all run **concurrently**.
- Returns immediately — the caller does not wait for handlers to finish.
- Task names include the handler name and first 8 chars of the event UUID (great for debugging with `asyncio` task introspection).
- Completed tasks remove themselves from `_tasks` via `discard` callback — no memory leak.

**When to use `emit()`:** Any time you want fire-and-forget behavior. STT emits a transcript and moves on; the NLU module will handle it whenever the loop gets to it.

---

### `emit_and_wait(event)` — Synchronous-style

```python
async def emit_and_wait(self, event: BaseEvent) -> None:
    await asyncio.gather(
        *[self._run_handler(h, event) for h in handlers],
        return_exceptions=True
    )
```

Runs **all handlers concurrently** but waits for all of them to finish before returning. Use this when you need to know that downstream processing is done before you continue (e.g., in tests, or in a sequential pipeline step).

---

### `_run_handler(handler, event)` — Crash Isolation

```python
async def _run_handler(self, handler, event) -> None:
    try:
        await handler(event)
    except Exception as exc:
        logger.exception(...)
        if SystemError in self._handlers:
            self.emit(SystemError(
                source_module=handler.__name__,
                error_message=str(exc),
                recoverable=True,
            ))
```

This is the key safety wrapper. If one handler crashes:
1. The exception is **logged** (not silently swallowed).
2. A `SystemError` event is **automatically emitted** if anything is listening to it.
3. **Other handlers for the same event are NOT affected** — crash isolation is per-handler.

---

### `drain()`

```python
async def drain(self) -> None:
    if self._tasks:
        await asyncio.gather(*self._tasks, return_exceptions=True)
```

Waits for all in-flight tasks to complete. Call this during **graceful shutdown** before exiting the event loop.

---

## 3. The Event System (`core/events.py`)

All events inherit from `BaseEvent`:

```python
@dataclass(frozen=True)
class BaseEvent:
    event_id: str  = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: str = field(default="default")
```

- `frozen=True` — events are **immutable** after creation. No handler can mutate an event — they can only react to it and emit new events.
- `event_id` — unique UUID per event, used for tracing and task naming.
- `session_id` — ties a chain of events to one user conversation turn.

### Complete Event Catalog

| Event | Emitted By | Consumed By |
|---|---|---|
| `WakeWordDetected` | Wake-word module | STT module |
| `TextInputReceived` | UI / text input | NLU / Intent module |
| `TranscriptReady` | STT module | NLU / Intent module |
| `IntentIdentified` | NLU module | Reasoning, Memory |
| `MemoryUpdateNeeded` | NLU / Reasoning | Memory module |
| `MemoryRetrieved` | Memory module | Reasoning module |
| `ReasoningRequested` | NLU module | Reasoning / LLM module |
| `ResponseReady` | Reasoning / LLM module | Output / TTS module |
| `TaskExecutionRequested` | Reasoning module | Actions module |
| `TaskCompleted` | Actions module | Reasoning / Response module |
| `SystemError` | Any module | Error handler |
| `ShutdownRequested` | Any module | All modules |

---

## 4. Core Bus Methods — Reference

| Method | Sync/Async | Blocks caller? | Use when |
|---|---|---|---|
| `subscribe(type, handler)` | sync | No | Startup wiring |
| `unsubscribe(type, handler)` | sync | No | Teardown / hot-unplug |
| `emit(event)` | sync | No | Fire-and-forget production flow |
| `emit_and_wait(event)` | async | Yes | Tests, sequential pipelines |
| `drain()` | async | Yes | Graceful shutdown |
| `handler_count(type)` | sync | No | Debugging / assertions |

---

## 5. The Full AI Pipeline Flow

Here is the complete data flow for a voice query like *"What is the weather in Kathmandu?"*

```
[Microphone]
     │
     ▼
WakeWordDetected  ──►  STT Module starts recording
     │
     ▼
TranscriptReady("What is the weather in Kathmandu?")
     │
     ├──►  NLU Module  ──►  IntentIdentified(intent=QUERY, entities={location: "Kathmandu"})
     │                              │
     │                              ├──►  MemoryUpdateNeeded  ──►  Memory Module stores turn
     │                              │
     │                              └──►  ReasoningRequested
     │                                          │
     │                                          ▼
     │                               LLM Module (hedged_generate)
     │                                          │
     │                                          ▼
     │                                   ResponseReady(text="...", requires_task=True)
     │                                          │
     │                                          ├──►  TTS Module → speaks response
     │                                          │
     │                                          └──►  TaskExecutionRequested("get_weather", {...})
     │                                                         │
     │                                                         ▼
     │                                                  Actions Module
     │                                                         │
     │                                                         ▼
     │                                                  TaskCompleted(result="25°C, Sunny")
     │                                                         │
     │                                                         ▼
     │                                               (emits another ResponseReady)
     │
     └──►  MemoryModule  ──►  stores transcript in episodic memory
```

---

## 6. Complete Integration Example — Wiring It All Together

This is a **full working example** of how to create a `KanchaAssistant` class that wires every module through the bus.

```python
# kancha_app.py
from __future__ import annotations

import asyncio
import logging

from core.bus import EventBus
from core.events import (
    WakeWordDetected,
    TranscriptReady,
    TextInputReceived,
    IntentIdentified,
    Intent,
    MemoryUpdateNeeded,
    MemoryLayer,
    MemoryRetrieved,
    ReasoningRequested,
    ResponseReady,
    TaskExecutionRequested,
    TaskCompleted,
    SystemError,
    ShutdownRequested,
)
from reasoning.llm_client_mulapi import get_pool, hedged_generate, ALL_MODELS, DEFAULT_HEDGE, REQUEST_TIMEOUT
from nlu.classifier import classify_tool_request

logger = logging.getLogger("kancha.app")


class KanchaAssistant:
    """
    Top-level orchestrator. Creates one EventBus and wires
    all module handlers to it at startup.
    """

    def __init__(self) -> None:
        self.bus = EventBus()
        self.pool = get_pool()          # Gemini key pool
        self._running = True
        self._wire()                    # Register all handlers

    # ─────────────────────────────────────────────────────────
    # WIRING — subscribe all handlers at startup
    # ─────────────────────────────────────────────────────────

    def _wire(self) -> None:
        bus = self.bus

        # Voice pipeline
        bus.subscribe(WakeWordDetected,        self._on_wake_word)
        bus.subscribe(TranscriptReady,         self._on_transcript)

        # Text input (CLI / UI)
        bus.subscribe(TextInputReceived,       self._on_text_input)

        # NLU → reasoning
        bus.subscribe(IntentIdentified,        self._on_intent)

        # Memory
        bus.subscribe(MemoryUpdateNeeded,      self._on_memory_update)
        bus.subscribe(MemoryRetrieved,         self._on_memory_retrieved)

        # LLM / reasoning
        bus.subscribe(ReasoningRequested,      self._on_reasoning_requested)

        # Output
        bus.subscribe(ResponseReady,           self._on_response_ready)

        # Tasks / actions
        bus.subscribe(TaskExecutionRequested,  self._on_task_requested)
        bus.subscribe(TaskCompleted,           self._on_task_completed)

        # System
        bus.subscribe(SystemError,             self._on_system_error)
        bus.subscribe(ShutdownRequested,       self._on_shutdown)

        logger.info("Bus wired: %r", self.bus)

    # ─────────────────────────────────────────────────────────
    # HANDLER IMPLEMENTATIONS
    # ─────────────────────────────────────────────────────────

    async def _on_wake_word(self, event: WakeWordDetected) -> None:
        """STT module: wake word detected → start transcription."""
        logger.info("[WAKE] confidence=%.2f  audio=%s", event.confidence, event.audio_path)
        # In production: call listen_and_transcribe(), then emit TranscriptReady
        # For example:
        # text = await listen_and_transcribe(event.audio_path)
        # self.bus.emit(TranscriptReady(text=text, session_id=event.session_id))

    async def _on_transcript(self, event: TranscriptReady) -> None:
        """Treat a voice transcript exactly like text input — route to NLU."""
        logger.info("[STT] transcript=%r  wer=%.2f", event.text, event.word_error_rate)
        self.bus.emit(TextInputReceived(text=event.text, session_id=event.session_id))

    async def _on_text_input(self, event: TextInputReceived) -> None:
        """NLU module: classify intent and emit IntentIdentified."""
        text = event.text.strip()
        if not text:
            return

        logger.info("[NLU] classifying: %r", text)

        # Check for a tool/task request first (fast regex path)
        tool = classify_tool_request(text)
        if tool:
            intent = Intent.TASK
            entities = tool.parameters
            entities["_task_name"] = tool.task_name
        else:
            # Default: conversational / query
            intent = Intent.QUERY if "?" in text else Intent.CONVERSATIONAL
            entities = {}

        self.bus.emit(IntentIdentified(
            raw_input=text,
            intent=intent,
            confidence=0.95,
            entities=entities,
            session_id=event.session_id,
        ))

        # Also trigger memory update for every user turn
        self.bus.emit(MemoryUpdateNeeded(
            content=text,
            layer=MemoryLayer.EPISODIC,
            metadata={"role": "user"},
            session_id=event.session_id,
        ))

    async def _on_intent(self, event: IntentIdentified) -> None:
        """Route based on intent: TASK → actions, QUERY/CONVERSATIONAL → LLM."""
        if event.intent == Intent.TASK:
            task_name = event.entities.get("_task_name", "unknown")
            params = {k: v for k, v in event.entities.items() if not k.startswith("_")}
            self.bus.emit(TaskExecutionRequested(
                task_name=task_name,
                parameters=params,
                session_id=event.session_id,
            ))
        else:
            # Go through LLM reasoning
            self.bus.emit(ReasoningRequested(
                intent_event=event,
                session_id=event.session_id,
            ))

    async def _on_memory_update(self, event: MemoryUpdateNeeded) -> None:
        """Memory module: persist content to the appropriate layer."""
        logger.info("[MEM] storing to %s: %r", event.layer, event.content[:60])
        # In production: write to ChromaDB / SQLite / short-term buffer

    async def _on_memory_retrieved(self, event: MemoryRetrieved) -> None:
        """Memory module: memory was retrieved (usually for RAG context)."""
        logger.info("[MEM] retrieved %d results for query: %r",
                    len(event.results), event.query)

    async def _on_reasoning_requested(self, event: ReasoningRequested) -> None:
        """LLM module: build prompt and call Gemini via hedged_generate."""
        intent = event.intent_event
        user_text = intent.raw_input

        # Build prompt (in production, use prompt_builder.py + memory context)
        prompt = (
            f"You are KANCHA, a smart personal AI assistant.\n"
            f"User: {user_text}\n"
            f"Assistant:"
        )

        logger.info("[LLM] calling hedged_generate for: %r", user_text[:60])

        try:
            response_text = await hedged_generate(
                pool=self.pool,
                models=ALL_MODELS,
                prompt=prompt,
                hedge_width=DEFAULT_HEDGE,
                timeout=REQUEST_TIMEOUT,
            )
        except Exception as exc:
            logger.error("[LLM] generation failed: %s", exc)
            self.bus.emit(SystemError(
                source_module="reasoning",
                error_message=str(exc),
                recoverable=True,
                session_id=event.session_id,
            ))
            return

        self.bus.emit(ResponseReady(
            text=response_text,
            llm_raw=response_text,
            requires_task=False,
            session_id=event.session_id,
        ))

    async def _on_response_ready(self, event: ResponseReady) -> None:
        """Output module: deliver the response to the user."""
        logger.info("[OUT] → %r", event.text[:80])
        print(f"\n🤖 KANCHA: {event.text}\n")
        # In production: also call speak(event.text) for TTS

    async def _on_task_requested(self, event: TaskExecutionRequested) -> None:
        """Actions module: execute a system task."""
        logger.info("[TASK] executing: %s  params=%s", event.task_name, event.parameters)

        # Dispatch to the appropriate action
        result = ""
        success = True
        try:
            if event.task_name == "open_app":
                from actions.apps import open_app
                open_app(event.parameters.get("app_name", ""))
                result = f"Opened {event.parameters.get('app_name')}"

            elif event.task_name == "get_weather":
                # from actions.weather import get_weather
                # result = await get_weather(event.parameters.get("query", ""))
                result = "Weather action: implement get_weather()"

            elif event.task_name == "sleep":
                from actions.power import sleep_now
                sleep_now()
                result = "Putting computer to sleep."

            else:
                result = f"Unknown task: {event.task_name}"
                success = False

        except Exception as exc:
            result = str(exc)
            success = False

        self.bus.emit(TaskCompleted(
            task_name=event.task_name,
            success=success,
            result=result,
            session_id=event.session_id,
        ))

    async def _on_task_completed(self, event: TaskCompleted) -> None:
        """After a task runs, speak/show the result."""
        if event.success:
            self.bus.emit(ResponseReady(
                text=event.result,
                session_id=event.session_id,
            ))
        else:
            self.bus.emit(ResponseReady(
                text=f"Task failed: {event.error or event.result}",
                session_id=event.session_id,
            ))

    async def _on_system_error(self, event: SystemError) -> None:
        """Central error handler — log and optionally recover."""
        logger.error(
            "[ERROR] from=%s  msg=%s  recoverable=%s",
            event.source_module, event.error_message, event.recoverable
        )
        if not event.recoverable:
            self.bus.emit(ShutdownRequested(
                reason=f"unrecoverable error in {event.source_module}",
                session_id=event.session_id,
            ))

    async def _on_shutdown(self, event: ShutdownRequested) -> None:
        """Graceful shutdown handler."""
        logger.info("[SHUTDOWN] reason=%s", event.reason)
        self._running = False

    # ─────────────────────────────────────────────────────────
    # ENTRY POINTS
    # ─────────────────────────────────────────────────────────

    async def run_text_mode(self) -> None:
        """Run KANCHA in interactive text (CLI) mode."""
        print("KANCHA ready. Type your message (Ctrl+C to quit).\n")
        while self._running:
            try:
                text = await asyncio.get_event_loop().run_in_executor(
                    None, input, "You: "
                )
            except (EOFError, KeyboardInterrupt):
                self.bus.emit(ShutdownRequested(reason="user exit"))
                break

            if text.strip():
                self.bus.emit(TextInputReceived(text=text.strip()))
                # Give handlers time to finish before showing next prompt
                await asyncio.sleep(0.1)

        await self.bus.drain()
        print("Goodbye.")

    async def handle_once(self, text: str) -> None:
        """Process a single text input and wait for all handlers to finish."""
        await self.bus.emit_and_wait(TextInputReceived(text=text))


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    assistant = KanchaAssistant()
    asyncio.run(assistant.run_text_mode())
```

---

## 7. Module-by-Module Integration Guide

### STT (Speech-to-Text)

The STT module listens for `WakeWordDetected` and emits `TranscriptReady`.

```python
# input/stt_handler.py
from core.bus import EventBus
from core.events import WakeWordDetected, TranscriptReady
from input.stt import listen_and_transcribe   # your existing STT function

def register_stt(bus: EventBus) -> None:
    async def on_wake(event: WakeWordDetected) -> None:
        text = await listen_and_transcribe(event.audio_path)
        bus.emit(TranscriptReady(
            text=text,
            session_id=event.session_id,
        ))

    bus.subscribe(WakeWordDetected, on_wake)
```

### NLU / Intent Classifier

```python
# nlu/nlu_handler.py
from core.bus import EventBus
from core.events import TextInputReceived, IntentIdentified, Intent
from nlu.classifier import classify_tool_request

def register_nlu(bus: EventBus) -> None:
    async def on_text(event: TextInputReceived) -> None:
        tool = classify_tool_request(event.text)
        if tool:
            entities = {**tool.parameters, "_task_name": tool.task_name}
            intent = Intent.TASK
        else:
            entities = {}
            intent = Intent.QUERY if "?" in event.text else Intent.CONVERSATIONAL

        bus.emit(IntentIdentified(
            raw_input=event.text,
            intent=intent,
            entities=entities,
            session_id=event.session_id,
        ))

    bus.subscribe(TextInputReceived, on_text)
```

### Reasoning / LLM

```python
# reasoning/reasoning_handler.py
from core.bus import EventBus
from core.events import ReasoningRequested, ResponseReady, SystemError
from reasoning.llm_client_mulapi import get_pool, hedged_generate, ALL_MODELS, DEFAULT_HEDGE, REQUEST_TIMEOUT

def register_reasoning(bus: EventBus) -> None:
    pool = get_pool()

    async def on_reasoning(event: ReasoningRequested) -> None:
        prompt = f"User: {event.intent_event.raw_input}\nAssistant:"
        try:
            text = await hedged_generate(pool, ALL_MODELS, prompt, DEFAULT_HEDGE, REQUEST_TIMEOUT)
            bus.emit(ResponseReady(text=text, session_id=event.session_id))
        except Exception as exc:
            bus.emit(SystemError(
                source_module="reasoning",
                error_message=str(exc),
                session_id=event.session_id,
            ))

    bus.subscribe(ReasoningRequested, on_reasoning)
```

### Memory

```python
# memory/memory_handler.py
from core.bus import EventBus
from core.events import MemoryUpdateNeeded, MemoryRetrieved, MemoryLayer

def register_memory(bus: EventBus) -> None:
    short_term: list[dict] = []   # replace with real ChromaDB / SQLite

    async def on_update(event: MemoryUpdateNeeded) -> None:
        short_term.append({
            "content": event.content,
            "layer": event.layer,
            "metadata": event.metadata,
            "session_id": event.session_id,
        })

    bus.subscribe(MemoryUpdateNeeded, on_update)
```

### Output / TTS

```python
# output/output_handler.py
from core.bus import EventBus
from core.events import ResponseReady
from output.tts import speak   # your TTS function

def register_output(bus: EventBus) -> None:
    async def on_response(event: ResponseReady) -> None:
        print(f"🤖 KANCHA: {event.text}")
        await speak(event.text)   # if async; otherwise use asyncio.to_thread(speak, event.text)

    bus.subscribe(ResponseReady, on_response)
```

### Modular `main.py`

```python
# main.py
import asyncio
import logging

from core.bus import EventBus
from nlu.nlu_handler import register_nlu
from reasoning.reasoning_handler import register_reasoning
from memory.memory_handler import register_memory
from output.output_handler import register_output
from core.events import TextInputReceived, SystemError

async def main():
    bus = EventBus()

    # Register all modules
    register_nlu(bus)
    register_reasoning(bus)
    register_memory(bus)
    register_output(bus)

    # Error logging
    async def on_error(e: SystemError):
        logging.error("[SYSTEM] %s: %s", e.source_module, e.error_message)
    bus.subscribe(SystemError, on_error)

    # Send one message
    await bus.emit_and_wait(TextInputReceived(text="Hello, who are you?"))
    await bus.drain()

asyncio.run(main())
```

---

## 8. Error Handling Through the Bus

The bus has two layers of error protection:

### Layer 1 — Handler crash isolation (`_run_handler`)

If `_on_reasoning_requested` raises an uncaught exception:
- The exception is **logged** with full traceback.
- A `SystemError` event is **automatically emitted**.
- All **other handlers** for the same event continue normally.

You never need a try/except around `bus.emit()` calls.

### Layer 2 — `SystemError` handler

Register a global `SystemError` handler to decide what to do:

```python
async def handle_system_error(event: SystemError) -> None:
    print(f"⚠️  Error in {event.source_module}: {event.error_message}")
    
    if not event.recoverable:
        # e.g. API key completely exhausted — shutdown gracefully
        bus.emit(ShutdownRequested(reason="unrecoverable: " + event.error_message))
    else:
        # e.g. one LLM call failed — tell user and continue
        bus.emit(ResponseReady(
            text="Sorry, I had a problem. Please try again.",
            session_id=event.session_id,
        ))

bus.subscribe(SystemError, handle_system_error)
```

### Shutdown flow

```python
async def handle_shutdown(event: ShutdownRequested) -> None:
    print(f"Shutting down: {event.reason}")
    # signal the main loop to stop
    shutdown_event.set()

bus.subscribe(ShutdownRequested, handle_shutdown)

# In main():
await bus.drain()   # wait for all in-flight tasks
```

---

## 9. Testing Your Integrations

Testing with the bus is straightforward because of `emit_and_wait`.

```python
import asyncio
import pytest
from core.bus import EventBus
from core.events import TextInputReceived, ResponseReady, IntentIdentified, Intent

@pytest.mark.asyncio
async def test_text_input_triggers_intent():
    bus = EventBus()
    received_intents = []

    async def capture_intent(event: IntentIdentified):
        received_intents.append(event)

    bus.subscribe(IntentIdentified, capture_intent)

    # Wire NLU
    from nlu.nlu_handler import register_nlu
    register_nlu(bus)

    # Trigger
    await bus.emit_and_wait(TextInputReceived(text="open Firefox"))

    assert len(received_intents) == 1
    assert received_intents[0].intent == Intent.TASK
    assert received_intents[0].entities.get("app_name") == "Firefox"


@pytest.mark.asyncio
async def test_crash_in_one_handler_doesnt_kill_others():
    bus = EventBus()
    log = []

    async def bad_handler(event: TextInputReceived):
        raise RuntimeError("I crashed!")

    async def good_handler(event: TextInputReceived):
        log.append("good ran")

    bus.subscribe(TextInputReceived, bad_handler)
    bus.subscribe(TextInputReceived, good_handler)

    await bus.emit_and_wait(TextInputReceived(text="hello"))

    assert "good ran" in log   # crash isolation confirmed


@pytest.mark.asyncio
async def test_full_pipeline_conversational():
    bus = EventBus()
    responses = []

    async def capture(event: ResponseReady):
        responses.append(event.text)

    bus.subscribe(ResponseReady, capture)

    # Wire everything
    from nlu.nlu_handler import register_nlu
    from reasoning.reasoning_handler import register_reasoning
    register_nlu(bus)
    register_reasoning(bus)

    await bus.emit_and_wait(TextInputReceived(text="Hello!"))
    await asyncio.sleep(0.5)   # let LLM respond (async)
    await bus.drain()

    assert len(responses) >= 1
    assert len(responses[0]) > 0
```

---

## 10. Advanced Patterns

### Multi-session support

Every event carries a `session_id`. Use it to maintain separate conversation contexts:

```python
# User 1 and User 2 can run simultaneously
bus.emit(TextInputReceived(text="Hello", session_id="user-1"))
bus.emit(TextInputReceived(text="Open Chrome", session_id="user-2"))
# Each event flows through the same handlers but carries its own session context
```

### Dynamic handler registration (hot-plug modules)

You can subscribe and unsubscribe handlers at runtime:

```python
# Enable voice during a session
async def voice_handler(event: TranscriptReady):
    ...

bus.subscribe(TranscriptReady, voice_handler)

# Disable voice (e.g., user turned off mic)
bus.unsubscribe(TranscriptReady, voice_handler)
```

### Middleware / event logging

Subscribe a catch-all debugger to any event type:

```python
from core.events import BaseEvent

async def audit_log(event: BaseEvent):
    with open("audit.log", "a") as f:
        f.write(f"{event.timestamp}  {type(event).__name__}  {event.session_id}\n")

# Attach to specific high-value events
for event_type in [TextInputReceived, IntentIdentified, ResponseReady, TaskCompleted]:
    bus.subscribe(event_type, audit_log)
```

### Checking bus health

```python
print(repr(bus))
# → EventBus(types=8, handlers=11)

print(bus.handler_count(TextInputReceived))
# → 2   (nlu handler + audit logger)
```

---

## Summary

| Concept | What to remember |
|---|---|
| `EventBus` | One instance, created at startup, shared everywhere |
| `subscribe` at startup | Wire all modules before any events flow |
| `emit` everywhere | Every module output = an emitted event, never a direct call |
| `emit_and_wait` | Use in tests or sequential steps |
| `drain` at shutdown | Always call before exiting to avoid dropped events |
| `frozen=True` events | Events are immutable; handlers only react and emit new events |
| Crash isolation | One bad handler never kills others; `SystemError` is auto-emitted |
| `session_id` | Flows through every event; enables multi-user/multi-turn tracing |

The bus is intentionally simple — it has no routing rules, no priorities, no filtering. Every complexity lives in the handlers. This keeps the bus itself fast, testable, and easy to reason about.
