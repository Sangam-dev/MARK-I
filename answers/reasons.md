# Explanation: `core/bus.py` — KANCHA's Async Pub/Sub EventBus

> Sourced from the existing knowledge graph (`graphify-out/graph.json`) and a full read of `core/bus.py`.
> Communities referenced are the graph's clustering output; cross-references via `[[kancha-bus-design]]`.

---

## The Question

> **Explain me the `bus.py`, why and how are we using it, how is it better than the traditional method?**

---

## What `bus.py` Is — In One Sentence

`core/bus.py` defines `EventBus`, an **async, fire-and-forget, in-process pub/sub bus** that is the **central nervous system of KANCHA** — every other module (STT, NLU, TTS, memory, genAI client) talks to every other module **only through events on this bus**, never via direct Python imports.

> *"Modules never import each other directly — bus only"* — `rationale_no_direct_module_imports` (community 0).

---

## Why We Are Using It

### 1. Architectural decoupling (the core reason)

Per the project design rules (`core/bus.py:1-25`, `plan.md GLOBAL CONTEXT, ARCHITECTURE`):

> *"all inter-module communication flows through this bus — modules never import each other directly"*

The pipeline STT → NLU → Memory → GenAI → TTS crosses **five different concerns** owned by **five different modules**. With direct imports, each module would need to know about every other module's API, leading to:

- **Tight coupling** — adding a new module (e.g. a logger, an analytics tap) requires editing every producer.
- **Circular import risk** — STT needs to tell NLU, but NLU might want to tell STT to "re-record". A direct call graph has no clean fix; an event bus does (both publish, both subscribe).
- **Hard-to-test monoliths** — every integration test has to wire up the full call graph.

The bus turns the dependency graph from a *mesh* into a *star*: every module knows only `core.events.BaseEvent` subclasses and `EventBus` itself.

### 2. Async fire-and-forget dispatch

`emit()` (line 153) is **synchronous, returns immediately**, and spawns one `asyncio.Task` per subscriber. The producer (`emit()`) never waits for handlers. This is critical because KANCHA's hot path is real-time audio:

```
WakeWordDetected → TextInputReceived → IntentIdentified → ResponseReady → TTS playback
```

If the producer blocked on the slowest subscriber, a slow STT retry would freeze wake-word detection. With the bus, **the slowest handler bounds latency, not the chain**.

### 3. Crash isolation between modules (rule #7 of the project design)

`_run_handler()` (line 273) catches **every** `Exception`, logs it, and emits a `SystemError` event so the rest of the system can react. It only re-raises `CancelledError` (cooperative shutdown) and `BaseException` (signals). One bad handler cannot crash a sibling — and the test suite explicitly enforces this (`test_handler_crash_isolation`).

### 4. Cross-thread safety for C-callback producers

`openWakeWord`'s wake-word detector and `sounddevice`'s audio callback both run on **non-asyncio C threads**. Calling `emit()` from there would race the loop's internals. `emit_threadsafe()` (line 233) solves this with `loop.call_soon_threadsafe(self.emit, event)` — a proven asyncio pattern.

The graph even shows the wiring:

```
top-level while-True driver [input/stt.py:82]
        │
        ▼
record_until_enter() [input/stt.py:19]
        │  ── publishes WakeWordDetected / TranscriptReady ──▶  bus.py
```

### 5. Observability & introspection for free

- `history()` (line 386) — bounded ring of the last `history_size` (default 1000) events for `/debug` endpoints and post-mortems.
- `stats()` (line 399) — handler count, in-flight task count, dedup cache size, closed flag. Perfect for a `/healthz` endpoint.
- `__repr__` (line 411) — single-line snapshot useful in logs.
- `_seen_ids` dedup cache — suppresses accidental re-publishes (e.g. STT retry publishing the same `TranscriptReady` twice).

---

## How We Are Using It — The Mechanics

### Public API surface (the only methods you need to know)

| Method | Purpose | Async? |
|---|---|---|
| `subscribe(event_type, handler)` | Register a coroutine for one event type | sync |
| `subscribe_all(handler)` | Register a coroutine for **every** event | sync |
| `unsubscribe(event_type, handler)` | Detach a handler (silent if absent) | sync |
| `emit(event)` | Fire-and-forget — returns instantly | sync |
| `emit_and_wait(event)` | Like `emit`, but awaits all handlers | **async** |
| `emit_threadsafe(event)` | Emit from a non-asyncio thread | sync |
| `drain()` | Wait for in-flight handlers | **async** |
| `close(timeout=5.0)` | Reject new subs/emits, drain, cancel stragglers | **async** |
| `history(event_type, limit)` | Inspect recent events | sync |
| `stats()` | Diagnostics snapshot | sync |

### Typical lifecycle (from the module docstring, line 50-56)

```python
bus = EventBus()
nlu = NLUClassifier(bus); nlu.register()        # subscribes
await text_input.run()                          # emits TextInputReceived
...
await bus.close()                               # graceful shutdown
```

### Two kinds of subscribers

- **Typed handlers** — `_handlers[type[BaseEvent]]` (line 68). Get events of *exactly* that class.
- **Global handlers** — `_global_handlers` (line 71). Get *every* event. Used for debug logging and the `ResponseFormatter` fan-in.

### Internal data structures

| Field | Purpose | Bounded? |
|---|---|---|
| `_handlers: dict[type, list[Handler]]` | Typed subscribers | grows with code |
| `_global_handlers: list[Handler]` | Wildcard subscribers | grows with code |
| `_tasks: set[asyncio.Task]` | In-flight handler tasks | discarded via `done_callback` |
| `_history: deque` | Recent events for introspection | `maxlen=history_size` |
| `_seen_ids: set` + `_seen_order: deque` | Dedup by `event_id` | `_DEDUP_MAX = 5000`, FIFO evict |
| `_closed: bool` | Lifecycle flag | n/a |
| `_loop: asyncio.AbstractEventLoop` | Captured at `__init__` for `emit_threadsafe` | n/a |

### Cross-module wiring (from the graph)

```
bus.py ◀── emits ── core/events.py  (BaseEvent, WakeWordDetected, TextInputReceived, …)
   ▲
   │  subscribes
   ├── input/stt.py   (groq whisper, wake word, while-True driver)
   ├── nlu/schemas.py + classifier.py
   ├── memory/  (short-term / structured / episodic)
   ├── genai/   (Round-Robin client)
   └── output/tts.py
```

The graph shows `tests/test_phase1.py` (community 1) explicitly verifies the contract: `test_basic_pubsub`, `test_handler_crash_isolation`, `test_multiple_handlers_same_event`, `test_emit_with_no_subscribers_does_not_raise`, `test_emit_threadsafe_from_worker_thread`, `test_subscribe_rejects_sync_handler`, `test_drain_waits_for_pending_tasks`, `test_events_have_unique_ids`, `test_events_are_frozen`.

---

## How Is This Better Than the "Traditional" Method?

### The "traditional method" = direct function calls between modules

```python
# Traditional (what we explicitly DON'T do)
class STTModule:
    def __init__(self, nlu, memory):
        self.nlu = nlu
        self.memory = memory

    def on_transcript(self, text):
        intent = self.nlu.classify(text)              # direct call
        ctx = self.memory.retrieve(text)               # direct call
        ...
```

This looks innocent. The problems show up the moment the system grows:

| Concern | Traditional direct calls | KANCHA's EventBus |
|---|---|---|
| **Adding a new consumer** (e.g. a logger) | Edit every producer to call `logger.log(...)` | Logger calls `bus.subscribe_all(...)` — zero edits to producers |
| **Cross-thread producers** (C callbacks) | Risky — must use `asyncio.run_coroutine_threadsafe` per call site | One method: `bus.emit_threadsafe(event)` |
| **One handler crashes** | Whole call chain dies (or you wrap every call in try/except) | `_run_handler` swallows it, emits `SystemError`, siblings continue |
| **Testing in isolation** | Must mock every dependency the module touches | Mock the bus (one `AsyncMock`); module sees only events |
| **Observability** | Add print/log statements in every module | `bus.history()` / `bus.stats()` give the whole picture |
| **Async semantics** | Each producer must `await` the next stage's return value | `emit()` is sync; `emit_and_wait()` is opt-in for tests/bootstrap |
| **Circular dependencies** | Real problem — Python imports break | N/A — modules only import `BaseEvent` + `EventBus` |
| **Late binding** | Compile-time imports | Runtime registration — easy to enable/disable features |
| **Hot-swappable handlers** | Re-import the module | `unsubscribe` + `subscribe` |
| **Cross-process / future distributed** | Rewrite the wiring layer | Same event classes, different transport (Redis pub/sub, NATS, etc.) |

### The killer feature: **observability**

The `plan.md` Phase 1 spec demands a system where you can answer *"what happened in the last 30 seconds?"* for debugging a voice assistant. With direct calls, this is a multi-module grep. With the bus, it's `bus.history(limit=300)` — every event any module emitted, in order, with timestamps.

The graph confirms this is the load-bearing design choice:

- `rationale_event_bus_central_nervous_system` — community 0, anchors the whole architecture
- `concept_async_pubsub` — community 0, tagged with `plan.md: GLOBAL CONTEXT, ARCHITECTURE`
- `rationale_no_direct_module_imports` — community 0, an explicit *prohibition* on direct imports

---

## The Trade-Offs (being honest)

The bus is not free. Compared to direct calls:

1. **Indirection cost** — to find "who handles `ResponseReady`?", you grep for `subscribe(ResponseReady, …)` instead of `await response_handler.handle(...)`. The graph (`graphify-out/graph.json`) is the antidote: queryable.
2. **No return values from `emit()`** — handlers must publish follow-up events (e.g. NLU publishes `IntentIdentified` for the genAI client to consume). This is a discipline, not a bug, but it means `emit_and_wait` exists for the rare cases that need synchronous results.
3. **Order of handler invocation is not guaranteed** — subscribers within a type receive events in registration order, but there's no global ordering across event types. Modules that need strict sequencing must use `await bus.emit_and_wait(...)` in their bootstrap or chain via explicit `*Completed` events.
4. **All I/O must be async** — `subscribe()` raises `TypeError` for sync handlers (`test_subscribe_rejects_sync_handler` enforces this). Sync I/O would block the loop and freeze every other module.
5. **In-memory only** — events are not persisted by default. The `_history` deque is bounded (default 1000) and lives only as long as the process. For replay, you would add a subscriber that writes to a sink.

---

## Memory Cross-References

- `[[kancha-bus-design]]` — invariants: capture loop at `__init__`, never re-raise in `_run_handler`, dedup by `event_id`, construct on the loop thread.

---

## Graph Sources (audit trail)

- `core_bus` (community 0) — the file node
- `core_bus_eventbus` (community 1) — the class
- `core_bus_rationale_1` — module docstring summary
- `core_bus_rationale_48` — class docstring / typical lifecycle
- `core_bus_rationale_121` — `subscribe_all` semantics
- `core_bus_rationale_358` — `close()` semantics
- `core_bus_rationale_400` — `stats()` semantics
- `rationale_event_bus_central_nervous_system` — design rationale
- `rationale_no_direct_module_imports` — design prohibition
- `concept_async_pubsub` — architectural pattern
- `tests/test_phase1.py` — community 1, contract verification
- `plan.md` — community 0, source of the architecture rule
