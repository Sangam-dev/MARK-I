```
PROJECT: KANCHA — Multimodal Personalized AI Assistant
LANGUAGE: Python 3.12
OS: Linux (Ubuntu)
ARCHITECTURE: Event-driven async system using asyncio pub/sub
STYLE: Production-quality, typed Python, no shortcuts, no toy implementations

CORE DESIGN RULES (never violate these):
1. Every inter-module communication goes through the EventBus — never direct imports between modules
2. All I/O operations must be async — never use time.sleep(), requests.get(), or any blocking call
3. All file paths use pathlib.Path — never hardcoded strings
4. Every public method has a type hint on every parameter and return value
5. Every module has a module-level docstring explaining its purpose and design decisions
6. Frozen dataclasses for all events — events are immutable facts
7. Error isolation — a crash in one module must never crash another
8. No global state — everything is injected via __init__

FOLDER STRUCTURE:
kancha/
├── kancha/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── events.py        # typed event dataclasses
│   │   ├── bus.py           # async pub/sub event bus
│   │   ├── context.py       # short-term RAM conversation buffer
│   │   ├── config.py        # settings and env loader
│   │   ├── logging.py       # structured logger setup
│   │   ├── platform.py      # cross-platform path/autostart abstraction
│   │   └── exceptions.py    # custom exception types
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── base.py          # abstract memory interface
│   │   ├── structured.py    # SQLite layer via aiosqlite
│   │   ├── vector.py        # Chroma vector store + nomic-embed-text
│   │   └── manager.py       # orchestrates all 3 memory layers
│   ├── nlu/
│   │   ├── __init__.py
│   │   ├── schemas.py       # Pydantic output models for NLU
│   │   └── classifier.py    # LLM-based intent + entity extraction
│   ├── reasoning/
│   │   ├── __init__.py
│   │   ├── llm_client.py    # Gemini 1.5 Flash async adapter
│   │   ├── prompt_builder.py # assembles full prompt from all sources
│   │   └── rag.py           # retrieval pipeline
│   ├── input/
│   │   ├── __init__.py
│   │   ├── text_input.py    # CLI async input loop (Phase 1 interface)
│   │   ├── wake_word.py     # openWakeWord detector
│   │   └── stt.py           # faster-whisper transcription
│   ├── output/
│   │   ├── __init__.py
│   │   ├── response_formatter.py  # terminal output handler
│   │   ├── tts.py                 # Piper TTS synthesis
│   │   └── speaker.py             # audio playback
│   └── tasks/
│       ├── __init__.py
│       ├── registry.py      # allowed task types + parameter schemas
│       ├── executor.py      # validated task dispatcher
│       └── scheduler.py     # asyncio timer-based reminders
├── tests/
│   ├── test_phase1.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   ├── test_phase4.py
│   ├── test_phase5.py
│   └── test_phase6.py
├── scripts/
│   └── benchmark_latency.py
├── models/                  # wake word .onnx files go here
├── data/                    # chroma and sqlite data (gitignored)
├── main.py                  # entry point
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── Makefile
└── README.md

TECH STACK:
- LLM: Gemini 1.5 Flash via google-genai SDK (NOT google-generativeai — that is deprecated)
- Embeddings: nomic-embed-text via Ollama running locally on localhost:11434
- Vector DB: ChromaDB with PersistentClient
- Structured DB: SQLite via aiosqlite (async — never use sqlite3 directly in async code)
- STT: faster-whisper with tiny model
- TTS: Piper TTS
- Wake word: openWakeWord
- Audio: sounddevice
- Validation: Pydantic v2
- Config: python-dotenv + PyYAML
- Tray icon: pystray + Pillow
- Cross-platform paths: pathlib.Path everywhere

ENVIRONMENT VARIABLES (from .env):
- GEMINI_API_KEY: Gemini API key
- KANCHA_SESSION_ID: current session identifier (default: "default")
- KANCHA_LOG_LEVEL: logging level (default: "INFO")
- KANCHA_DATA_DIR: override for data directory (optional)
```

---

## PHASE 1 — ASYNC FOUNDATION

### FILE: kancha/core/events.py

```
TASK: Write kancha/core/events.py

PURPOSE:
This file defines every typed event dataclass used by the KANCHA system.
It is the shared communication language of the entire system.
Every inter-module message is one of these dataclasses — never raw dicts.

REQUIREMENTS:
1. Import: from __future__ import annotations, uuid, dataclasses, datetime, enum, typing
2. Define Intent enum (str, Enum) with values: QUERY, TASK, CONVERSATIONAL
3. Define MemoryLayer enum (str, Enum) with values: SHORT_TERM, STRUCTURED, EPISODIC
4. Define BaseEvent frozen dataclass with:
   - event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
   - timestamp: datetime = field(default_factory=datetime.utcnow)
   - session_id: str = field(default="default")
5. Define these event classes, all frozen=True, all inheriting BaseEvent:
   - WakeWordDetected(audio_path: str, confidence: float)
   - TextInputReceived(text: str)
   - TranscriptReady(text: str, word_error_rate: float, language: str)
   - IntentIdentified(raw_text: str, intent: Intent, entities: dict[str, Any], confidence: float)
   - MemoryUpdateNeeded(content: str, layer: MemoryLayer, metadata: dict[str, Any])
   - MemoryRetrieved(query: str, structured_results: list[dict], episodic_chunks: list[str])
   - ReasoningRequested(intent_event: IntentIdentified, memory_event: MemoryRetrieved)
   - ResponseReady(text: str, llm_raw: str, requires_task: bool)
   - TaskExecutionRequested(task_type: str, params: dict[str, Any])
   - TaskCompleted(task_type: str, success: bool, result: str, error: str)
   - SystemError(source_module: str, error_message: str, recoverable: bool)
   - ShutdownRequested(reason: str)

DESIGN RULES:
- All mutable defaults (dict, list) must use field(default_factory=...)
- All events must be frozen=True — immutability is non-negotiable
- Every class needs a docstring explaining: who emits it, who consumes it, what each field means
- from __future__ import annotations must be first line

ACCEPTANCE CRITERIA:
- e = TranscriptReady(text="hello") works with no errors
- e.text = "mutate" raises FrozenInstanceError
- Two instances of the same event type have different event_ids
- All dict/list fields default to empty, not None
```

---

### FILE: kancha/core/bus.py

```
TASK: Write kancha/core/bus.py

PURPOSE:
The async event bus is the central nervous system of KANCHA.
Modules communicate exclusively through this bus — never by calling each other directly.

REQUIREMENTS:
1. Class EventBus with:
   - __init__: _handlers dict[type, list[Handler]] using defaultdict(list), _tasks set[asyncio.Task]
   - subscribe(event_type, handler): validate handler is coroutinefunction, append to _handlers
   - unsubscribe(event_type, handler): remove handler if present
   - emit(event): get handlers, create_task for each, add to _tasks set with done callback
   - emit_and_wait(event): like emit but uses gather — waits for all handlers
   - _run_handler(handler, event): try/except wrapper, logs crash, emits SystemError on failure
   - drain(): await gather on all _tasks — for clean shutdown
   - handler_count(event_type) -> int
   - __repr__ showing total event types and handlers registered

DESIGN RULES:
- emit() must be fire-and-forget: create_task, NOT await handler directly
- _run_handler must NEVER re-raise — catch all exceptions, log them, emit SystemError
- Use task.add_done_callback(self._tasks.discard) to prevent memory leak
- Never use a global singleton — bus is instantiated in main.py and injected
- Handler type: Callable[[Any], Coroutine[Any, Any, None]]
- TypeError if non-async handler is passed to subscribe()

ACCEPTANCE CRITERIA:
- subscribe + emit + asyncio.sleep(0.05) → handler received event
- Two handlers on same event → both run
- Bad handler crashes → good handler on same event still runs
- emit() returns immediately without waiting for handlers
- drain() waits for all tasks to complete
```

---

### FILE: kancha/core/context.py

```
TASK: Write kancha/core/context.py

PURPOSE:
Short-term conversational memory — the RAM buffer.
Holds the last N conversation turns for injection into every LLM prompt.
Does NOT persist — cleared on restart.

REQUIREMENTS:
1. Turn dataclass with role: Literal["user","assistant","system"], content: str, to_dict() method
2. ConversationContext class with:
   - __init__(max_turns: int = 10): deque(maxlen=max_turns), asyncio.Lock
   - add_turn(role, content) async: acquire lock, skip empty content, append Turn
   - as_messages() async -> list[dict]: acquire lock, return [t.to_dict() for t in buffer]
   - clear() async: acquire lock, clear deque
   - token_estimate() async -> int: rough count via total_chars / 4.0
   - max_turns property
   - __repr__

DESIGN RULES:
- Every method that touches the buffer must acquire the asyncio.Lock
- Never store empty or whitespace-only turns
- as_messages() returns OpenAI-compatible format: [{"role": "user", "content": "..."}]
- deque maxlen enforces rolling eviction automatically — do not manage manually

ACCEPTANCE CRITERIA:
- add 2 turns, as_messages() returns 2 items in correct order
- add more turns than max_turns, oldest are evicted automatically
- concurrent add_turn and as_messages() calls do not corrupt state
- clear() empties the buffer
```

---

### FILE: tests/test_phase1.py

```
TASK: Write tests/test_phase1.py

PURPOSE:
Manual integration test harness for Phase 1 foundation.
Verifies event bus, context buffer, and event immutability.

REQUIREMENTS:
Write an async main() function run via asyncio.run(main()) that tests:
1. EventBus basic pub/sub: subscribe handler, emit event, verify received after asyncio.sleep(0.05)
2. EventBus multiple handlers: two handlers on same event type, both must run
3. EventBus no handlers: emit with no subscribers must not raise
4. EventBus crash isolation: bad handler crashes, good handler on same event still runs
5. ConversationContext add and retrieve: add 2 turns, verify order and content
6. ConversationContext rolling eviction: maxlen=4, add 6 items, verify oldest evicted
7. Events frozen: attempt mutation, verify FrozenInstanceError or AttributeError raised
8. Events unique IDs: two instances of same type have different event_ids

FORMAT:
- Print [PASS] in green or [FAIL] in red for each test
- At the end print total passed/total
- Exit with sys.exit(1) if any test fails
- Do NOT use pytest — plain asyncio.run() harness only

ACCEPTANCE CRITERIA:
- All 8 tests pass with [PASS]
- The stderr crash log from test 4 is expected and acceptable
- Script exits with code 0 on all pass
```

---

## PHASE 2 — HYBRID MEMORY SYSTEM

### FILE: kancha/core/config.py

```
TASK: Write kancha/core/config.py

PURPOSE:
Single source of truth for all configuration.
Loads .env file, provides typed access to all settings.
No other module hardcodes values — they import from here.

REQUIREMENTS:
1. Load .env using python-dotenv at module level
2. Define KanchaConfig dataclass with:
   - gemini_api_key: str (from GEMINI_API_KEY env var)
   - gemini_model: str = "gemini-1.5-flash"
   - embedding_model: str = "nomic-embed-text"
   - ollama_base_url: str = "http://localhost:11434"
   - chroma_collection: str = "kancha_memory"
   - sqlite_db_name: str = "kancha.db"
   - max_context_turns: int = 10
   - rag_top_k: int = 3
   - max_prompt_tokens: int = 4000
   - log_level: str = "INFO"
   - session_id: str = "default"
   - whisper_model: str = "tiny"
   - tts_voice: str = "en_US-lessac-medium"
3. Define get_data_dir() -> Path using platform detection
4. Define get_config() -> KanchaConfig singleton function
5. Validate that gemini_api_key is not empty on load — raise ValueError with helpful message

DESIGN RULES:
- Use pathlib.Path for all directory paths
- Data dir: ~/.local/share/kancha/ on Linux, %APPDATA%/kancha/ on Windows
- Config dir: ~/.config/kancha/ on Linux
- Singleton pattern: config loaded once, reused

ACCEPTANCE CRITERIA:
- get_config() returns same instance on multiple calls
- Missing GEMINI_API_KEY raises ValueError with clear message
- get_data_dir() returns a Path that exists after first call (mkdir)
```

---

### FILE: kancha/core/exceptions.py

```
TASK: Write kancha/core/exceptions.py

PURPOSE:
Custom exception hierarchy for KANCHA.
Using typed exceptions lets callers handle specific failure modes.

REQUIREMENTS:
Define these exception classes with docstrings:
- KanchaError(Exception): base for all KANCHA errors
- ConfigurationError(KanchaError): bad config, missing env vars
- MemoryError(KanchaError): database or vector store failures
- LLMError(KanchaError): Gemini API failures
- LLMRateLimitError(LLMError): specifically rate limit errors
- LLMTimeoutError(LLMError): timeout errors
- NLUError(KanchaError): intent classification failures
- STTError(KanchaError): transcription failures
- TTSError(KanchaError): synthesis failures
- TaskError(KanchaError): task execution failures
- TaskNotAllowedError(TaskError): task type not in registry

Each exception should accept an optional context: dict parameter for structured error info.
```

---

### FILE: kancha/memory/base.py

```
TASK: Write kancha/memory/base.py

PURPOSE:
Abstract base class defining the interface all memory backends must implement.
Enforces a contract — swap SQLite for Postgres without changing any other code.

REQUIREMENTS:
1. Abstract class AbstractMemoryBackend using ABC
2. Abstract async methods:
   - store(content: str, metadata: dict[str, Any]) -> str (returns record id)
   - retrieve(query: str, limit: int = 5) -> list[dict[str, Any]]
   - delete(record_id: str) -> bool
   - clear_session(session_id: str) -> int (returns count deleted)
   - health_check() -> bool
3. Concrete property: name -> str (backend identifier)

DESIGN RULES:
- All methods must be async
- Method signatures are the contract — implementations must match exactly
- Include type hints on every parameter and return value
```

---

### FILE: kancha/memory/structured.py

```
TASK: Write kancha/memory/structured.py

PURPOSE:
SQLite-backed structured memory using aiosqlite.
Stores: conversation history, user facts, scheduled tasks.
This is the precise, queryable layer — use for known facts and tasks.

REQUIREMENTS:
1. Class StructuredMemory implementing AbstractMemoryBackend
2. __init__(db_path: Path): store path, connection will be created in initialize()
3. async initialize(): create tables if not exist, enable WAL mode
4. Tables to create:
   interactions(
     id TEXT PRIMARY KEY,
     session_id TEXT NOT NULL,
     role TEXT NOT NULL,
     content TEXT NOT NULL,
     timestamp TEXT NOT NULL,
     metadata TEXT DEFAULT '{}'
   )
   facts(
     id TEXT PRIMARY KEY,
     key TEXT NOT NULL,
     value TEXT NOT NULL,
     session_id TEXT NOT NULL,
     created_at TEXT NOT NULL,
     updated_at TEXT NOT NULL
   )
   tasks(
     id TEXT PRIMARY KEY,
     description TEXT NOT NULL,
     due_at TEXT,
     status TEXT DEFAULT 'pending',
     session_id TEXT NOT NULL,
     created_at TEXT NOT NULL,
     metadata TEXT DEFAULT '{}'
   )
5. Implement AbstractMemoryBackend methods
6. Additional methods:
   - store_interaction(session_id, role, content, metadata) -> str
   - get_recent_interactions(session_id, limit=10) -> list[dict]
   - store_fact(key, value, session_id) -> str
   - get_fact(key, session_id) -> str | None
   - get_all_facts(session_id) -> list[dict]
   - store_task(description, due_at, session_id, metadata) -> str
   - get_pending_tasks(session_id) -> list[dict]
   - update_task_status(task_id, status) -> bool
   - close() async

DESIGN RULES:
- ALWAYS use aiosqlite — never sqlite3 (it blocks the event loop)
- Enable WAL mode: PRAGMA journal_mode=WAL
- Use parameterized queries — never string formatting in SQL
- Store timestamps as ISO format strings
- Store metadata and dicts as JSON strings
- Use uuid4 for all primary keys
- Connection should be a single persistent connection, not reopened per query

ACCEPTANCE CRITERIA:
- store_interaction then get_recent_interactions returns the stored item
- WAL mode enabled (verified by PRAGMA query)
- Concurrent async reads do not raise
- store_fact then get_fact returns correct value
- All queries use parameterized form (?, not f-strings)
```

---

### FILE: kancha/memory/vector.py

```
TASK: Write kancha/memory/vector.py

PURPOSE:
ChromaDB-backed vector memory using nomic-embed-text embeddings via Ollama.
Stores semantic memories — past conversations, insights, context.
Enables "what did we discuss about X?" style retrieval.

REQUIREMENTS:
1. Class VectorMemory implementing AbstractMemoryBackend
2. __init__(persist_dir: Path, collection_name: str, ollama_url: str, embedding_model: str)
3. async initialize(): create ChromaDB PersistentClient, get or create collection
4. async embed(text: str) -> list[float]:
   - POST to ollama_url/api/embeddings with model and prompt
   - Use httpx.AsyncClient — never requests (blocking)
   - Return embedding vector
   - Run in executor if needed for CPU-bound parts
5. async store(content, metadata) -> str:
   - Generate embedding
   - Add to Chroma collection with id, embedding, document, metadata
   - Return id
6. async retrieve(query, limit=3) -> list[dict]:
   - Embed the query
   - collection.query() with n_results=limit
   - Return list of {content, metadata, distance} dicts
7. async delete(record_id) -> bool
8. async clear_session(session_id) -> int: delete by metadata filter
9. async health_check() -> bool: attempt embed of "health check", return True/False
10. async close(): cleanup

DESIGN RULES:
- Use httpx.AsyncClient for ALL HTTP calls — never requests
- Embedding is CPU-bound — use asyncio.get_event_loop().run_in_executor() if it blocks
- ChromaDB PersistentClient saves to disk automatically — no manual flush needed
- Metadata must always include: session_id, timestamp, type fields
- handle httpx errors gracefully — raise KanchaError with context

ACCEPTANCE CRITERIA:
- store 3 items, query with similar text, correct item returned in top result
- health_check() returns True when Ollama is running
- health_check() returns False (not raises) when Ollama is down
- Stored data persists after process restart (PersistentClient)
```

---

### FILE: kancha/memory/manager.py

```
TASK: Write kancha/memory/manager.py

PURPOSE:
Orchestrates all three memory layers: short-term (context), structured (SQLite), episodic (Chroma).
Single interface for the rest of the system — no module touches memory backends directly.
Subscribes to MemoryUpdateNeeded events and emits MemoryRetrieved events.

REQUIREMENTS:
1. Class MemoryManager with:
   - __init__(bus, structured: StructuredMemory, vector: VectorMemory, context: ConversationContext, config)
   - async initialize(): call initialize() on both backends
   - async remember(content, session_id, role, metadata, layer): 
       if layer == SHORT_TERM: context.add_turn()
       if layer == STRUCTURED: structured.store_interaction()
       if layer == EPISODIC: vector.store()
       if layer not specified: write to BOTH structured and episodic
   - async recall(query, session_id) -> MemoryRetrieved:
       structured_results = await structured.get_recent_interactions(session_id)
       episodic_chunks = await vector.retrieve(query)
       return MemoryRetrieved(query=query, structured_results=..., episodic_chunks=...)
   - async on_memory_update_needed(event: MemoryUpdateNeeded): handler for bus events
   - async close(): close both backends
2. subscribe to MemoryUpdateNeeded events in initialize()

DESIGN RULES:
- recall() should run structured and vector queries CONCURRENTLY using asyncio.gather()
- Never let one backend failure crash the other — wrap each in try/except
- Log every memory operation with timestamp and session_id
- The manager is the ONLY thing that knows about both backends

ACCEPTANCE CRITERIA:
- remember() then recall() returns the stored content in results
- structured and vector queries run concurrently (measurable by timing)
- One backend being down does not crash recall() — returns partial results
- MemoryUpdateNeeded event triggers storage correctly
```

---

### FILE: tests/test_phase2.py

```
TASK: Write tests/test_phase2.py

PURPOSE:
Integration tests for the hybrid memory system.
Tests persistence, retrieval accuracy, and concurrent access.

REQUIREMENTS:
Test these scenarios in async main():
1. SQLite store_interaction then get_recent_interactions returns correct data
2. SQLite store_fact then get_fact returns correct value
3. SQLite WAL mode is enabled (PRAGMA journal_mode returns 'wal')
4. Chroma store then retrieve returns semantically similar result in top-3
5. Chroma health_check returns True (requires Ollama running)
6. MemoryManager recall runs both queries (verify both return data)
7. Persistence test: store to Chroma, create new VectorMemory instance pointing to same dir, retrieve — data must still be there
8. ConversationContext integration with MemoryManager

FORMAT: Same [PASS]/[FAIL] format as test_phase1.py

NOTE: Tests 4, 5, 7 require Ollama running with nomic-embed-text pulled.
Skip gracefully with [SKIP] if Ollama is unavailable.
```

---

## PHASE 3 — LLM + RAG REASONING

### FILE: kancha/reasoning/llm_client.py

```
TASK: Write kancha/reasoning/llm_client.py

PURPOSE:
Async adapter for the Gemini 1.5 Flash API.
All LLM calls in KANCHA go through this class.
Handles retries, rate limits, timeouts, and fallbacks.

REQUIREMENTS:
1. Class GeminiClient with:
   - __init__(api_key, model="gemini-1.5-flash", max_retries=3, timeout=30.0)
   - async initialize(): configure google-genai client
   - async generate(prompt: str, system: str = "") -> str:
       call Gemini with prompt and system instruction
       return text response
       on failure after retries: return fallback string, never raise to caller
   - async generate_json(prompt: str, schema_description: str, system: str = "") -> dict:
       call Gemini asking for JSON output only
       parse response with json.loads()
       on JSON parse failure: retry up to max_retries
       on total failure: return empty dict {}
   - async health_check() -> bool
   - _build_retry_delay(attempt: int) -> float: exponential backoff

DESIGN RULES:
- Use google-genai SDK (import google.genai) NOT google-generativeai (deprecated)
- All calls must be async — use the async client methods
- Exponential backoff: 1s, 2s, 4s between retries
- Log every API call with: model, prompt length, response length, latency
- Rate limit errors (429) get longer backoff: 10s, 20s, 40s
- Timeout errors get retried immediately once, then fail gracefully
- FALLBACK_RESPONSE = "I'm having trouble thinking right now. Could you try again?"
- Never expose the API key in logs

ACCEPTANCE CRITERIA:
- generate() returns a string on success
- generate() returns FALLBACK_RESPONSE on API failure, never raises
- generate_json() returns a dict on success
- generate_json() returns {} on parse failure, never raises
- Retry logic fires on 429 responses
- All calls logged with latency
```

---

### FILE: kancha/reasoning/prompt_builder.py

```
TASK: Write kancha/reasoning/prompt_builder.py

PURPOSE:
Assembles the complete prompt sent to the LLM on every turn.
Combines: system persona + structured facts + episodic memories + conversation history + current input.
Enforces a token budget to prevent context overflow.

REQUIREMENTS:
1. KANCHA_PERSONA constant: multiline string defining KANCHA's personality
   - Name: KANCHA
   - Helpful, concise, remembers the user personally
   - Mentions when it's drawing from memory
   - Admits when it doesn't know something
2. Class PromptBuilder with:
   - __init__(config): stores max_prompt_tokens
   - async build(
       user_input: str,
       intent_event: IntentIdentified,
       memory_event: MemoryRetrieved,
       context_messages: list[dict],
     ) -> tuple[str, list[dict]]:
       Returns (system_prompt, messages_list) for Gemini
   - _format_structured_facts(results: list[dict]) -> str
   - _format_episodic_chunks(chunks: list[str]) -> str
   - _estimate_tokens(text: str) -> int: len(text) // 4
   - _trim_to_budget(system: str, messages: list[dict]) -> tuple[str, list[dict]]:
       if over budget, drop oldest episodic chunks first, then oldest messages

PROMPT STRUCTURE (in this exact order):
  system = PERSONA + "\n\n" + STRUCTURED_FACTS + "\n\n" + EPISODIC_MEMORIES
  messages = context_messages + [{"role": "user", "content": user_input}]

DESIGN RULES:
- Total token estimate must not exceed max_prompt_tokens (default 4000)
- When trimming: drop episodic chunks before dropping conversation history
- Log what was included and what was dropped
- The system prompt must always include the persona — never drop it

ACCEPTANCE CRITERIA:
- build() returns valid (system_str, messages_list) tuple
- When context is large, output stays under token budget
- Episodic chunks are dropped before conversation history when trimming
- Persona is always present in system string
```

---

### FILE: kancha/reasoning/rag.py

```
TASK: Write kancha/reasoning/rag.py

PURPOSE:
Retrieval pipeline — given a user query, fetches relevant memories from all layers.
Runs structured and vector retrieval concurrently.
Emits MemoryRetrieved event for the reasoning layer to consume.

REQUIREMENTS:
1. Class RAGPipeline with:
   - __init__(memory_manager: MemoryManager, bus: EventBus, config)
   - async retrieve(query: str, session_id: str) -> MemoryRetrieved:
       Run concurrently using asyncio.gather():
         structured = memory_manager.structured.get_recent_interactions(session_id, limit=5)
         episodic = memory_manager.vector.retrieve(query, limit=config.rag_top_k)
       Combine results
       Return MemoryRetrieved event
   - async on_intent_identified(event: IntentIdentified):
       Call retrieve(event.raw_text, event.session_id)
       Emit ReasoningRequested(intent_event=event, memory_event=retrieved)

DESIGN RULES:
- Structured and vector retrieval MUST run concurrently — asyncio.gather() not sequential awaits
- Log retrieval time for both sources separately
- If one source fails, return empty list for that source — do not fail the whole retrieval
- Subscribe to IntentIdentified in __init__ or a separate register() method

ACCEPTANCE CRITERIA:
- retrieve() returns MemoryRetrieved with results from both sources
- Both queries run concurrently (verify with timing: total time ≈ max(t1, t2) not t1+t2)
- One source failing returns partial results, not an error
```

---

### FILE: tests/test_phase3.py

```
TASK: Write tests/test_phase3.py

REQUIREMENTS:
Test in async main():
1. GeminiClient.generate() returns a non-empty string (requires API key)
2. GeminiClient.generate_json() returns a dict (requires API key)
3. GeminiClient.generate() returns fallback string when API key is invalid
4. PromptBuilder.build() returns tuple of (str, list)
5. PromptBuilder respects token budget — output under max_prompt_tokens
6. RAGPipeline.retrieve() returns MemoryRetrieved
7. RAGPipeline runs both queries concurrently — total time < sum of individual times
8. End-to-end: store a fact in memory, ask about it, verify answer contains fact

Skip gracefully if GEMINI_API_KEY not set or Ollama unavailable.
```

---

## PHASE 4 — NLU + CLI INTERFACE

### FILE: kancha/nlu/schemas.py

```
TASK: Write kancha/nlu/schemas.py

PURPOSE:
Pydantic v2 models defining the expected JSON output from the NLU classifier.
These models validate LLM output before it enters the system.

REQUIREMENTS:
1. IntentResult(BaseModel):
   - intent: Intent (from core.events)
   - entities: dict[str, Any] = {}
   - confidence: float = Field(ge=0.0, le=1.0, default=1.0)
   - reasoning: str = "" (LLM's explanation of why it chose this intent)
2. EntityType enum: PERSON, DATETIME, DURATION, LOCATION, TASK_NAME, QUANTITY, OTHER
3. ExtractedEntity(BaseModel): value: str, entity_type: EntityType, raw_text: str
4. NLUResult(BaseModel):
   - intent: Intent
   - entities: list[ExtractedEntity] = []
   - confidence: float
   - requires_task_execution: bool = False
   - task_type: str | None = None (set if requires_task_execution is True)
   - task_params: dict[str, Any] = {}

DESIGN RULES:
- Use Pydantic v2 syntax (model_validate, not parse_obj)
- All fields have defaults so partial LLM output doesn't crash validation
- Add model_config = ConfigDict(str_strip_whitespace=True)
```

---

### FILE: kancha/nlu/classifier.py

```
TASK: Write kancha/nlu/classifier.py

PURPOSE:
LLM-based intent classification and entity extraction.
Uses Gemini to classify every user input as QUERY, TASK, or CONVERSATIONAL
and extract structured entities.
Replaces the spaCy approach — simpler, more accurate, no training data needed.

REQUIREMENTS:
1. NLU_SYSTEM_PROMPT constant: instructs LLM to:
   - Classify intent as exactly one of: query, task, conversational
   - Extract entities with types
   - Determine if task execution is needed and what type
   - Return ONLY valid JSON matching NLUResult schema
   - No preamble, no markdown, no explanation outside JSON

2. Class NLUClassifier with:
   - __init__(llm_client: GeminiClient)
   - async classify(text: str, session_id: str = "default") -> NLUResult:
       Build prompt with text
       Call llm_client.generate_json()
       Validate with NLUResult.model_validate()
       On validation failure: return default NLUResult(intent=CONVERSATIONAL)
       Return validated result
   - async on_text_input(event: TextInputReceived): classify, emit IntentIdentified
   - async on_transcript_ready(event: TranscriptReady): same flow

DESIGN RULES:
- This runs on EVERY user input — prompt must be short and fast
- Total latency target: under 500ms
- Always return a valid NLUResult — never raise to callers
- Log: input text, classified intent, confidence, latency

ACCEPTANCE CRITERIA:
- "remind me tomorrow at 3pm" → intent=TASK, entities contain datetime
- "what is the capital of Nepal?" → intent=QUERY
- "hello how are you" → intent=CONVERSATIONAL
- Invalid LLM output → default NLUResult, no crash
```

---

### FILE: kancha/input/text_input.py

```
TASK: Write kancha/input/text_input.py

PURPOSE:
Async CLI input loop — the text interface during development.
Replaced by voice in Phase 6 but stays as a fallback mode.
Emits TextInputReceived events on every line of input.

REQUIREMENTS:
1. Class TextInputHandler with:
   - __init__(bus: EventBus, config, session_id: str = "default")
   - async run(): main input loop
       Print a prompt indicator (">")
       Read input using asyncio executor (not blocking input())
       Skip empty input
       On "quit" or "exit": emit ShutdownRequested, break loop
       Emit TextInputReceived(text=line, session_id=session_id)
   - async _read_line() -> str: wraps input() in run_in_executor

DESIGN RULES:
- NEVER use input() directly in async code — it blocks the event loop
- Use loop.run_in_executor(None, input, "> ") for non-blocking input
- Handle KeyboardInterrupt (Ctrl+C) gracefully: emit ShutdownRequested
- Handle EOF (Ctrl+D) gracefully: emit ShutdownRequested

ACCEPTANCE CRITERIA:
- Input is read without blocking the event loop
- "exit" triggers ShutdownRequested event
- Ctrl+C triggers ShutdownRequested event, not a crash
- Empty input is silently skipped
```

---

### FILE: kancha/output/response_formatter.py

```
TASK: Write kancha/output/response_formatter.py

PURPOSE:
Terminal output handler — displays KANCHA responses in the CLI.
Subscribes to ResponseReady events.
Temporary interface — replaced by TTS in Phase 6.
Written so that swapping to TTS requires changing one line.

REQUIREMENTS:
1. Class ResponseFormatter with:
   - __init__(bus: EventBus)
   - async on_response_ready(event: ResponseReady):
       Print with "KANCHA: " prefix
       If event.requires_task: also print task confirmation
   - register(): subscribe on_response_ready to ResponseReady events

DESIGN RULES:
- Subscribe to ResponseReady in register() not __init__
- Prefix every response with "KANCHA: " in a distinct color (use ANSI codes)
- Log the response and its latency (timestamp comparison with event.timestamp)
- Never block on print — it's synchronous but acceptable for CLI

ACCEPTANCE CRITERIA:
- ResponseReady event triggers printed output
- Output has KANCHA: prefix
- Task responses show confirmation
```

---

### FILE: main.py

```
TASK: Write main.py

PURPOSE:
Entry point for KANCHA. Wires all modules together and starts the system.
Handles startup, running, and graceful shutdown.

REQUIREMENTS:
async def run():
1. Load config via get_config()
2. Setup logging via setup_logging(config)
3. Instantiate EventBus
4. Instantiate ConversationContext(max_turns=config.max_context_turns)
5. Instantiate StructuredMemory(db_path=get_data_dir() / config.sqlite_db_name)
6. Instantiate VectorMemory(persist_dir, collection_name, ollama_url, embedding_model)
7. Instantiate MemoryManager(bus, structured, vector, context, config)
8. await memory_manager.initialize()
9. Instantiate GeminiClient(api_key, model)
10. await gemini_client.initialize()
11. Instantiate RAGPipeline(memory_manager, bus, config)
12. Instantiate NLUClassifier(gemini_client)
13. Instantiate PromptBuilder(config)
14. Subscribe all handlers to bus:
    - NLUClassifier.on_text_input → TextInputReceived
    - NLUClassifier.on_transcript_ready → TranscriptReady
    - RAGPipeline.on_intent_identified → IntentIdentified
    - MemoryManager.on_memory_update_needed → MemoryUpdateNeeded
15. Instantiate ResponseFormatter(bus) and call register()
16. Instantiate TextInputHandler(bus, config)
17. Subscribe ShutdownRequested to shutdown handler
18. Print startup banner
19. await text_input.run()
20. On shutdown: await bus.drain(), close all backends

if __name__ == "__main__":
    asyncio.run(run())

DESIGN RULES:
- Startup order matters: memory before LLM, LLM before NLU, NLU before input
- Every instantiation wrapped in try/except with clear error message
- Shutdown must be graceful: drain bus, close DB connections, stop Chroma
- Print a clean startup banner showing all active modules and their status

ACCEPTANCE CRITERIA:
- python main.py starts without errors (requires Ollama + Gemini key)
- Type a message, get a response
- Memory persists: restart main.py, ask about previous conversation
- Ctrl+C shuts down cleanly with no hanging tasks
```

---

## PHASE 5 — TASK EXECUTION ENGINE

### FILE: kancha/tasks/registry.py

```
TASK: Write kancha/tasks/registry.py

PURPOSE:
The security boundary for all task execution.
Defines EXACTLY which tasks are allowed and what parameters they accept.
Nothing outside this registry can ever be executed.

REQUIREMENTS:
1. TaskSpec dataclass with:
   - name: str
   - description: str
   - required_params: list[str]
   - optional_params: list[str]
   - param_types: dict[str, type]

2. TASK_REGISTRY: dict[str, TaskSpec] with these allowed tasks:
   - "set_reminder": requires [description, delay_seconds], optional [repeat]
   - "save_note": requires [content], optional [title, tags]
   - "get_tasks": requires [], optional [status, limit]
   - "get_facts": requires [], optional [key]
   - "save_fact": requires [key, value]
   - "open_url": requires [url], optional [] — ONLY allow http/https, validate URL
   - "get_time": requires [], optional [timezone]
   - "get_weather": requires [city], optional [units]

3. Functions:
   - validate_task(task_type: str, params: dict) -> tuple[bool, str]:
       check task_type in registry
       check all required_params present
       check param types match
       return (True, "") or (False, error_message)
   - get_allowed_tasks() -> list[str]

DESIGN RULES:
- This is a hardcoded allowlist — never dynamic, never user-configurable at runtime
- validate_task() is called BEFORE any execution — always
- open_url must validate URL scheme is http or https — no file://, no localhost
- Log every validation attempt with result

ACCEPTANCE CRITERIA:
- validate_task("set_reminder", {"description": "test", "delay_seconds": 60}) → (True, "")
- validate_task("unknown_task", {}) → (False, "Task type not allowed: unknown_task")
- validate_task("set_reminder", {}) → (False, "Missing required params: ...")
- open_url with file:// URL → (False, ...)
```

---

### FILE: kancha/tasks/scheduler.py

```
TASK: Write kancha/tasks/scheduler.py

PURPOSE:
asyncio-based reminder and scheduled task system.
Reminders fire at the right time even if set while other tasks are running.
Pending reminders survive process restarts via SQLite storage.

REQUIREMENTS:
1. Class TaskScheduler with:
   - __init__(bus: EventBus, structured_memory: StructuredMemory)
   - async initialize(): load pending tasks from SQLite, reschedule them
   - async schedule_reminder(description: str, delay_seconds: float, session_id: str, task_id: str = None) -> str:
       Create task_id if not provided
       Store in SQLite with status="pending"
       asyncio.create_task(_fire_reminder(task_id, description, delay_seconds, session_id))
       Return task_id
   - async _fire_reminder(task_id, description, delay_seconds, session_id):
       await asyncio.sleep(delay_seconds)
       Update SQLite status to "fired"
       Emit ResponseReady(text=f"⏰ Reminder: {description}", session_id=session_id)
   - async cancel_reminder(task_id: str) -> bool
   - async get_pending_reminders(session_id: str) -> list[dict]
   - async _reschedule_on_startup(): load pending from DB, fire those still in future

DESIGN RULES:
- Store due_at as ISO timestamp string so scheduler can calculate remaining seconds on restart
- If a reminder's due_at has already passed on restart: fire immediately
- Track asyncio Task objects by task_id for cancellation
- On process shutdown: persist all pending tasks — they'll be rescheduled on next start

ACCEPTANCE CRITERIA:
- schedule_reminder with 5 second delay fires after 5 seconds
- Cancelled reminder does not fire
- Pending reminder survives restart and fires on next startup
```

---

### FILE: kancha/tasks/executor.py

```
TASK: Write kancha/tasks/executor.py

PURPOSE:
Validated task dispatcher. Receives TaskExecutionRequested events,
validates against the registry, executes the approved task handler,
and emits TaskCompleted.

REQUIREMENTS:
1. Class TaskExecutor with:
   - __init__(bus: EventBus, scheduler: TaskScheduler, structured_memory: StructuredMemory, config)
   - async on_task_requested(event: TaskExecutionRequested):
       validate with registry.validate_task()
       if invalid: emit TaskCompleted(success=False, error=reason)
       dispatch to correct handler
       emit TaskCompleted with result
   - Handler methods (one per task type):
     - async _handle_set_reminder(params, session_id) -> str
     - async _handle_save_note(params, session_id) -> str
     - async _handle_get_tasks(params, session_id) -> str
     - async _handle_get_facts(params, session_id) -> str
     - async _handle_save_fact(params, session_id) -> str
     - async _handle_open_url(params, session_id) -> str (uses webbrowser.open)
     - async _handle_get_time(params, session_id) -> str
   - register(): subscribe on_task_requested to TaskExecutionRequested

DESIGN RULES:
- ALWAYS validate before ANY execution — no exceptions
- Never use subprocess in Phase 5
- webbrowser.open() is acceptable for open_url (stdlib, safe)
- Log every task attempt: type, params (sanitized), result, latency
- Each handler returns a human-readable result string

ACCEPTANCE CRITERIA:
- TaskExecutionRequested("set_reminder", valid_params) → fires reminder, emits TaskCompleted(success=True)
- TaskExecutionRequested("unknown_task", {}) → emits TaskCompleted(success=False)
- TaskExecutionRequested("open_url", {"url": "file:///etc/passwd"}) → TaskCompleted(success=False)
```

---

## PHASE 6 — VOICE LAYER

### FILE: kancha/input/stt.py

```
TASK: Write kancha/input/stt.py

PURPOSE:
Speech-to-text using faster-whisper.
Records audio from microphone, detects speech end via silence detection,
transcribes with Whisper tiny model, emits TranscriptReady.

REQUIREMENTS:
1. Class STTProcessor with:
   - __init__(bus, config, model_size="tiny")
   - async initialize(): load WhisperModel in executor (CPU-bound, blocks)
   - async transcribe_file(audio_path: Path) -> str: transcribe a WAV file
   - async transcribe_from_microphone() -> str:
       Record audio chunks via sounddevice
       Detect silence (energy below threshold for 1.5 seconds)
       Save to temp WAV file
       Transcribe in executor
       Return text
   - async on_wake_word_detected(event: WakeWordDetected):
       If audio_path provided: transcribe file
       Else: record from microphone
       Emit TranscriptReady

DESIGN RULES:
- WhisperModel loading and transcription are CPU-bound — ALWAYS use run_in_executor
- Use sounddevice.rec() for recording — configure via platform.get_audio_config()
- Energy-based VAD: if RMS of chunk below SILENCE_THRESHOLD for SILENCE_DURATION → stop recording
- Save audio as 16000Hz mono WAV (Whisper requirement)
- Log: transcription text, WER if available, latency

ACCEPTANCE CRITERIA:
- transcribe_file() on a known WAV returns correct text
- Event loop is not blocked during transcription (verify with concurrent task)
- TranscriptReady emitted after recording + transcription
```

---

### FILE: kancha/input/wake_word.py

```
TASK: Write kancha/input/wake_word.py

PURPOSE:
Always-on wake word detection using openWakeWord.
Runs in a background thread (not the event loop).
Emits WakeWordDetected events via thread-safe bridge to the event loop.

REQUIREMENTS:
1. Class WakeWordDetector with:
   - __init__(bus, model_path: Path, threshold: float = 0.5)
   - async start(): start background thread
   - async stop(): set stop event, join thread
   - _detection_loop(loop: asyncio.AbstractEventLoop):
       Run in background thread
       Initialize openWakeWord model
       Continuously read audio from sounddevice InputStream
       When score > threshold: emit via asyncio.run_coroutine_threadsafe()
       Mute detection while TTS is playing (check self._muted flag)
   - async mute(): set _muted = True (call before TTS plays)
   - async unmute(): set _muted = False (call after TTS finishes)

DESIGN RULES:
- CRITICAL: wake word runs in a THREAD, not a coroutine
- CRITICAL: use asyncio.run_coroutine_threadsafe(bus.emit(event), loop) to bridge thread→loop
- Get the event loop BEFORE starting the thread: loop = asyncio.get_event_loop()
- Set daemon=True on the thread so it doesn't prevent shutdown
- Mute during TTS playback to prevent feedback loop (hearing own voice as wake word)
- Log every detection with confidence score

ACCEPTANCE CRITERIA:
- Detector starts in background without blocking event loop
- WakeWordDetected event arrives on the event loop (not the thread)
- Muting prevents events during TTS playback
- Stop() joins the thread cleanly
```

---

### FILE: kancha/output/tts.py

```
TASK: Write kancha/output/tts.py

PURPOSE:
Text-to-speech synthesis using Piper TTS.
Converts ResponseReady text to audio and triggers playback.
Mutes wake word detector during playback to prevent feedback.

REQUIREMENTS:
1. Class TTSEngine with:
   - __init__(bus, wake_word_detector, voice_model_path: Path)
   - async initialize(): verify Piper model exists, warm up
   - async synthesize(text: str) -> Path: convert text to WAV file, return path
   - async speak(text: str): synthesize then play audio
   - async on_response_ready(event: ResponseReady):
       await wake_word_detector.mute()
       await speak(event.text)
       await wake_word_detector.unmute()
   - _play_audio(wav_path: Path): play via sounddevice in executor
   - register(): subscribe to ResponseReady

DESIGN RULES:
- Synthesis is CPU-bound — run in executor
- Playback blocks until audio finishes — that's intentional (can't listen while speaking)
- Always unmute wake word in a finally block — even if synthesis fails
- Clean up temp WAV files after playback
- Log: text length, synthesis latency, playback duration

ACCEPTANCE CRITERIA:
- on_response_ready plays audio and then unmutes detector
- Wake word is muted during entire speak() call
- Unmute happens even if synthesis raises an exception
- Temp files are cleaned up after playback
```

---

### FILE: tests/test_phase6.py

```
TASK: Write tests/test_phase6.py

REQUIREMENTS:
Test in async main():
1. STTProcessor.transcribe_file() on a test WAV returns non-empty string
2. TTSEngine.synthesize() returns a valid WAV path
3. WakeWordDetector starts and stops without error
4. WakeWordDetector.mute() prevents events from firing
5. Full pipeline test: inject TranscriptReady → verify ResponseReady fires
   (skip actual audio synthesis in CI)

Include a note that tests 1-2 require audio hardware and model files.
Skip gracefully if models not present.
```

---

## PHASE 7 — HARDENING AND DOCUMENTATION

### FILE: scripts/benchmark_latency.py

```
TASK: Write scripts/benchmark_latency.py

PURPOSE:
Measures and reports latency of every pipeline stage.
Run this after Phase 6 to get real performance numbers for the report.

REQUIREMENTS:
Run 20 test queries through the full pipeline and measure:
- STT latency (file transcription time)
- Embedding latency (nomic-embed-text via Ollama)
- Memory retrieval latency (Chroma query time)
- LLM latency (Gemini API response time)
- TTS synthesis latency (Piper synthesis time)
- End-to-end latency (total from text input to response ready)

Report:
- P50 (median)
- P95 (95th percentile)
- Min and Max
- Mean

Format output as a clean table printable to terminal.
Also save results as JSON to data/benchmark_results.json.
```

---

### FILE: Makefile

```
TASK: Write Makefile

REQUIREMENTS:
Targets:
- install: pip install -e ".[dev]"
- run: python main.py
- voice: python main.py --voice
- test: pytest tests/ -v --tb=short
- test-phase1 through test-phase6: run individual phase tests
- benchmark: python scripts/benchmark_latency.py
- lint: ruff check kancha/ && mypy kancha/
- clean: remove __pycache__, .pyc files
- setup: interactive first-time setup (ask for Gemini key, write .env)

All targets must have a brief comment explaining what they do.
```

---

### FILE: pyproject.toml

```
TASK: Write pyproject.toml

REQUIREMENTS:
[build-system]: setuptools + wheel

[project]:
- name: kancha
- version: 0.1.0
- requires-python: >=3.10
- description: Multimodal personalized AI assistant with persistent memory
- dependencies: list all required packages with minimum versions:
  google-genai>=1.0.0, chromadb>=0.5.0, aiosqlite>=0.20.0,
  faster-whisper>=1.0.0, sounddevice>=0.4.6, openWakeWord>=0.6.0,
  pydantic>=2.0.0, python-dotenv>=1.0.0, httpx>=0.27.0,
  piper-tts>=1.2.0, pystray>=0.19.0, Pillow>=10.0.0,
  numpy>=1.24.0, PyYAML>=6.0

[project.optional-dependencies]:
- dev: pytest, pytest-asyncio, ruff, mypy

[project.scripts]:
- kancha = "kancha.__main__:main"

[tool.ruff]: line-length = 100, select common rules
[tool.mypy]: strict = false, ignore_missing_imports = true
[tool.pytest.ini_options]: asyncio_mode = "auto"
```

---

### FILE: .env.example

```
TASK: Write .env.example

CONTENT:
# KANCHA Environment Configuration
# Copy this file to .env and fill in your values
# NEVER commit .env to git

# Required: Get your free API key at https://aistudio.google.com
GEMINI_API_KEY=your_gemini_api_key_here

# Optional overrides (defaults shown)
KANCHA_LOG_LEVEL=INFO
KANCHA_SESSION_ID=default
# KANCHA_DATA_DIR=/custom/path/to/data  # uncomment to override
```

---

### FILE: .gitignore

```
TASK: Write .gitignore

MUST INCLUDE:
.env (the real one — never commit)
.venv/
__pycache__/
*.pyc
data/chroma/
data/sqlite/
data/*.db
models/*.onnx
models/*.tflite
dist/
build/
*.egg-info/
.DS_Store
.idea/
.vscode/
benchmark_results.json
*.wav (temp audio files)
```

---

## FINAL CHECKLIST
> Run through this before calling the project complete

### Code Quality
- [ ] All files have module-level docstrings
- [ ] All public methods have complete type hints
- [ ] No blocking calls in any async function
- [ ] No hardcoded file paths (all use pathlib + config)
- [ ] No hardcoded API keys or secrets
- [ ] All error paths have logging

### Testing
- [ ] test_phase1.py: 8/8 passing
- [ ] test_phase2.py: all passing (skip if Ollama unavailable)
- [ ] test_phase3.py: all passing (skip if no API key)
- [ ] test_phase4.py: all passing
- [ ] test_phase5.py: all passing
- [ ] test_phase6.py: all passing (skip if no audio hardware)

### Performance
- [ ] End-to-end latency P50 under 3 seconds
- [ ] End-to-end latency P95 under 6 seconds
- [ ] STT transcription under 1.5 seconds (whisper tiny, CPU)
- [ ] Gemini response under 2 seconds
- [ ] Memory retrieval under 500ms

### Security
- [ ] Task registry validated before every execution
- [ ] No arbitrary subprocess calls
- [ ] open_url validates http/https scheme only
- [ ] .env never committed (verify with git log)
- [ ] API key never appears in logs

### Deployment
- [ ] install.sh works on clean Linux machine
- [ ] systemd service file present and tested
- [ ] pystray tray icon starts with system
- [ ] README has complete setup instructions
- [ ] Demo video recorded showing memory persistence across restarts