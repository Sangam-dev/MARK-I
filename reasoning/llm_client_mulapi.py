"""
KANCHA — Gemini client with multi-key rotation + hedged model fallback.

Key rotation logic:
  - Keys are tried round-robin.
  - On 429 / quota exhausted → key is marked cooling for COOLDOWN_SECS.
  - Cooling keys are skipped; they re-enter the pool automatically.
  - If ALL keys are cooling, we wait for the soonest recovery instead of dying.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from typing import Any, AsyncIterator

from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai import errors as genai_errors

except ImportError:
    sys.exit("Run: pip install google-genai")


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
DEFAULT_FALLBACKS = os.getenv(
    "GEMINI_MODEL_FALLBACKS",
    "gemini-3.5-flash,gemini-2.0-flash,gemini-2.0-flash-lite",
)
DEFAULT_HEDGE = int(os.getenv("HEDGE", "2"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "12.0"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))
COOLDOWN_SECS = float(
    os.getenv("KEY_COOLDOWN_SECS", "60.0")
)  # per-key cooldown on quota hit


# ── Key pool ──────────────────────────────────────────────────────────────────


class _KeyEntry:
    def __init__(self, key: str, index: int):
        self.key = key
        self.index = index
        self.client = genai.Client(api_key=key)
        self.cooling_until: float = 0.0  # epoch seconds; 0 = available

    @property
    def is_available(self) -> bool:
        return time.monotonic() >= self.cooling_until

    def cool_down(self, secs: float = COOLDOWN_SECS) -> None:
        self.cooling_until = time.monotonic() + secs
        _log(f"  🔑 key[{self.index}] cooling for {secs:.0f}s")

    def secs_until_ready(self) -> float:
        return max(0.0, self.cooling_until - time.monotonic())


class KeyPool:
    """
    Round-robin pool of Gemini API keys.
    Thread-safe via asyncio.Lock (single-threaded async loop).
    """

    def __init__(self, keys: list[str]):
        if not keys:
            sys.exit(
                "No Gemini API keys found.\n"
                "Set GEMINI_API_KEY_1, GEMINI_API_KEY_2, … (or GEMINI_API_KEY) in .env\n"
            )
        self._entries = [_KeyEntry(k, i) for i, k in enumerate(keys)]
        self._cursor = 0
        self._lock = asyncio.Lock()

    async def next(self) -> _KeyEntry:
        """Return the next available key entry, waiting if all are cooling."""
        async with self._lock:
            n = len(self._entries)
            for _ in range(n):
                entry = self._entries[self._cursor % n]
                self._cursor += 1
                if entry.is_available:
                    return entry

            wait = min(e.secs_until_ready() for e in self._entries)
            _log(f"  ⏳ all keys cooling — waiting {wait:.1f}s for next available key")

        await asyncio.sleep(wait + 0.1)
        return await self.next()  # recurse after wait (lock released above)

    def mark_quota(self, entry: _KeyEntry, exc: Exception) -> None:
        """Cool a key down based on error type."""
        if _is_quota_exhausted(exc):
            # Daily quota gone — cool for much longer (won't recover today)
            entry.cool_down(secs=3600.0)
        else:
            server_wait = _parse_retry_after(exc)
            entry.cool_down(secs=server_wait if server_wait else COOLDOWN_SECS)

    def status(self) -> str:
        lines = []
        for e in self._entries:
            if e.is_available:
                lines.append(f"  key[{e.index}] ✅ available")
            else:
                lines.append(f"  key[{e.index}] 🔴 cooling {e.secs_until_ready():.0f}s")
        return "\n".join(lines)


def _load_key_pool() -> KeyPool:
    """
    Load keys from env.  Supports two styles:
      Single:   GEMINI_API_KEY=key1
      Multi:    GEMINI_API_KEY_1=key1  GEMINI_API_KEY_2=key2  ...
    Both can coexist — duplicates are deduplicated.
    """
    keys: list[str] = []
    seen: set[str] = set()

    for i in range(1, 10):
        k = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
        if k and k not in seen:
            keys.append(k)
            seen.add(k)

    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        k = os.getenv(name, "").strip()
        if k and k not in seen:
            keys.append(k)
            seen.add(k)

    _log(f"Loaded {len(keys)} API key(s)")
    return KeyPool(keys)


# ── Error helpers ─────────────────────────────────────────────────────────────


def _log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _parse_retry_after(exc: Exception) -> float | None:
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", str(exc))
    return float(match.group(1)) if match else None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, genai_errors.APIError):
        return exc.code in (429, 503)
    msg = str(exc).lower()
    return any(t in msg for t in ("429", "503", "resource_exhausted", "unavailable"))


def _is_quota_exhausted(exc: Exception) -> bool:
    msg = str(exc)
    return "limit: 0" in msg or "GenerateRequestsPerDay" in msg


def _is_not_found(exc: Exception) -> bool:
    if isinstance(exc, genai_errors.APIError):
        return exc.code == 404
    msg = str(exc).lower()
    return any(t in msg for t in ("404", "not_found", "not found"))


def _split_models(value: str) -> list[str]:
    return [m.strip() for m in value.split(",") if m.strip()]


# ── Core: generate with key rotation ─────────────────────────────────────────


async def _generate_call(
    pool: KeyPool,
    model: str,
    timeout: float,
    **call_kwargs: Any,
) -> str:
    """
    Shared retry/rotation core for both plain-prompt and structured-conversation
    calls. `call_kwargs` is passed straight through to `generate_content`
    (e.g. contents=... or contents=..., config=...).

    NOTE: each key-rotation on a quota hit still consumes one iteration of
    `range(MAX_RETRIES + 1)` — it does NOT get a free retry. With the default
    MAX_RETRIES=1 you get 2 total attempts before this raises, regardless of
    how many healthy keys remain in the pool. Bump MAX_RETRIES if you want
    rotation to actually exhaust the pool before giving up.
    """
    for attempt in range(MAX_RETRIES + 1):
        entry = await pool.next()
        try:
            resp = await asyncio.wait_for(
                asyncio.to_thread(
                    entry.client.models.generate_content,
                    model=model,
                    **call_kwargs,
                ),
                timeout=timeout,
            )
            return getattr(resp, "text", "") or ""

        except asyncio.TimeoutError:
            _log(f"[{model}] key[{entry.index}] timeout after {timeout}s")
            raise

        except Exception as exc:
            if _is_not_found(exc):
                _log(f"[{model}] not found — skipping model")
                raise

            if _is_retryable(exc):
                pool.mark_quota(entry, exc)
                if attempt < MAX_RETRIES:
                    _log(f"[{model}] key[{entry.index}] quota hit — rotating key")
                    continue

            raise


async def _generate_one(pool: KeyPool, model: str, prompt: str, timeout: float) -> str:
    return await _generate_call(pool, model, timeout, contents=prompt)


async def _generate_one_conv(
    pool: KeyPool,
    model: str,
    contents: list,
    config: Any,
    timeout: float,
) -> str:
    """Like _generate_one but accepts structured contents + GenerateContentConfig."""
    kwargs: dict[str, Any] = {"contents": contents}
    if config is not None:
        kwargs["config"] = config
    return await _generate_call(pool, model, timeout, **kwargs)


async def _stream_one(
    pool: KeyPool,
    model: str,
    prompt: str,
    timeout: float,
) -> AsyncIterator[str]:
    """
    Streaming version with key rotation on quota errors.
    Yields text chunks; raises on unrecoverable errors.
    """
    for attempt in range(MAX_RETRIES + 1):
        entry = await pool.next()
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None | Exception] = asyncio.Queue()

        def _producer() -> None:
            try:
                response = entry.client.models.generate_content_stream(
                    model=model,
                    contents=prompt,
                )
                for chunk in response:
                    text = getattr(chunk, "text", "")
                    if text:
                        loop.call_soon_threadsafe(queue.put_nowait, text)
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)

        # Keep a reference — an unreferenced Task can be garbage-collected
        # mid-flight (see asyncio docs on fire-and-forget tasks).
        producer_task = asyncio.ensure_future(asyncio.to_thread(_producer))
        _ = producer_task  # kept alive for the life of this generator frame

        try:
            first_chunk = True
            while True:
                item = await asyncio.wait_for(
                    queue.get(),
                    timeout=timeout if first_chunk else None,
                )
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                first_chunk = False
                yield item
            return

        except asyncio.TimeoutError:
            _log(f"[{model}] key[{entry.index}] stream TTFT timeout")
            raise

        except Exception as exc:
            if _is_not_found(exc):
                _log(f"[{model}] not found — skipping")
                raise

            if _is_retryable(exc):
                pool.mark_quota(entry, exc)
                if attempt < MAX_RETRIES:
                    _log(
                        f"[{model}] key[{entry.index}] quota hit — rotating key for stream"
                    )
                    continue

            raise


# ── Hedged requests ───────────────────────────────────────────────────────────


async def _hedged_race(
    models: list[str],
    hedge_width: int,
    run_one: Any,  # async fn(model: str) -> str
) -> str:
    """
    Shared race/fallback logic: fires `run_one` for the first `hedge_width`
    models concurrently, returns the first success, cancels the rest.
    If all hedged attempts fail, falls back sequentially through the tail.
    """
    if not models:
        raise ValueError("No models provided")

    hedged = models[:hedge_width]
    tail = models[hedge_width:]

    tasks: dict[asyncio.Task, str] = {
        asyncio.create_task(run_one(m), name=m): m for m in hedged
    }

    while tasks:
        done, _ = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            model_name = tasks.pop(task)
            if task.exception() is None:
                for t in tasks:
                    t.cancel()
                _log(f"[{model_name}] won hedge race ✓")
                return task.result()
            _log(f"[{model_name}] hedge failed: {task.exception()}")

    for model in tail:
        try:
            _log(f"[{model}] sequential fallback")
            return await run_one(model)
        except Exception as exc:
            _log(f"[{model}] failed: {exc}")

    raise RuntimeError("All models + keys exhausted — no response available")


async def hedged_generate(
    pool: KeyPool,
    models: list[str],
    prompt: str,
    hedge_width: int,
    timeout: float,
) -> str:
    return await _hedged_race(
        models, hedge_width, lambda m: _generate_one(pool, m, prompt, timeout)
    )


async def hedged_generate_conv(
    pool: KeyPool,
    models: list[str],
    contents: list,
    config: Any,
    hedge_width: int,
    timeout: float,
) -> str:
    """Hedged multi-model race for structured conversation requests."""
    return await _hedged_race(
        models,
        hedge_width,
        lambda m: _generate_one_conv(pool, m, contents, config, timeout),
    )


async def hedged_stream(
    pool: KeyPool,
    models: list[str],
    prompt: str,
    hedge_width: int,
    timeout: float,
) -> None:
    if not models:
        raise ValueError("No models provided")

    hedged = models[:hedge_width]
    tail = models[hedge_width:]
    start = time.perf_counter()

    async def _race_first_chunk(model: str):
        gen = _stream_one(pool, model, prompt, timeout)
        first = await gen.__anext__()
        return model, first, gen

    tasks: dict[asyncio.Task, str] = {
        asyncio.create_task(_race_first_chunk(m), name=m): m for m in hedged
    }

    winner_model = winner_first = winner_gen = None

    while tasks and winner_model is None:
        done, _ = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            model_name = tasks.pop(task)
            if task.exception() is None:
                winner_model, winner_first, winner_gen = task.result()
                for t in tasks:
                    t.cancel()
                break
            else:
                _log(f"[{model_name}] stream race failed: {task.exception()!r}")

    if winner_model is None:
        for model in tail:
            try:
                gen = _stream_one(pool, model, prompt, timeout)
                winner_first = await gen.__anext__()
                winner_model, winner_gen = model, gen
                break
            except Exception as exc:
                _log(f"[{model}] tail stream failed: {exc!r}")

    if winner_model is None:
        raise RuntimeError("All models + keys exhausted")

    _log(f"[{winner_model}] streaming | TTFT {time.perf_counter() - start:.2f}s")
    print(winner_first, end="", flush=True)
    async for chunk in winner_gen:
        print(chunk, end="", flush=True)
    print(flush=True)
    _log(f"[Done {time.perf_counter() - start:.2f}s]")


# ── Module-level singleton (for KANCHA integration) ───────────────────────────
_pool: KeyPool | None = None


def get_pool() -> KeyPool:
    """Return (or create) the global key pool. Call once at startup."""
    global _pool
    if _pool is None:
        _pool = _load_key_pool()
    return _pool


# ── Exports for KANCHA ────────────────────────────────────────────────────────

ALL_MODELS: list[str] = list(
    dict.fromkeys([DEFAULT_MODEL] + _split_models(DEFAULT_FALLBACKS))
)

# Alias so kancha.py can import a consistent name
_stream_raw = _stream_one