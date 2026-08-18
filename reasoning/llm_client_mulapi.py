"""
KANCHA — Gemini client with sticky active-key failover + model fallback.

Selection strategy
------------------
The pool keeps **one active key** at a time. Every request (streaming or
non-streaming) uses that same key. The key only changes when it becomes
unusable (daily quota exhausted, invalid key, unrecoverable network
failure, repeated retryable errors).

Per key, models are tried in priority order. A model that returns
NOT_FOUND for a key is recorded on that key's *model capability cache*
and never tried again on that key for the process lifetime. The next
model in the priority list is attempted on the **same key** — we only
advance to the next key once every configured model has failed for the
current key.

Why this layout
---------------
  * **Lowest latency** — no per-request key rotation; the active key's
    HTTPS connection is reused.
  * **Lowest token usage** — exactly one generation request is in
    flight per user request, ever.
  * **Maximum lifetime** — Key 1 keeps serving until its quota is
    actually gone; we don't waste Key 2's quota on requests Key 1 could
    have served.
  * **High reliability** — when the active key fails the entire model
    list, the next key becomes active automatically.

Compatibility
-------------
All public symbols, function signatures, environment variables, and
streaming behaviour used by KANCHA are preserved. The
``hedged_*`` functions are kept as thin wrappers around the new
sticky driver.

Environment variables
---------------------
  GEMINI_MODEL                       primary model
  GEMINI_MODEL_FALLBACKS             comma-separated fallback models
  HEDGE                              retained for compatibility
  REQUEST_TIMEOUT                    per-request timeout in seconds
  KEY_COOLDOWN_SECS                  cooldown after a temporary rate limit
  KEY_DAILY_QUOTA_COOLDOWN_SECS      cooldown after a daily-quota hit
                                     (default 3600s — daily reset assumed)
  MAX_RETRIES                        retained for compatibility, no
                                     longer bounds key rotation
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
    from google.genai import errors as genai_errors,types

except ImportError:
    sys.exit("Run: pip install google-genai")


# ── Config ────────────────────────────────────────────────────────────────────gemini-flash-lite-latest
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
DEFAULT_FALLBACKS = os.getenv(
    "GEMINI_MODEL_FALLBACKS",
    "gemini-3.5-flash,gemini-2.0-flash,gemini-2.0-flash-lite",
)
DEFAULT_HEDGE = int(os.getenv("HEDGE", "1"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "12.0"))
# Retained for backward compatibility — no longer bounds key rotation.
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))
COOLDOWN_SECS = float(
    os.getenv("KEY_COOLDOWN_SECS", "60.0")
)  # per-key cooldown on a temporary rate limit
DAILY_QUOTA_COOLDOWN_SECS = float(
    os.getenv("KEY_DAILY_QUOTA_COOLDOWN_SECS", "3600.0")
)  # per-key cooldown when the daily quota is gone


# ── Key pool ──────────────────────────────────────────────────────────────────


class _KeyEntry:
    """A single API key plus its per-process model capability cache.

    The capability cache lets us skip a (key, model) pair we already
    know cannot work, without burning an HTTP round-trip. Both sets are
    process-lifetime only — a process restart re-probes from scratch.
    """

    def __init__(self, key: str, index: int):
        self.key = key
        self.index = index
        self.client = genai.Client(api_key=key)
        self.cooling_until: float = 0.0  # epoch seconds; 0 = available

        # Per-key model capability cache.
        self.supported_models: set[str] = set()
        self.unsupported_models: set[str] = set()

        # A key can be marked permanently dead (invalid auth, unrecoverable
        # configuration error). A dead key never re-enters rotation for
        # the process lifetime.
        self.dead: bool = False

    # ── Availability ─────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return not self.dead and time.monotonic() >= self.cooling_until

    def cool_down(self, secs: float, reason: str = "rate limit") -> None:
        self.cooling_until = time.monotonic() + secs
        _log(f"  🔑 key[{self.index}] cooling for {secs:.0f}s ({reason})")

    def secs_until_ready(self) -> float:
        return max(0.0, self.cooling_until - time.monotonic())

    def mark_dead(self, reason: str) -> None:
        self.dead = True
        _log(f"  💀 key[{self.index}] dead ({reason}) — removed from rotation")

    # ── Model capability cache ───────────────────────────────────────────

    def supports(self, model: str) -> bool:
        """Return True if this key is known to support the model.

        Unknown models (never tried) are conservatively treated as
        supported so we still probe them. Once a model lands in
        ``unsupported_models`` we skip the key entirely.
        """
        return model not in self.unsupported_models

    def mark_unsupported(self, model: str) -> None:
        if model not in self.unsupported_models:
            self.unsupported_models.add(model)
            _log(f"  📕 key[{self.index}] marked as NOT supporting '{model}'")

    def mark_supported(self, model: str) -> None:
        # A successful response is the only authoritative signal that a
        # key can serve a model. Recording the positive keeps future
        # lookups O(1) and makes the cache self-pruning.
        self.supported_models.add(model)

    def has_any_usable_model(self, models: list[str]) -> bool:
        """Return True if at least one model in ``models`` is not
        already cached as unsupported on this key. Used to decide
        whether the key has *anything* to offer before we cycle off it.
        """
        return any(m not in self.unsupported_models for m in models)


FORWARD = 1
REVERSE = -1


class KeyLane:
    """One consumer's cursor over a shared :class:`KeyPool`.

    Key *health* — cooldowns, dead flags, per-model capability — is a
    property of the key itself and stays shared: a key exhausted by one
    caller is exhausted for everyone, and discovering that twice would
    waste a request.

    What is **not** shared is where a consumer starts and which way it
    walks. Two lanes over the same pool let the Conversation LLM and the
    Task LLM sit on different keys instead of contending for one, which
    matters because they frequently run in the same turn: the controller
    is streaming its acknowledgement while the planner is decomposing.

    A lane running ``REVERSE`` from the last key meets a ``FORWARD`` lane
    in the middle only once the keys between them are cooling or dead —
    at which point sharing is correct, because there is nothing else left
    to use.
    """

    def __init__(
        self,
        pool: "KeyPool",
        name: str,
        direction: int = FORWARD,
        start: int | None = None,
    ) -> None:
        self._pool = pool
        self.name = name
        self._step = FORWARD if direction >= 0 else REVERSE
        n = len(pool)
        if start is None:
            # Forward lanes open on the first key, reverse lanes on the
            # last, so with one pool and two lanes nothing overlaps until
            # it has to.
            start = 0 if self._step == FORWARD else n - 1
        self._active_index = start % n
        self._lock = asyncio.Lock()

    def __len__(self) -> int:
        return len(self._pool)

    @property
    def active_index(self) -> int:
        return self._active_index

    def _offset(self, base: int, steps: int) -> int:
        return (base + self._step * steps) % len(self._pool)

    # ── Sticky active-key selection ──────────────────────────────────────

    async def acquire_active(self) -> _KeyEntry:
        """Return this lane's active key, waiting if it's cooling.

        If the active key is dead or cooling, step to the next available
        key *in this lane's direction*. If every key is dead, raise
        ``RuntimeError`` — there is no recovery. If every key is
        currently cooling, wait for the soonest (single sleep, no
        busy-loop).
        """
        entries = self._pool.entries()
        async with self._lock:
            n = len(entries)
            for _ in range(n):
                entry = entries[self._active_index]
                if entry.is_available:
                    return entry
                # Step to the next live key. ``is_available`` is False
                # for both cooling and dead keys, so a dead key is
                # naturally skipped. We do NOT step past dead keys
                # permanently — if every other key is also dead we fall
                # through to the wait-or-raise branch below.
                self._active_index = self._offset(self._active_index, 1)

            live = [e for e in entries if not e.dead]
            if not live:
                raise RuntimeError(
                    "All API keys are dead — no recovery possible within "
                    "this process. Restart to retry."
                )

            soonest = min(e.secs_until_ready() for e in live)
            self._active_index = entries.index(
                min(live, key=lambda e: e.secs_until_ready())
            )
            _log(
                f"  ⏳ [{self.name}] all live keys cooling — waiting "
                f"{soonest:.1f}s for key[{entries[self._active_index].index}]"
            )

        # Release the lock while we sleep so other coroutines aren't
        # blocked. The next acquire_active() call will see the recovered
        # key and return it without re-sleeping.
        await asyncio.sleep(soonest + 0.1)
        return await self.acquire_active()

    async def advance_active(self) -> None:
        """Step this lane's cursor to the next available key.

        Called by the driver after a key-level failure (quota, auth,
        timeout, repeated retryable error). If no live key is available
        the cursor stays put and the next :meth:`acquire_active` either
        waits for a cooldown or raises.
        """
        entries = self._pool.entries()
        async with self._lock:
            n = len(entries)
            for steps in range(1, n):
                idx = self._offset(self._active_index, steps)
                if entries[idx].is_available:
                    self._active_index = idx
                    return
            # No other live key — leave the cursor alone. acquire_active
            # will see this and either wait or raise.

    async def next(self) -> _KeyEntry | None:
        """Legacy round-robin accessor. New code uses
        :meth:`acquire_active`."""
        entries = self._pool.entries()
        async with self._lock:
            for _ in range(len(entries)):
                entry = entries[self._active_index]
                self._active_index = self._offset(self._active_index, 1)
                if entry.is_available:
                    return entry
            return None

    # ── Key state mutations delegate to the shared pool ──────────────────

    def mark_quota(self, entry: _KeyEntry, exc: Exception) -> None:
        self._pool.mark_quota(entry, exc)

    def mark_not_found(self, entry: _KeyEntry, model: str) -> None:
        self._pool.mark_not_found(entry, model)

    def mark_invalid(self, entry: _KeyEntry, exc: Exception) -> None:
        self._pool.mark_invalid(entry, exc)

    def mark_success(self, entry: _KeyEntry, model: str) -> None:
        self._pool.mark_success(entry, model)

    def entries(self) -> list[_KeyEntry]:
        return self._pool.entries()

    def status(self) -> str:
        return self._pool.status(active_index=self._active_index, lane=self.name)


class KeyPool:
    """
    Sticky active-key pool of Gemini API keys.

    The pool owns the keys and their health. *Which* key a given consumer
    is currently on belongs to a :class:`KeyLane` — see :meth:`lane`.
    Every lane sees the same cooldowns and the same dead keys.

    Thread-safe via ``asyncio.Lock`` per lane (the hot paths run on a
    single async loop, but the locks keep cursor changes atomic under any
    concurrent call pattern).
    """

    def __init__(self, keys: list[str]):
        if not keys:
            sys.exit(
                "No Gemini API keys found.\n"
                "Set GEMINI_API_KEY_1, GEMINI_API_KEY_2, … (or GEMINI_API_KEY) in .env\n"
            )
        self._entries = [_KeyEntry(k, i) for i, k in enumerate(keys)]
        self._lanes: dict[str, KeyLane] = {}
        # Callers that never ask for a lane get the historical behaviour:
        # start at key 1, walk forward.
        self._default_lane = self.lane("default", FORWARD)

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> list[_KeyEntry]:
        return list(self._entries)

    def lane(
        self, name: str, direction: int = FORWARD, start: int | None = None
    ) -> KeyLane:
        """Get (or create) a named cursor over this pool.

        Named rather than positional so two callers asking for
        ``"conversation"`` share one cursor — the point is to separate
        *roles*, not every call site.
        """
        existing = self._lanes.get(name)
        if existing is not None:
            return existing
        created = KeyLane(self, name, direction=direction, start=start)
        self._lanes[name] = created
        _log(
            f"  ⇢ key lane '{name}' starts at key[{created.active_index}] "
            f"going {'forward' if direction >= 0 else 'backward'}"
        )
        return created

    def lanes(self) -> dict[str, KeyLane]:
        return dict(self._lanes)

    # ── Sticky active-key selection (default lane) ───────────────────────

    async def acquire_active(self) -> _KeyEntry:
        return await self._default_lane.acquire_active()

    async def advance_active(self) -> None:
        await self._default_lane.advance_active()

    async def next(self) -> _KeyEntry | None:
        return await self._default_lane.next()

    # ── Key state mutations ──────────────────────────────────────────────

    def mark_quota(self, entry: _KeyEntry, exc: Exception) -> None:
        """Cool a key down based on error type.

        Daily-quota exhaustion (limit: 0, GenerateRequestsPerDay) is
        treated as long-lived — the key is parked for
        ``DAILY_QUOTA_COOLDOWN_SECS`` (default 1h, configurable), on the
        assumption that daily quotas only reset on a daily boundary.

        Temporary 429s honour the server's ``Retry-After`` when present
        and otherwise use ``COOLDOWN_SECS`` (default 60s).
        """
        if _is_quota_exhausted(exc):
            entry.cool_down(
                secs=DAILY_QUOTA_COOLDOWN_SECS,
                reason="daily quota exhausted",
            )
            return

        server_wait = _parse_retry_after(exc)
        if server_wait is not None:
            entry.cool_down(secs=server_wait, reason="Retry-After")
        else:
            entry.cool_down(secs=COOLDOWN_SECS, reason="rate limited")

    def mark_not_found(self, entry: _KeyEntry, model: str) -> None:
        """Record that ``entry`` cannot serve ``model``. Does NOT cool
        the key — 404 is a capability error, not a capacity error.
        Callers should ask the next model from the same key."""
        entry.mark_unsupported(model)

    def mark_invalid(self, entry: _KeyEntry, exc: Exception) -> None:
        """Mark the key as permanently dead (auth/configuration error)."""
        entry.mark_dead(reason=f"invalid: {exc!r}")

    def mark_success(self, entry: _KeyEntry, model: str) -> None:
        entry.mark_supported(model)

    def status(self, active_index: int | None = None, lane: str = "") -> str:
        """Render key health.

        With no arguments, every lane's position is marked, which is what
        you want when diagnosing the pool as a whole. Passing
        ``active_index`` marks just that one — used by
        :meth:`KeyLane.status`.
        """
        if active_index is None:
            positions = {
                l.active_index: name for name, l in self._lanes.items()
                if name != "default" or len(self._lanes) == 1
            }
        else:
            positions = {active_index: lane or "active"}

        lines = []
        for i, e in enumerate(self._entries):
            tag = ""
            if i in positions and e.is_available:
                tag = f" ← {positions[i]}"
            if e.dead:
                lines.append(f"  key[{e.index}] 💀 dead{tag}")
            elif e.is_available:
                lines.append(f"  key[{e.index}] ✅ available{tag}")
            else:
                lines.append(
                    f"  key[{e.index}] 🔴 cooling {e.secs_until_ready():.0f}s{tag}"
                )
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
    msg = str(exc)
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", msg)
    if match:
        return float(match.group(1))
    match = re.search(r"[Rr]etry in\s+(\d+(?:\.\d+)?)s", msg)
    return float(match.group(1)) if match else None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, genai_errors.APIError):
        return exc.code in (429, 503)
    msg = str(exc).lower()
    return any(t in msg for t in ("429", "503", "resource_exhausted", "unavailable"))


def _is_quota_exhausted(exc: Exception) -> bool:
    """Return True for daily-quota exhaustion. These are the errors that
    should remove a key from rotation for an extended period; ordinary
    429s and ``resource_exhausted`` per-minute limits are NOT daily
    exhaustion.

    The free tier reports its daily request cap as
    ``generate_content_free_tier_requests, limit: <n>`` (e.g. 500 for
    gemini-3.5-flash-lite) — no ``limit: 0``, no ``PerDay`` suffix — so
    that metric name is recognised explicitly, or an account whose day
    is gone would be re-tried every minute instead of parked for the
    day."""
    msg = str(exc)
    return (
        "limit: 0" in msg
        or "GenerateRequestsPerDay" in msg
        or "generate_content_free_tier_requests" in msg
    )


def _is_not_found(exc: Exception) -> bool:
    if isinstance(exc, genai_errors.APIError):
        return exc.code == 404
    msg = str(exc).lower()
    return any(t in msg for t in ("404", "not_found", "not found"))


def _is_invalid_key(exc: Exception) -> bool:
    if isinstance(exc, genai_errors.APIError):
        return exc.code in (401, 403)
    msg = str(exc).lower()
    return any(t in msg for t in ("401", "403", "api_key_invalid", "permission_denied"))


def _split_models(value: str) -> list[str]:
    return [m.strip() for m in value.split(",") if m.strip()]


# ── Per-key call primitives ──────────────────────────────────────────────────


async def _run_blocking(
    entry: _KeyEntry,
    model: str,
    timeout: float,
    call_kwargs: dict[str, Any],
) -> str:
    """One non-streaming call against one key, wrapped in a timeout."""
    resp = await asyncio.wait_for(
        asyncio.to_thread(
            entry.client.models.generate_content,
            model=model,
            **call_kwargs,
        ),
        timeout=timeout,
    )
    return getattr(resp, "text", "") or ""


async def _run_stream(
    entry: _KeyEntry,
    model: str,
    timeout: float,
    call_kwargs: dict[str, Any],
) -> AsyncIterator[str]:
    """Return an async iterator that yields text deltas from a single
    streaming call against one key. The first-chunk wait honours
    ``timeout``; subsequent chunks stream as fast as the model emits
    them.

    If the stream errors out before the caller has consumed it, the
    exception is surfaced on the next ``__anext__()`` call — that's how
    :func:`_try_with_active_key` learns to rotate."""
    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None | Exception] = asyncio.Queue()

    def _producer() -> None:
        try:
            response = entry.client.models.generate_content_stream(
                model=model,
                **call_kwargs,
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

    async def _gen() -> AsyncIterator[str]:
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

    return _gen()


# ── Sticky-key + model-fallback driver ───────────────────────────────────────


async def _try_with_active_key(
    pool: KeyPool | KeyLane,
    models: list[str],
    call_kwargs: dict[str, Any],
    timeout: float,
    *,
    is_stream: bool,
) -> Any:
    """Try the active key against the model list; if the active key
    becomes unusable, fall over to the next key and repeat.

    Returns either a ``str`` (non-streaming) or an ``AsyncIterator[str]``
    (streaming) — the caller already knows which based on ``is_stream``.

    Behaviour:

      * Pick the current active key (sticky — same key is used for
        subsequent calls as long as it stays healthy).
      * Try each model in priority order. A 404/NOT_FOUND on a model
        records the (key, model) miss in the per-key capability cache
        and continues to the next model on the **same** key.
      * A 429/503/auth/network error on a (key, model) attempt treats
        the key as exhausted for this call and rotates to the next key.
      * Streaming is validated on the first chunk: the driver waits for
        it inside the rotation guard, so a first-chunk error rotates
        instead of escaping to the caller, then replays that chunk.
      * Success returns the result.
      * Daily-quota exhaustion removes the key from rotation for
        ``DAILY_QUOTA_COOLDOWN_SECS``.
      * Invalid key permanently marks the key dead.
      * Raises ``RuntimeError`` if every key + every model is exhausted.
    """
    n = len(pool)
    for _ in range(n):
        # ── Step 1: pick the current active key (sticky) ───────────────
        try:
            entry = await pool.acquire_active()
        except RuntimeError:
            # All keys dead.
            raise

        # ── Step 2: try every model on this key ─────────────────────────
        key_failed = False
        saw_any_attempt = False
        for model in models:
            if not entry.supports(model):
                # Cached 404 — skip silently and try the next model on
                # the same key. The capability cache ensures we never
                # burn a request on a known-bad (key, model) pair.
                continue
            saw_any_attempt = True
            try:
                if is_stream:
                    gen = await _run_stream(entry, model, timeout, call_kwargs)
                    # Pull the first chunk *inside* the try so a key-level
                    # failure (quota, auth, timeout) surfaces here and
                    # rotates to the next key, exactly like the
                    # non-streaming path below. Before this the generator
                    # was returned with no request in flight, so the first
                    # chunk's error escaped to the caller with the key
                    # never cooled or advanced — a streaming lane stayed
                    # pinned to one exhausted key forever. The first chunk
                    # is replayed by ``_replay`` so the caller's stream
                    # looks exactly as it would have.
                    try:
                        first = await anext(gen)
                    except StopAsyncIteration:
                        # An empty stream is not a key failure — the
                        # request itself succeeded. The old code handed
                        # the caller a stream that simply had no chunks;
                        # keep that, instead of surfacing an exception.
                        first = None
                    pool.mark_success(entry, model)

                    async def _replay() -> AsyncIterator[str]:
                        if first is not None:
                            yield first
                        async for chunk in gen:
                            yield chunk

                    return _replay()
                text = await _run_blocking(entry, model, timeout, call_kwargs)
                pool.mark_success(entry, model)
                return text

            except asyncio.TimeoutError:
                # A single timeout doesn't kill a key — record a short
                # cooldown so a hung network doesn't immediately
                # re-enter rotation, then advance to the next key.
                _log(f"[{model}] key[{entry.index}] timeout after {timeout}s")
                entry.cool_down(secs=COOLDOWN_SECS, reason="timeout")
                key_failed = True
                break

            except Exception as exc:
                if _is_invalid_key(exc):
                    pool.mark_invalid(entry, exc)
                    _log(
                        f"[{model}] key[{entry.index}] invalid — removed from rotation"
                    )
                    key_failed = True
                    break

                if _is_not_found(exc):
                    # 404: this key doesn't support this model. Record
                    # the miss and try the next model on the SAME key.
                    pool.mark_not_found(entry, model)
                    continue

                if _is_retryable(exc):
                    # 429 / 503 / RESOURCE_EXHAUSTED — treat as key-level
                    # failure for this call. The key is cooled
                    # appropriately and the next key takes over.
                    pool.mark_quota(entry, exc)
                    _log(
                        f"[{model}] key[{entry.index}] quota — "
                        f"cooling + rotating to next key"
                    )
                    key_failed = True
                    break

                # Non-retryable, non-model-specific, non-auth error:
                # surface to the caller. A real bug (bad prompt, schema
                # mismatch) must not be hidden by a fallback model.
                raise

        # ── Step 3: did this key have *anything* to offer? ──────────────
        # If we never even attempted a model on this key (every model
        # was cached as unsupported), there's no point trying it again
        # — treat the key as exhausted and move on. Otherwise the
        # inner loop ended because of a real failure, which we already
        # handled above.
        if not saw_any_attempt and not key_failed:
            _log(
                f"  key[{entry.index}] no usable models in cache — rotating to next key"
            )
            key_failed = True

        if not key_failed:
            # Inner loop completed without raising — every model was
            # skipped via the cache. Should be unreachable given the
            # guard above, but stay defensive.
            key_failed = True

        # ── Step 4: rotate to the next live key ─────────────────────────
        await pool.advance_active()

    raise RuntimeError("All models + keys exhausted — no response available")


# ── Public multi-model entry points ──────────────────────────────────────────
#
# The ``hedged_*`` names are kept for backward compatibility — they
# previously implemented a hedge race; now they route through the
# sticky driver. The ``hedge_width`` parameter is accepted but ignored
# (sequential now).


async def hedged_generate(
    pool: KeyPool | KeyLane,
    models: list[str],
    prompt: str,
    hedge_width: int,  # pyright: ignore[reportArgumentType]
    timeout: float,
) -> str:
    """Sequential sticky-key + model-fallback for a plain prompt.

    The ``hedge_width`` parameter is accepted for backward compatibility
    but no longer enables concurrent hedging: KANCHA's benchmarks
    showed hedged requests wasted tokens by duplicating generation. The
    new implementation is strictly sequential — the active key is used
    for every request, and only switches when it becomes unusable.
    """
    _ = hedge_width
    return await _try_with_active_key(
        pool=pool,
        models=models,
        call_kwargs={"contents": prompt},
        timeout=timeout,
        is_stream=False,
    )


async def hedged_generate_conv(
    pool: KeyPool | KeyLane,
    models: list[str],
    contents: list,
    config: Any,
    hedge_width: int,  # pyright: ignore[reportArgumentType]
    timeout: float,
) -> str:
    """Sequential sticky-key + model-fallback for a structured conversation."""
    _ = hedge_width
    kwargs: dict[str, Any] = {"contents": contents}
    if config is not None:
        kwargs["config"] = config
    return await _try_with_active_key(
        pool=pool,
        models=models,
        call_kwargs=kwargs,
        timeout=timeout,
        is_stream=False,
    )


async def hedged_stream(
    pool: KeyPool | KeyLane,
    models: list[str],
    prompt: str,
    hedge_width: int,  # pyright: ignore[reportArgumentType]
    timeout: float,
) -> None:
    """Stream from the first successful (active key, model) pair for a
    plain prompt.

    The previous hedge-race version printed streamed output directly
    here; that responsibility is preserved so callers using the bare
    ``hedged_stream`` path see the same on-stdout behaviour.
    """
    _ = hedge_width
    start = time.perf_counter()
    first = True
    gen = await _try_with_active_key(
        pool=pool,
        models=models,
        call_kwargs={"contents": prompt},
        timeout=timeout,
        is_stream=True,
    )
    async for chunk in gen:
        if first:
            _log(f"[stream] TTFT {time.perf_counter() - start:.2f}s")
            first = False
        print(chunk, end="", flush=True)
    print(flush=True)
    _log(f"[stream] done in {time.perf_counter() - start:.2f}s")


async def hedged_stream_conv(
    pool: KeyPool | KeyLane,
    models: list[str],
    contents: list,
    config: Any,
    hedge_width: int,  # pyright: ignore[reportArgumentType]
    timeout: float,
) -> AsyncIterator[str]:
    """Stream from the first successful (active key, model) pair for a
    structured conversation. Yields raw text deltas — callers handle
    buffering and JSON extraction."""
    _ = hedge_width
    kwargs: dict[str, Any] = {"contents": contents}
    if config is not None:
        kwargs["config"] = config
    gen = await _try_with_active_key(
        pool=pool,
        models=models,
        call_kwargs=kwargs,
        timeout=timeout,
        is_stream=True,
    )
    async for chunk in gen:
        yield chunk


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
