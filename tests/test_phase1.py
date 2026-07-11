"""
tests/test_phase1.py — Phase 1 integration tests.

Verifies the EventBus (core/bus.py), the events module (core/events.py),
and a minimal ConversationContext per the plan.md Phase 1 spec.

Run with::

    uv run python tests/test_phase1.py

Exits 0 on full pass, 1 on any failure. Output is plain text with
[PASS] / [FAIL] markers — no pytest dependency.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from core.bus import EventBus
from core.events import (
    BaseEvent,
    IntentIdentified,
    Intent,
    SystemError,
    TextInputReceived,
    TranscriptReady,
)


# ── Minimal ConversationContext (mirrors plan.md spec) ────────────────────
# Defined locally because core/context.py is still empty; will be replaced
# with the real import once that module is implemented.

@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant", "system"]
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ConversationContext:
    def __init__(self, max_turns: int = 10) -> None:
        self._buffer: deque[Turn] = deque(maxlen=max_turns)
        self._lock = asyncio.Lock()
        self._max_turns = max_turns

    async def add_turn(self, role: str, content: str) -> None:
        async with self._lock:
            if not content or not content.strip():
                return
            self._buffer.append(Turn(role=role, content=content))

    async def as_messages(self) -> list[dict[str, str]]:
        async with self._lock:
            return [t.to_dict() for t in self._buffer]

    async def clear(self) -> None:
        async with self._lock:
            self._buffer.clear()

    async def token_estimate(self) -> int:
        async with self._lock:
            return int(sum(len(t.content) for t in self._buffer) / 4.0)

    @property
    def max_turns(self) -> int:
        return self._max_turns

    def __repr__(self) -> str:
        return f"<ConversationContext turns={len(self._buffer)}/{self._max_turns}>"


# ── Tiny test harness ────────────────────────────────────────────────────

_PASS = "\033[32m[PASS]\033[0m"
_FAIL = "\033[31m[FAIL]\033[0m"
_results: list[tuple[str, bool, str]] = []


def _record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"{( _PASS if ok else _FAIL)} {name}" + (f"  — {detail}" if detail and not ok else ""))


async def _expect(name: str, coro) -> None:
    try:
        await coro
        _record(name, True)
    except AssertionError as e:
        _record(name, False, str(e) or "assertion failed")
    except Exception as e:  # noqa: BLE001
        _record(name, False, f"{type(e).__name__}: {e}")


# ── Tests ────────────────────────────────────────────────────────────────

async def test_basic_pubsub() -> None:
    bus = EventBus()
    received: list[str] = []

    async def handler(event: TextInputReceived) -> None:
        received.append(event.text)

    bus.subscribe(TextInputReceived, handler)
    bus.emit(TextInputReceived(text="hello"))
    await asyncio.sleep(0.05)
    await bus.close()
    assert received == ["hello"], f"expected ['hello'], got {received}"


async def test_multiple_handlers_same_event() -> None:
    bus = EventBus()
    a, b = [], []

    async def h1(e: TextInputReceived) -> None:
        a.append(e.text)

    async def h2(e: TextInputReceived) -> None:
        b.append(e.text)

    bus.subscribe(TextInputReceived, h1)
    bus.subscribe(TextInputReceived, h2)
    bus.emit(TextInputReceived(text="x"))
    await asyncio.sleep(0.05)
    await bus.close()
    assert a == ["x"] and b == ["x"], f"a={a} b={b}"


async def test_emit_with_no_subscribers_does_not_raise() -> None:
    bus = EventBus()
    bus.emit(TextInputReceived(text="nobody home"))  # must not raise
    await asyncio.sleep(0.01)
    await bus.close()


async def test_handler_crash_isolation() -> None:
    bus = EventBus()
    survivor: list[str] = []

    async def bad(e: TextInputReceived) -> None:
        raise RuntimeError("boom")

    async def good(e: TextInputReceived) -> None:
        survivor.append(e.text)

    bus.subscribe(TextInputReceived, bad)
    bus.subscribe(TextInputReceived, good)
    bus.emit(TextInputReceived(text="survive"))
    await asyncio.sleep(0.05)
    await bus.close()
    assert survivor == ["survive"], f"good handler was killed: {survivor}"


async def test_context_add_and_retrieve() -> None:
    ctx = ConversationContext(max_turns=5)
    await ctx.add_turn("user", "hi")
    await ctx.add_turn("assistant", "hello!")
    msgs = await ctx.as_messages()
    assert msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
    ], f"got {msgs}"


async def test_context_rolling_eviction() -> None:
    ctx = ConversationContext(max_turns=4)
    for i in range(6):
        await ctx.add_turn("user", f"msg-{i}")
    msgs = await ctx.as_messages()
    contents = [m["content"] for m in msgs]
    assert contents == ["msg-2", "msg-3", "msg-4", "msg-5"], (
        f"expected oldest evicted, got {contents}"
    )


async def test_events_are_frozen() -> None:
    e = TextInputReceived(text="lock me")
    try:
        e.text = "mutate"  # type: ignore[misc]
    except (Exception,) as exc:  # FrozenInstanceError or AttributeError
        assert type(exc).__name__ in {"FrozenInstanceError", "AttributeError"}, (
            f"unexpected exception type: {type(exc).__name__}"
        )
    else:
        raise AssertionError("event was mutable — should be frozen")


async def test_events_have_unique_ids() -> None:
    a = TextInputReceived(text="x")
    b = TextInputReceived(text="x")
    assert a.event_id != b.event_id, "event_ids must be unique"
    assert isinstance(a.timestamp, datetime), "timestamp should be a datetime"


# ── Bus-specific tests ───────────────────────────────────────────────────

async def test_emit_threadsafe_from_worker_thread() -> None:
    bus = EventBus()
    delivered: list[str] = []
    loop_holder: dict[str, Any] = {}
    # Coordinate so the worker thread doesn't publish before the asyncio
    # loop has been entered (otherwise ``get_event_loop`` would create a
    # new loop in the worker, defeating the test).
    go = threading.Event()

    async def handler(e: TextInputReceived) -> None:
        # Capture the running loop to prove the handler ran on the main loop.
        loop_holder["loop"] = asyncio.get_running_loop()
        delivered.append(e.text)

    bus.subscribe(TextInputReceived, handler)

    def worker() -> None:
        # Block until the asyncio side is definitely up.
        assert go.wait(timeout=2.0), "main never signalled go"
        bus.emit_threadsafe(TextInputReceived(text="from-thread"))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    # Yield to the loop so it's running, then release the worker.
    await asyncio.sleep(0)
    go.set()
    # Give the loop time to drain the call_soon_threadsafe callback.
    await asyncio.sleep(0.2)
    t.join(timeout=1.0)
    await bus.close()
    assert delivered == ["from-thread"], f"thread event not delivered: {delivered}"
    assert "loop" in loop_holder, "handler never ran on the loop"


async def test_emit_and_wait_blocks_until_handlers_done() -> None:
    bus = EventBus()
    order: list[str] = []

    async def slow(e: TextInputReceived) -> None:
        await asyncio.sleep(0.05)
        order.append("slow")

    async def fast(e: TextInputReceived) -> None:
        order.append("fast")

    bus.subscribe(TextInputReceived, slow)
    bus.subscribe(TextInputReceived, fast)
    bus.emit(TextInputReceived(text="go"))
    # emit (fire-and-forget) returns immediately — order is still empty.
    assert order == [], f"emit should not block, but order={order}"
    await bus.emit_and_wait(TextInputReceived(text="wait"))
    assert "fast" in order and "slow" in order, f"missing handlers: {order}"
    await bus.close()


async def test_drain_waits_for_pending_tasks() -> None:
    bus = EventBus()
    counter = {"n": 0}

    async def slow(e: TextInputReceived) -> None:
        await asyncio.sleep(0.1)
        counter["n"] += 1

    bus.subscribe(TextInputReceived, slow)
    for i in range(5):
        bus.emit(TextInputReceived(text=f"m{i}"))
    assert len(bus._tasks) == 5, f"expected 5 inflight, got {len(bus._tasks)}"
    await bus.drain()
    assert counter["n"] == 5, f"expected 5 done, got {counter['n']}"
    assert len(bus._tasks) == 0, "tasks set should be empty after drain"
    await bus.close()


async def test_subscribe_rejects_sync_handler() -> None:
    bus = EventBus()

    def not_async(e: TextInputReceived) -> None:
        pass

    try:
        bus.subscribe(TextInputReceived, not_async)  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("subscribe should have raised TypeError for sync handler")
    await bus.close()


# ── Runner ───────────────────────────────────────────────────────────────

async def main() -> None:
    print("=== Phase 1 integration tests ===\n")
    tests = [
        ("EventBus basic pub/sub",              test_basic_pubsub),
        ("EventBus two handlers on one event",  test_multiple_handlers_same_event),
        ("EventBus emit with no subscribers",   test_emit_with_no_subscribers_does_not_raise),
        ("EventBus crash isolation",            test_handler_crash_isolation),
        ("ConversationContext add/retrieve",    test_context_add_and_retrieve),
        ("ConversationContext rolling eviction", test_context_rolling_eviction),
        ("Events are frozen",                   test_events_are_frozen),
        ("Events have unique IDs",              test_events_have_unique_ids),
        ("emit_threadsafe from worker thread",  test_emit_threadsafe_from_worker_thread),
        ("emit_and_wait blocks until done",     test_emit_and_wait_blocks_until_handlers_done),
        ("drain waits for pending tasks",       test_drain_waits_for_pending_tasks),
        ("subscribe rejects sync handler",      test_subscribe_rejects_sync_handler),
    ]
    for name, fn in tests:
        await _expect(name, fn())

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print()
    if passed == total:
        print(f"\033[32m[ALL PASS: {passed}/{total}]\033[0m")
        sys.exit(0)
    else:
        print(f"\033[31m[FAILURES: {total - passed}/{total}]\033[0m")
        for name, ok, detail in _results:
            if not ok:
                print(f"  - {name}: {detail}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
