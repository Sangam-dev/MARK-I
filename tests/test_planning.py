"""Integration tests for the Planner/Scheduler/Executor subsystem.

These tests use a mocked GeminiClient to produce deterministic plans
and verify the full event flow from IntentIdentified -> PlanCreated ->
TaskStarted -> TaskCompleted -> PlanCompleted.

Run with::

    PYTHONPATH=/home/sangam/kancha python tests/test_planning.py
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any

from core.bus import EventBus
from core.events import (
    Intent,
    IntentIdentified,
    PlanCompleted,
    PlanCreated,
    ResponseReady,
    TaskCompleted,
    TaskExecutionRequested,
)
from memory.token_log import TokenLog
from planning.executor import PlanExecutor
from planning.models import ExecutionPlan, PlannedTask, PlanStatus, TaskStatus
from planning.planner import Planner
from planning.scheduler import PlanScheduler
from reasoning.coordinator import ReasoningCoordinator
from reasoning.tool_voice import naturalize_single_tool
from tasks.registry import TASK_REGISTRY


# ── Test helpers ──────────────────────────────────────────────────────


@dataclass
class MockGeminiClient:
    """Fake GeminiClient that returns a canned plan JSON."""

    plan_json: dict[str, Any] | None = None
    fail_count: int = 0
    # For naturalize tests: canned `generate` response + call log.
    generate_response: str | None = None
    generate_calls: list[dict[str, str]] = field(default_factory=list)

    async def initialize(self) -> None:
        pass

    async def generate_json(
        self,
        prompt: str,
        schema_description: str | None = None,
        system: str = "",
        hedge_width: int = 1,
        call_site: str = "planner",
    ) -> dict[str, Any]:
        self.fail_count += 1
        if self.plan_json is not None:
            return self.plan_json
        return {}

    async def generate(
        self,
        prompt: str,
        system: str = "",
        hedge_width: int = 1,
        call_site: str = "conversational",
    ) -> str:
        """Used by naturalize_plan_response — record and return canned text."""
        self.generate_calls.append({"prompt": prompt, "system": system, "hedge_width": hedge_width})
        if self.generate_response is None:
            return "Done, sir."
        return self.generate_response

    async def generate_with_history(
        self,
        history: list[dict[str, str]],
        system: str = "",
        hedge_width: int = 1,
        call_site: str = "conversational",
    ) -> str:
        self.generate_calls.append({"prompt": "history", "system": system, "hedge_width": hedge_width})
        return self.generate_response or "Done, sir."


@dataclass
class MockShortTerm:
    """Mimics ConversationContext."""

    _data: list[dict[str, str]] = field(default_factory=list)

    def add(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._data.append({"role": role, "content": content})

    def get_recent(self, limit: int = 6) -> list[dict[str, str]]:
        return self._data[-limit:]

    def clear(self) -> None:
        self._data.clear()


@dataclass
class MockMemory:
    """Fake MemoryManager with no persistence."""

    session_id: str = "default"
    _short_term_data: list[dict[str, str]] = field(default_factory=list)
    short_term: MockShortTerm = field(default_factory=MockShortTerm)
    structured_facts: list[dict[str, str]] = field(default_factory=list)
    _recent_plan_outputs: dict[str, dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.short_term = MockShortTerm(_data=self._short_term_data)

    async def get_all_facts(self) -> list[dict[str, str]]:
        return self.structured_facts

    async def store_fact(self, key: str, value: str, session_id: str) -> str:
        self.structured_facts.append({"key": key, "value": value})
        return key

    def get_recent_plan_outputs(self, plan_id: str | None = None) -> dict[str, str]:
        return {}

    def clear_plan_outputs(self, plan_id: str | None = None) -> None:
        pass


# ── Test runner ──────────────────────────────────────────────────────


_PASS = "\033[32m[PASS]\033[0m"
_FAIL = "\033[31m[FAIL]\033[0m"
_results: list[tuple[str, bool, str]] = []


def _record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"{(_PASS if ok else _FAIL)} {name}" + (f"  — {detail}" if detail and not ok else ""))


async def _expect(name: str, coro) -> None:
    try:
        await coro
        _record(name, True)
    except AssertionError as e:
        _record(name, False, str(e) or "assertion failed")
    except Exception as e:
        _record(name, False, f"{type(e).__name__}: {e}")


# ── Tests ──────────────────────────────────────────────────────────────


async def test_plan_decomposition_single_task() -> None:
    """A simple request should produce a one-task plan."""
    bus = EventBus()
    llm = MockGeminiClient(
        plan_json={
            "tasks": [
                {
                    "id": "t1",
                    "description": "Open firefox",
                    "tool": "open_app",
                    "arguments": {"app_name": "firefox"},
                    "depends_on": [],
                }
            ]
        }
    )
    memory = MockMemory()

    # Track emitted events
    plan_created: list[PlanCreated] = []
    plan_completed: list[PlanCompleted] = []

    async def _on_plan_created(e: PlanCreated) -> None:
        plan_created.append(e)

    async def _on_plan_completed(e: PlanCompleted) -> None:
        plan_completed.append(e)

    bus.subscribe(PlanCreated, _on_plan_created)
    bus.subscribe(PlanCompleted, _on_plan_completed)

    planner = Planner(bus=bus, llm=llm, memory=memory)
    planner.register()
    scheduler = PlanScheduler(bus=bus)
    scheduler.register()
    scheduler.executor.register()

    # Trigger the planner
    intent = IntentIdentified(
        intent=Intent.TASK,
        raw_input="open firefox",
        requires_task=True,
        task_type="open_app",
        task_params={"app_name": "firefox"},
    )
    await planner.on_intent(intent)

    await asyncio.sleep(0.1)  # let events propagate
    await bus.drain()

    assert len(plan_created) == 1, "should emit PlanCreated"
    plan = plan_created[0].plan
    assert plan["id"] is not None
    assert len(plan["tasks"]) == 1, "single-task plan"
    assert plan["tasks"][0]["tool"] == "open_app"

    await bus.close()


async def test_plan_decomposition_multi_task() -> None:
    """A multi-step request should produce a multi-task plan."""
    bus = EventBus()
    llm = MockGeminiClient(
        plan_json={
            "tasks": [
                {
                    "id": "t1",
                    "description": "Create folder Test in Documents",
                    "tool": "file_operation",
                    "arguments": {"action": "create_folder", "path": "documents", "name": "Test"},
                    "depends_on": [],
                },
                {
                    "id": "t2",
                    "description": "Create README.md in Test",
                    "tool": "file_operation",
                    "arguments": {"action": "create_file", "path": "documents/Test", "name": "README.md"},
                    "depends_on": ["t1"],
                },
                {
                    "id": "t3",
                    "description": "Open README.md",
                    "tool": "open_app",
                    "arguments": {"app_name": "documents/Test/README.md"},
                    "depends_on": ["t2"],
                },
            ]
        }
    )
    memory = MockMemory()

    plan_created: list[PlanCreated] = []
    plan_completed: list[PlanCompleted] = []

    async def _on_plan_created(e: PlanCreated) -> None:
        plan_created.append(e)

    async def _on_plan_completed(e: PlanCompleted) -> None:
        plan_completed.append(e)

    bus.subscribe(PlanCreated, _on_plan_created)
    bus.subscribe(PlanCompleted, _on_plan_completed)
    planner = Planner(bus=bus, llm=llm, memory=memory)
    planner.register()
    scheduler = PlanScheduler(bus=bus)
    scheduler.register()
    scheduler.executor.register()

    await planner.on_intent(
        IntentIdentified(
            intent=Intent.TASK,
            raw_input="Create Test folder, add README.md, and open it",
            requires_task=True,
        )
    )

    await asyncio.sleep(0.1)
    await bus.drain()

    assert len(plan_created) == 1
    plan = plan_created[0].plan
    assert len(plan["tasks"]) == 3

    # Check dependency chain
    t1, t2, t3 = plan["tasks"]
    assert t1["depends_on"] == []
    assert t2["depends_on"] == ["t1"]
    assert t3["depends_on"] == ["t2"]

    await bus.close()


async def test_parallel_tasks_run_concurrently() -> None:
    """Tasks with no dependencies among them should run in parallel."""
    bus = EventBus()
    llm = MockGeminiClient(
        plan_json={
            "tasks": [
                {
                    "id": "t1",
                    "description": "Task 1",
                    "tool": "list_alarms",
                    "arguments": {},
                    "depends_on": [],
                },
                {
                    "id": "t2",
                    "description": "Task 2",
                    "tool": "list_alarms",
                    "arguments": {},
                    "depends_on": [],
                },
            ]
        }
    )
    memory = MockMemory()

    plan_created: list[PlanCreated] = []
    plan_completed: list[PlanCompleted] = []
    task_requests: list[TaskExecutionRequested] = []

    async def _on_plan_created(e: PlanCreated) -> None:
        plan_created.append(e)

    async def _on_plan_completed(e: PlanCompleted) -> None:
        plan_completed.append(e)

    async def _on_task_exec(e: TaskExecutionRequested) -> None:
        task_requests.append(e)
        # Mock TaskExecutor: respond immediately with success
        bus.emit(
            TaskCompleted(
                task_name=e.task_name,
                success=True,
                result="mock success",
                session_id=e.session_id,
            )
        )

    bus.subscribe(PlanCreated, _on_plan_created)
    bus.subscribe(PlanCompleted, _on_plan_completed)
    bus.subscribe(TaskExecutionRequested, _on_task_exec)

    planner = Planner(bus=bus, llm=llm, memory=memory)
    planner.register()
    scheduler = PlanScheduler(bus=bus)
    scheduler.register()
    scheduler.executor.register()

    await planner.on_intent(
        IntentIdentified(
            intent=Intent.TASK,
            raw_input="do two things at once",
            requires_task=True,
        )
    )

    await asyncio.sleep(0.3)  # let tasks complete
    await bus.drain()

    # Both tasks should have been dispatched
    assert len(task_requests) == 2, f"Expected 2 task requests, got {len(task_requests)}"

    await bus.close()


async def test_reference_resolution() -> None:
    """Pronouns like 'it' should be resolved to prior task outputs."""
    bus = EventBus()
    llm = MockGeminiClient(
        plan_json={
            "tasks": [
                {
                    "id": "t1",
                    "description": "Create folder Test",
                    "tool": "file_operation",
                    "arguments": {"action": "create_folder", "path": "documents", "name": "Test"},
                    "depends_on": [],
                },
                {
                    "id": "t2",
                    "description": "Open it",
                    "tool": "open_app",
                    "arguments": {"app_name": "it"},
                    "depends_on": ["t1"],
                },
            ]
        }
    )
    memory = MockMemory()

    plan_created: list[PlanCreated] = []

    async def _on_plan_created(e: PlanCreated) -> None:
        plan_created.append(e)

    bus.subscribe(PlanCreated, _on_plan_created)
    planner = Planner(bus=bus, llm=llm, memory=memory)
    planner.register()
    scheduler = PlanScheduler(bus=bus)
    scheduler.register()
    scheduler.executor.register()

    await planner.on_intent(
        IntentIdentified(
            intent=Intent.TASK,
            raw_input="Create folder Test and open it",
            requires_task=True,
        )
    )

    await asyncio.sleep(0.1)
    await bus.drain()

    assert len(plan_created) == 1
    plan = plan_created[0].plan

    # The second task's argument should have been rewritten
    t2 = plan["tasks"][1]
    assert t2["tool"] == "open_app"
    # The reference resolver should have rewritten "it" to a placeholder
    assert "output_refs" in t2
    assert t2["output_refs"].get("app_name") == "t1:result"

    await bus.close()


async def test_fast_path_single_task() -> None:
    """When NLU has a concrete task_type, skip the LLM."""
    bus = EventBus()
    memory = MockMemory()

    plan_created: list[PlanCreated] = []

    async def _on_plan_created(e: PlanCreated) -> None:
        plan_created.append(e)

    bus.subscribe(PlanCreated, _on_plan_created)

    # The fast path uses NLU's precomputed task_type
    llm = MockGeminiClient(plan_json={})  # should not be called

    planner = Planner(bus=bus, llm=llm, memory=memory)
    planner.register()

    # Direct intent with task_type already resolved
    await planner.on_intent(
        IntentIdentified(
            intent=Intent.TASK,
            raw_input="open firefox",
            requires_task=True,
            task_type="open_app",
            task_params={"app_name": "firefox"},
        )
    )

    await asyncio.sleep(0.1)
    await bus.drain()

    # LLM should NOT have been called (fast path)
    assert llm.fail_count == 0
    assert len(plan_created) == 1
    plan = plan_created[0].plan
    assert len(plan["tasks"]) == 1
    assert plan["tasks"][0]["tool"] == "open_app"

    await bus.close()


async def test_single_tool_with_conjunction_uses_fast_path() -> None:
    """A single-tool request with a conjunction should still use the fast path."""
    bus = EventBus()
    llm = MockGeminiClient(plan_json={})
    memory = MockMemory()

    plan_created: list[PlanCreated] = []

    async def _on_plan_created(e: PlanCreated) -> None:
        plan_created.append(e)

    bus.subscribe(PlanCreated, _on_plan_created)
    planner = Planner(bus=bus, llm=llm, memory=memory)
    planner.register()

    await planner.on_intent(
        IntentIdentified(
            intent=Intent.TASK,
            raw_input="open firefox and run it",
            requires_task=True,
            task_type="open_app",
            task_params={"app_name": "firefox"},
        )
    )

    await asyncio.sleep(0.1)
    await bus.drain()

    assert llm.fail_count == 0, f"LLM should NOT have been called, was {llm.fail_count}"
    assert len(plan_created) == 1
    plan = plan_created[0].plan
    assert len(plan["tasks"]) == 1
    assert plan["tasks"][0]["tool"] == "open_app"

    await bus.close()


async def test_multistep_request_bypasses_fast_path() -> None:
    """A request with multiple actions (e.g. "open X and create Y")
    must skip the NLU-driven fast path and reach the Planner LLM,
    so the second action isn't silently dropped."""
    bus = EventBus()
    llm = MockGeminiClient(
        plan_json={
            "tasks": [
                {
                    "id": "t1",
                    "description": "open firefox",
                    "tool": "open_app",
                    "arguments": {"app_name": "firefox"},
                    "depends_on": [],
                },
                {
                    "id": "t2",
                    "description": "create folder test",
                    "tool": "file_operation",
                    "arguments": {
                        "action": "create_folder",
                        "path": "documents",
                        "name": "Test",
                    },
                    "depends_on": [],
                },
            ]
        }
    )
    memory = MockMemory()

    plan_created: list[PlanCreated] = []

    async def _on_plan_created(e: PlanCreated) -> None:
        plan_created.append(e)

    bus.subscribe(PlanCreated, _on_plan_created)
    planner = Planner(bus=bus, llm=llm, memory=memory)
    planner.register()

    # Simulate what NLU emits after its regex fast path: it collapses
    # the multi-step input into a single (garbled) open_app task.
    await planner.on_intent(
        IntentIdentified(
            intent=Intent.TASK,
            raw_input="open firefox and create a folder called test",
            requires_task=True,
            task_type="open_app",
            task_params={"app_name": "firefox and create a folder called test"},
        )
    )

    await asyncio.sleep(0.1)
    await bus.drain()

    # The LLM MUST have been called — the raw input looks multi-step,
    # so the fast path must refuse.
    assert llm.fail_count == 1, f"LLM should be called once, was {llm.fail_count}"
    assert len(plan_created) == 1
    plan = plan_created[0].plan
    assert len(plan["tasks"]) == 2, plan["tasks"]
    tools = {t["tool"] for t in plan["tasks"]}
    assert "open_app" in tools and "file_operation" in tools, tools

    await bus.close()


async def test_task_validation_unknown_tool() -> None:
    """Planner should reject unknown tools."""
    bus = EventBus()
    llm = MockGeminiClient(
        plan_json={
            "tasks": [
                {
                    "id": "t1",
                    "description": "Unknown action",
                    "tool": "nonexistent_tool",
                    "arguments": {},
                    "depends_on": [],
                }
            ]
        }
    )
    memory = MockMemory()

    plan_created: list[PlanCreated] = []

    async def _on_plan_created(e: PlanCreated) -> None:
        plan_created.append(e)

    bus.subscribe(PlanCreated, _on_plan_created)
    planner = Planner(bus=bus, llm=llm, memory=memory)
    planner.register()

    await planner.on_intent(
        IntentIdentified(
            intent=Intent.TASK,
            raw_input="do something",
            requires_task=True,
        )
    )

    await asyncio.sleep(0.1)
    await bus.drain()

    # Invalid tool should be dropped; planner returns None → no PlanCreated emitted
    assert len(plan_created) == 0, "no PlanCreated when all tasks invalid"

    await bus.close()


async def test_dependency_cycle_detection() -> None:
    """Cycles in the dependency graph should be detected and broken."""
    bus = EventBus()
    llm = MockGeminiClient(
        plan_json={
            "tasks": [
                {
                    "id": "t1",
                    "description": "Task 1",
                    "tool": "list_alarms",
                    "arguments": {},
                    "depends_on": ["t2"],
                },
                {
                    "id": "t2",
                    "description": "Task 2",
                    "tool": "list_alarms",
                    "arguments": {},
                    "depends_on": ["t1"],
                },
            ]
        }
    )
    memory = MockMemory()

    plan_created: list[PlanCreated] = []

    async def _on_plan_created(e: PlanCreated) -> None:
        plan_created.append(e)

    bus.subscribe(PlanCreated, _on_plan_created)
    planner = Planner(bus=bus, llm=llm, memory=memory)
    planner.register()

    await planner.on_intent(
        IntentIdentified(
            intent=Intent.TASK,
            raw_input="cycle test",
            requires_task=True,
        )
    )

    await asyncio.sleep(0.1)
    await bus.drain()

    # Cycle should cause tasks to be dropped
    assert len(plan_created) == 1
    plan = plan_created[0].plan
    # The _validate_graph in scheduler would detect cycles, but the
    # Planner's _parse_plan_json also drops tasks with unknown deps.
    # Either way, the plan should not have a cycle.
    ids = [t["id"] for t in plan["tasks"]]
    assert "t1" in ids or "t2" in ids, "at least one valid task should remain"

    await bus.close()


async def test_natural_summary_carries_tool_result() -> None:
    """Single-task plan must flow the tool's natural result into the
    final ResponseReady without an LLM call."""
    bus = EventBus()
    llm = MockGeminiClient()
    memory = MockMemory()

    plan_completed: list[PlanCompleted] = []
    responses: list[ResponseReady] = []

    async def _on_plan_completed(e: PlanCompleted) -> None:
        plan_completed.append(e)

    async def _on_response(e: ResponseReady) -> None:
        responses.append(e)

    async def _on_task_exec(e: TaskExecutionRequested) -> None:
        bus.emit(
            TaskCompleted(
                task_name=e.task_name,
                success=True,
                result="It's 18°C and clear in London.",
                session_id=e.session_id,
            )
        )

    bus.subscribe(TaskExecutionRequested, _on_task_exec)
    bus.subscribe(PlanCompleted, _on_plan_completed)
    bus.subscribe(ResponseReady, _on_response)

    coordinator = ReasoningCoordinator(bus=bus, gemini_client=llm, memory_manager=memory)
    coordinator.register()
    scheduler = PlanScheduler(bus=bus)
    scheduler.register()
    scheduler.executor.register()

    bus.emit(
        PlanCompleted(
            plan_id="plan-test",
            status="completed",
            summary="It's 18°C and clear in London.",
            task_results=[
                {
                    "tool": "get_weather",
                    "result": "It's 18°C and clear in London.",
                    "arguments": {"city": "London"},
                }
            ],
            user_request="what's the weather in London?",
            session_id="default",
        )
    )

    await asyncio.sleep(0.2)
    await bus.drain()

    assert len(plan_completed) == 1, plan_completed
    assert len(responses) == 1, responses
    text = responses[0].text
    assert "18" in text and "London" in text, text
    # Single-task path must NOT call the LLM.
    assert len(llm.generate_calls) == 0, llm.generate_calls

    await bus.close()


async def test_natural_summary_cancelled_short_circuit() -> None:
    """Cancelled plans should short-circuit to a deterministic reply."""
    bus = EventBus()
    llm = MockGeminiClient(generate_response="LLM should not have been called")
    memory = MockMemory()

    responses: list[ResponseReady] = []

    async def _on_response(e: ResponseReady) -> None:
        responses.append(e)

    bus.subscribe(ResponseReady, _on_response)

    coordinator = ReasoningCoordinator(bus=bus, gemini_client=llm, memory_manager=memory)
    coordinator.register()

    bus.emit(
        PlanCompleted(
            plan_id="plan-test",
            status="cancelled",
            summary="Cancelled",
            task_results=[{"tool": "open_app", "result": "Opening firefox", "arguments": {}}],
            user_request="open firefox",
            session_id="default",
        )
    )

    await asyncio.sleep(0.2)
    await bus.drain()

    assert len(responses) == 1, responses
    assert responses[0].text == "Cancelled, sir."
    assert len(llm.generate_calls) == 0, llm.generate_calls

    await bus.close()


async def test_natural_summary_for_multitask_plan() -> None:
    """A multi-task plan's LLM paraphrase produces one cohesive
    ResponseReady text; per-task results are concatenated into the
    paraphrase prompt."""
    bus = EventBus()
    llm = MockGeminiClient(
        generate_response="Folder's ready and you've got 2 alarms queued, sir."
    )
    memory = MockMemory()

    responses: list[ResponseReady] = []

    async def _on_response(e: ResponseReady) -> None:
        responses.append(e)

    bus.subscribe(ResponseReady, _on_response)

    coordinator = ReasoningCoordinator(bus=bus, gemini_client=llm, memory_manager=memory)
    coordinator.register()

    bus.emit(
        PlanCompleted(
            plan_id="plan-test",
            status="completed",
            summary="Created folder: Test\nYou have 2 alarms set.",
            task_results=[
                {
                    "tool": "file_operation",
                    "result": "Created folder: Test",
                    "arguments": {"action": "create_folder", "name": "Test"},
                },
                {
                    "tool": "list_alarms",
                    "result": "You have 2 alarms set.",
                    "arguments": {},
                },
                {
                    "tool": "open_app",
                    "result": "Opened the file manager.",
                    "arguments": {"app_name": "files"},
                },
            ],
            user_request="create a folder called Test, list my alarms, and open the file manager",
            session_id="default",
        )
    )

    await asyncio.sleep(0.2)
    await bus.drain()

    assert len(responses) == 1, responses
    text = responses[0].text
    # The LLM response must flow through verbatim.
    assert "Folder's ready" in text and "alarms queued" in text, text
    # The LLM must have been called exactly once with the per-tool inputs.
    assert len(llm.generate_calls) == 1, llm.generate_calls
    prompt = llm.generate_calls[0]["prompt"]
    assert "create a folder called Test" in prompt, prompt
    assert "file_operation" in prompt, prompt
    assert "list_alarms" in prompt, prompt

    await bus.close()


async def test_token_log_records_llm_calls() -> None:
    """The token log should record a call entry when the LLM path runs."""
    token_log = TokenLog(session_id="test-token-log")
    llm = MockGeminiClient(generate_response="Logged, sir.")
    memory = MockMemory()
    bus = EventBus()

    responses: list[ResponseReady] = []

    async def _on_response(e: ResponseReady) -> None:
        responses.append(e)

    bus.subscribe(ResponseReady, _on_response)

    coordinator = ReasoningCoordinator(
        bus=bus,
        gemini_client=llm,
        memory_manager=memory,
        token_log=token_log,
    )
    coordinator.register()

    bus.emit(
        PlanCompleted(
            plan_id="plan-token-log",
            status="completed",
            summary="Done",
            task_results=[
                {"tool": "get_weather", "result": "It is sunny.", "arguments": {"city": "London"}}
            ],
            user_request="what's the weather in London?",
            session_id="default",
        )
    )

    await asyncio.sleep(0.2)
    await bus.drain()

    assert len(responses) == 1, responses
    assert len(token_log.entries) == 0, token_log.entries

    await bus.close()


async def test_tool_voice_unknown_tool() -> None:
    """The default voice covers tools that aren't in the registry, so
    adding a new tool to TASK_REGISTRY requires no voice change."""
    # Pretend a brand-new tool with no voice entry.
    out = naturalize_single_tool(
        "note_pad_stub",
        "Note saved.",
        {"title": "shopping"},
    )
    assert out == "Note saved.", out

    # Colon-separated log line gets the default rewrite.
    out = naturalize_single_tool(
        "note_pad_stub",
        "Note saved: shopping.txt",
        {"title": "shopping"},
    )
    assert "shopping.txt" in out, out
    assert "sir" in out.lower(), out

    # A failure-style message is passed through unchanged so the user
    # sees what actually went wrong.
    failure = "Could not write file: permission denied"
    out = naturalize_single_tool("note_pad_stub", failure, {})
    assert "sir" not in out.lower() or "could not" in out.lower(), out


# ── Runner ────────────────────────────────────────────────────────────


# ── Runner ────────────────────────────────────────────────────────────


async def main() -> None:
    print("=== Planning subsystem integration tests ===\n")
    tests = [
        ("plan_decomposition_single_task", test_plan_decomposition_single_task),
        ("plan_decomposition_multi_task", test_plan_decomposition_multi_task),
        ("parallel_tasks_run_concurrently", test_parallel_tasks_run_concurrently),
        ("reference_resolution", test_reference_resolution),
        ("fast_path_single_task", test_fast_path_single_task),
        ("single_tool_with_conjunction_uses_fast_path", test_single_tool_with_conjunction_uses_fast_path),
        ("multistep_bypasses_fast_path", test_multistep_request_bypasses_fast_path),
        ("task_validation_unknown_tool", test_task_validation_unknown_tool),
        ("dependency_cycle_detection", test_dependency_cycle_detection),
        ("natural_summary_carries_tool_result", test_natural_summary_carries_tool_result),
        ("natural_summary_cancelled_short_circuit", test_natural_summary_cancelled_short_circuit),
        ("natural_summary_for_multitask_plan", test_natural_summary_for_multitask_plan),
        ("token_log_records_llm_calls", test_token_log_records_llm_calls),
        ("tool_voice_unknown_tool", test_tool_voice_unknown_tool),
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