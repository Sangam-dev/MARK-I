"""Planner — the Task LLM. Converts a delegated task into an :class:`ExecutionPlan`.

The Planner is a **subordinate executor**. It subscribes to
:class:`TaskDispatched` — the work the Task Orchestrator
(``planning/orchestrator.py``) decided to run — and never to raw user
input, nor even to the Conversation LLM's :class:`TaskRequested`. It
therefore never decides *whether* something is a task; that decision
belongs to the Conversation LLM alone. Its job starts once the work has
already been decided, and is limited to: pick the tool(s), order them,
get them run, report back.

It also never speaks to the user. When it needs a value it does not
have, or the work turns out to be sensitive, it says so through
:class:`TaskProtocolResponse` in the closed vocabulary of
:mod:`planning.protocol` — ``input_required``,
``confirmation_required``, ``execute``, ``completed``, ``failed`` — and
the Orchestrator decides what reaches the conversation.

Approval never originates here. A plan runs a sensitive tool only when
:attr:`TaskDispatched.user_confirmed` is set, which only the
Orchestrator sets, and only from a real subsequent user message. Any
``confirm`` flag the planning LLM invents is stripped
(:func:`_sanitise_confirm`).

It asks the LLM to decompose the delegated instruction into atomic tool
tasks. Each task is validated against
:data:`tasks.registry.TASK_REGISTRY`, references are resolved, and the
plan is emitted as a :class:`PlanCreated` event for the Scheduler.

The Planner is **only** a strategist. It never runs tools itself. On
permanent task failure the Scheduler emits
:class:`PlanReplanRequested`; the Planner's :meth:`replan` rebuilds a
new plan starting from the remaining work.

When the plan reaches a terminal state the Planner correlates the
:class:`PlanCompleted` back to the originating :class:`TaskDispatched`
and reports it as a ``completed``/``failed`` protocol response —
structured execution data, never a user-facing sentence. Phrasing the
outcome (including failures) is the Conversation LLM's job.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from core.bus import EventBus
from core.events import (
    PlanCompleted,
    PlanCreated,
    PlanReplanRequested,
    TaskDispatched,
    TaskProtocolResponse,
)
from memory.manager import MemoryManager
from memory.token_log import TokenLog
from nlu.classifier import classify_tool_request
from planning.models import (
    ExecutionPlan,
    PlannedTask,
    PlanStatus,
    TaskStatus,
)
from planning.prompts import REPLANNER_SYSTEM_PROMPT, build_planner_prompt
from planning.reference import ReferenceResolver, TaskArtifact
from reasoning.llm_client import GeminiClient
from tasks.policy import (
    describe_sensitive_action,
    missing_required_fields,
    question_for_fields,
)
from tasks.registry import TASK_REGISTRY, validate_task

logger = logging.getLogger("kancha.planning.planner")


# ── Helpers ───────────────────────────────────────────────────────────


# Heuristic: multi-step requests contain one or more of these conjunctions
# between distinct actions. We deliberately err on the side of "looks
# multi-step" so the Planner's LLM gets a chance to decompose — better
# to over-decompose than to silently drop steps.
_MULTISTEP_RE = re.compile(
    r"""
    \s+(?:and|then|after\s+that|afterwards|next|also|plus|,\s+|\;\s+)
    \s+
    """,
    re.IGNORECASE | re.VERBOSE,
)

_VERBS_RE = re.compile(
    r"\b(?:create|make|delete|remove|set|cancel|list|show|find|search|get|read|write|move|copy|rename|open|launch|start|send|reply|tell|ask|remind)\b",
    re.IGNORECASE,
)


def _looks_multistep(text: str) -> bool:
    """True if *text* probably contains two or more distinct actions.

    Returns True for BOTH of these shapes:
      - Two or more verbs joined by a conjunction
          ("create a folder and open Firefox" → verb_count=2)
      - One verb with multiple targets joined by a conjunction
          ("open firefox and file explorer" → verb_count=1, but still two tool calls)
    """
    if not text:
        return False
    has_conjunction = _MULTISTEP_RE.search(text) is not None
    verb_count = len(_VERBS_RE.findall(text))
    # Any conjunction + at least one action verb → let the LLM planner decide.
    # Previously required verb_count >= 2, which silently dropped the second
    # target in "open X and Y" (one verb, two apps).
    if has_conjunction and verb_count >= 1:
        return True
    # More than one enumerator ("first ... second ...", numbered list).
    if re.search(
        r"\b(?:first|1\.|1\))\s.*\b(?:second|2\.|2\))\s",
        text,
        re.IGNORECASE | re.DOTALL,
    ):
        return True
    return False


def _strip_session_default(request: TaskDispatched) -> str:
    """The TaskDispatched.session_id is set by the Conversation LLM; if the
    planner is invoked directly (e.g. from tests), fall back to
    'default'."""
    return request.session_id or "default"


def _format_delegation_context(request: TaskDispatched) -> str:
    """Render the delegation's extra fields for the decomposition prompt.

    Everything here is a hint from the Conversation LLM — the expected
    shape of the answer, parameters it already resolved, and what any
    conversational reference pointed at. The instruction itself is passed
    separately as the user request.
    """
    lines: list[str] = []
    if request.task_type:
        lines.append(f"- suggested tool: {request.task_type}")
    if request.parameters:
        lines.append(f"- known parameters: {json.dumps(request.parameters)}")
    if request.expected_result:
        lines.append(f"- expected result: {request.expected_result}")
    if request.context:
        lines.append(f"- resolved references: {json.dumps(request.context)}")
    if not lines:
        return ""
    return (
        "Delegated task details (the suggested tool is a hint, not an "
        "instruction — use the catalog):\n" + "\n".join(lines)
    )


def _strip_confirm_flag(arguments: dict[str, Any]) -> dict[str, Any]:
    """Arguments minus any ``confirm`` flag."""
    return {k: v for k, v in (arguments or {}).items() if k != "confirm"}


def _sanitise_confirm(
    tool: str, arguments: dict[str, Any], user_confirmed: bool
) -> dict[str, Any]:
    """Apply approval to a tool's arguments — or remove it.

    The planning LLM is perfectly capable of writing ``"confirm": true``
    into the arguments it invents, and a tool that trusted it would
    execute a destructive action nobody approved. So the flag is always
    stripped first, then re-added only when the Orchestrator says a real
    user approved this task, and only for a step that actually needs it.
    """
    clean = _strip_confirm_flag(arguments)
    if user_confirmed and describe_sensitive_action(tool, clean) is not None:
        clean["confirm"] = True
    return clean


def _plan_to_dict(plan: ExecutionPlan) -> dict[str, Any]:
    """Serialize a plan for inclusion in :class:`PlanCreated`."""
    return {
        "id": plan.id,
        "user_request": plan.user_request,
        "session_id": plan.session_id,
        "status": plan.status.value,
        "tasks": [
            {
                "id": t.id,
                "description": t.description,
                "tool": t.tool,
                "arguments": dict(t.arguments),
                "depends_on": list(t.depends_on),
                "retryable": t.retryable,
                "max_retries": t.max_retries,
                "status": t.status.value,
                "output_refs": dict(t.output_refs),
            }
            for t in plan.tasks
        ],
    }


def _one_task_plan(
    tool: str,
    params: dict[str, Any],
    instruction: str,
    session_id: str,
) -> ExecutionPlan:
    """Wrap a single validated tool call in a one-task plan.

    It still goes through the Scheduler, so retries, replan and the
    normal completion path all behave identically to a decomposed plan.
    """
    return ExecutionPlan(
        id=f"plan-{uuid.uuid4().hex[:8]}",
        user_request=instruction,
        tasks=[
            PlannedTask(
                id="t1",
                description=f"{tool}({params})",
                tool=tool,
                arguments=params,
                depends_on=(),
                retryable=True,
                max_retries=1,
                status=TaskStatus.PENDING,
            )
        ],
        session_id=session_id,
        status=PlanStatus.CREATED,
    )


def _build_single_task_plan(
    request: TaskDispatched,
    session_id: str,
) -> ExecutionPlan | None:
    """Fast path: build a one-task plan without an LLM call.

    Two ways to hit it, in order of trust:

    1. The Conversation LLM named a ``task_type`` that exists in the
       registry and whose parameters validate.
    2. The deterministic matcher in :mod:`nlu.classifier` recognises the
       instruction ("open firefox", "what's the weather in London").

    Note what neither of these does: decide whether the turn *is* a
    task. That is settled before we get here — the delegation itself is
    the decision. This is tool **selection** only, and it exists purely
    to keep the common single-action case free of a second LLM call.

    If the instruction contains conjunctions ("open firefox **and**
    create a folder") or enumerations, we deliberately refuse the fast
    path: the regex matcher latches onto the first verb and silently
    drops the rest. The Planner's LLM call is the only thing that can
    decompose multi-step requests.
    """
    instruction = (request.instruction or "").strip()

    if _looks_multistep(instruction):
        logger.info("Fast-path skipped — request looks multi-step: %r", instruction)
        return None

    # 1. Trust the Conversation LLM's tool hint when it holds up.
    task_type = (request.task_type or "").strip()
    if task_type:
        if task_type not in TASK_REGISTRY:
            logger.info(
                "Delegated task_type %r is not in the registry — decomposing instead",
                task_type,
            )
        else:
            params = dict(request.parameters or {})
            ok, reason = validate_task(task_type, params)
            if ok:
                return _one_task_plan(task_type, params, instruction, session_id)
            logger.info(
                "Delegated parameters for %s failed validation (%s) — "
                "falling through to tool selection",
                task_type,
                reason,
            )

    # 2. Deterministic matcher on the (already self-contained) instruction.
    if instruction:
        decision = classify_tool_request(instruction)
        if decision is not None:
            params = dict(decision.parameters)
            ok, _ = validate_task(decision.task_name, params)
            if ok:
                logger.info(
                    "Matched delegated instruction to %s%s (no LLM call)",
                    decision.task_name,
                    f" params={params}" if params else "",
                )
                return _one_task_plan(
                    decision.task_name, params, instruction, session_id
                )

    return None


# ── Planner ───────────────────────────────────────────────────────────


class Planner:
    """LLM-driven task decomposer + replanner."""

    def __init__(
        self,
        bus: EventBus,
        llm: GeminiClient,
        memory: MemoryManager,
        max_tasks_per_plan: int = 8,
        token_log: TokenLog | None = None,
    ) -> None:
        self._bus = bus
        self._llm = llm
        self._memory = memory
        self._max_tasks = max_tasks_per_plan
        self._token_log = token_log
        self._resolver = ReferenceResolver()
        # plan_id -> the TaskDispatched that produced it. This is what lets
        # a PlanCompleted find its way back to the Conversation LLM as a
        # TaskResultReady. A replan re-registers the same request under
        # the new plan id.
        self._delegations: dict[str, TaskDispatched] = {}
        # Plans whose replan is being built. Their completion is held back
        # until we know whether a replacement plan exists, so one
        # delegation never produces two results (and the user never hears
        # two answers for one request).
        self._replanning: set[str] = set()
        self._deferred_completions: dict[str, PlanCompleted] = {}

    # ── registration ──────────────────────────────────────────────

    def register(self) -> None:
        """Subscribe the Task LLM to dispatches — and nothing else.

        Deliberately absent: any subscription to ``TextInputReceived``,
        ``TranscriptReady``, ``IntentIdentified`` **or**
        ``TaskRequested``. Raw user input must reach the Conversation
        LLM only, and conversational delegations must pass through the
        Orchestrator, which is the only emitter of
        :class:`TaskDispatched`.
        """
        self._bus.subscribe(TaskDispatched, self.on_task_dispatched)
        self._bus.subscribe(PlanReplanRequested, self.on_replan_requested)
        self._bus.subscribe(PlanCompleted, self.on_plan_completed)

    # ── public handlers ───────────────────────────────────────────

    async def on_task_dispatched(self, event: TaskDispatched) -> None:
        """Plan one dispatched task, or report what stands in the way.

        Three outcomes, in the order they are checked:

        1. Something required is missing → ``input_required``. Checked
           before anything runs, so "send an email to john" asks what to
           write rather than sending an empty message.
        2. The work is sensitive and unapproved → ``confirmation_required``.
        3. Otherwise the plan is dispatched to the Scheduler.
        """
        session_id = _strip_session_default(event)
        logger.info(
            "Task dispatched to the Task LLM | task_id=%s type=%s attempt=%d | %r",
            event.task_id,
            event.task_type or "(unspecified)",
            event.attempt,
            event.instruction,
        )

        plan = await self.plan_from_request(event, session_id)
        if plan is None:
            self._report_unplannable(event)
            return

        # Ask before doing: a plan whose arguments are incomplete must
        # not be half-run and then abandoned.
        missing = self._missing_across_plan(plan)
        if missing:
            self._emit_protocol(
                event.task_id,
                "input_required",
                {
                    "task_id": event.task_id,
                    "missing_fields": missing,
                    "question": question_for_fields(missing, event.task_type),
                },
                session_id,
            )
            return

        sensitive = self._sensitive_in_plan(plan)
        if sensitive and not event.user_confirmed:
            action, description, data = sensitive
            self._emit_protocol(
                event.task_id,
                "confirmation_required",
                {
                    "task_id": event.task_id,
                    "action": action,
                    "description": description,
                    "confirmation_data": data,
                },
                session_id,
            )
            return

        # Approval, where it exists, is applied here and nowhere else.
        for task in plan.tasks:
            task.arguments = _sanitise_confirm(
                task.tool, task.arguments, event.user_confirmed
            )

        # Apply reference resolution pass.
        self._apply_references(plan)

        self._delegations[plan.id] = event
        self._emit_protocol(
            event.task_id,
            "execute",
            {
                "task_id": event.task_id,
                "action": plan.tasks[0].tool if plan.tasks else event.task_type,
                "params": dict(plan.tasks[0].arguments) if plan.tasks else {},
            },
            session_id,
        )
        self._bus.emit(
            PlanCreated(
                plan=_plan_to_dict(plan),
                session_id=session_id,
            )
        )

    def _report_unplannable(self, event: TaskDispatched) -> None:
        """No plan could be built — ask, if asking would help.

        A delegation naming a known tool but lacking a required argument
        is answerable ("which city?"); one we simply could not decompose
        is not.
        """
        task_type = (event.task_type or "").strip()
        if task_type in TASK_REGISTRY:
            missing = missing_required_fields(task_type, dict(event.parameters or {}))
            if missing:
                self._emit_protocol(
                    event.task_id,
                    "input_required",
                    {
                        "task_id": event.task_id,
                        "missing_fields": missing,
                        "question": question_for_fields(missing, task_type),
                    },
                    event.session_id,
                )
                return

        self._emit_protocol(
            event.task_id,
            "failed",
            {
                "task_id": event.task_id,
                "error": "could not decompose the instruction into known tools",
            },
            event.session_id,
        )

    @staticmethod
    def _missing_across_plan(plan: ExecutionPlan) -> list[str]:
        """Required arguments absent from any task in the plan.

        Arguments filled by a ``<<task:result>>`` reference are not
        missing — they arrive when the task they depend on runs.
        """
        missing: list[str] = []
        for task in plan.tasks:
            resolved = {
                key: value
                for key, value in task.arguments.items()
                if key not in task.output_refs
            }
            for name in missing_required_fields(task.tool, resolved):
                if name not in missing and name not in task.output_refs:
                    missing.append(name)
        return missing

    @staticmethod
    def _sensitive_in_plan(
        plan: ExecutionPlan,
    ) -> tuple[str, str, dict[str, Any]] | None:
        """The first sensitive step, as (action, description, data)."""
        descriptions: list[str] = []
        first_tool = ""
        data: dict[str, Any] = {}
        for task in plan.tasks:
            described = describe_sensitive_action(task.tool, task.arguments)
            if described is None:
                continue
            if not first_tool:
                first_tool = task.tool
                data = {
                    "tool": task.tool,
                    "arguments": _strip_confirm_flag(task.arguments),
                }
            descriptions.append(described)

        if not descriptions:
            return None
        if len(descriptions) == 1:
            return first_tool, descriptions[0], data
        joined = ", ".join(descriptions[:-1]) + f" and {descriptions[-1]}"
        return first_tool, joined, data

    async def on_plan_completed(self, event: PlanCompleted) -> None:
        """Return the finished plan to the Conversation LLM as a Task Result.

        Plans this Planner did not create (none exist today, but the bus
        is open) are ignored rather than answered for. A plan that is
        being replanned right now is held: reporting it *and* its
        replacement would answer one request twice.
        """
        if event.plan_id in self._replanning:
            self._deferred_completions[event.plan_id] = event
            logger.debug(
                "Holding completion of plan %s while its replan is decided",
                event.plan_id,
            )
            return
        self._report_completion(event)

    def _report_completion(self, event: PlanCompleted) -> None:
        request = self._delegations.pop(event.plan_id, None)
        if request is None:
            logger.debug(
                "PlanCompleted for %s has no delegation on record — ignoring",
                event.plan_id,
            )
            return

        error = "" if event.status == "completed" else (event.summary or "").strip()
        self._emit_result(
            request,
            status=event.status,
            results=list(event.task_results or []),
            error=error,
            session_id=event.session_id or request.session_id,
        )

    def _flush_deferred_completion(self, plan_id: str) -> None:
        """Report a completion that was held while a replan was decided."""
        held = self._deferred_completions.pop(plan_id, None)
        if held is not None:
            self._report_completion(held)

    def _emit_result(
        self,
        request: TaskDispatched,
        status: str,
        results: list[dict[str, Any]],
        error: str,
        session_id: str,
    ) -> None:
        """Report an outcome to the Orchestrator. Execution data, no prose."""
        logger.info(
            "Task result | task_id=%s status=%s tools=%d%s",
            request.task_id,
            status,
            len(results),
            f" error={error!r}" if error else "",
        )
        if status == "failed" and not results:
            self._emit_protocol(
                request.task_id,
                "failed",
                {"task_id": request.task_id, "error": error or "the task failed"},
                session_id,
            )
            return

        self._emit_protocol(
            request.task_id,
            "completed",
            {
                "task_id": request.task_id,
                "result": {
                    "status": status,
                    "results": results,
                    "error": error,
                },
            },
            session_id,
        )

    def _emit_protocol(
        self,
        task_id: str,
        response_type: str,
        payload: dict[str, Any],
        session_id: str,
    ) -> None:
        """Emit one structured Task LLM response.

        Everything this layer has to say goes out through here, which is
        what keeps "the Task LLM never addresses the user" checkable
        rather than aspirational.
        """
        logger.info(
            "task_protocol_response | task_id=%s type=%s", task_id, response_type
        )
        self._bus.emit(
            TaskProtocolResponse(
                task_id=task_id,
                type=response_type,
                payload=payload,
                session_id=session_id or "default",
            )
        )

    async def on_replan_requested(self, event: PlanReplanRequested) -> None:
        """Build a new plan starting from a previously-failed task."""
        # We don't have the original ExecutionPlan in memory (the
        # Scheduler keeps it private), so the replan prompt is built
        # from the failed-task context the caller passes in.
        failed_task = {
            "id": event.failed_task_id,
            "reason": event.reason,
            "remaining": event.remaining_tasks,
        }
        prompt_body = (
            f"{REPLANNER_SYSTEM_PROMPT}\n\n"
            f"Original user request: {event.remaining_tasks[0].get('user_request', '') if event.remaining_tasks else ''}\n\n"
            f"Failed task context: {json.dumps(failed_task)}"
        )
        # The delegation (if this plan came from one) knows the original
        # instruction, which is better replan material than the sparse
        # remaining-task payload.
        request = self._delegations.get(event.plan_id)
        user_request = (
            event.remaining_tasks[0].get("user_request", "")
            if event.remaining_tasks
            else ""
        )
        if not user_request and request is not None:
            user_request = request.instruction

        if request is not None:
            self._replanning.add(event.plan_id)
        try:
            new_plan = await self._ask_llm_for_plan(
                user_request=user_request,
                extra_context=(
                    f"Previously failed task: {event.failed_task_id} ({event.reason}). "
                    "Skip it and continue with the rest of the request."
                ),
            )
        finally:
            self._replanning.discard(event.plan_id)

        if new_plan is None:
            logger.warning(
                "Replan for plan %s produced no plan; giving up.",
                event.plan_id,
            )
            # Nothing replaces it, so the original outcome is the answer.
            self._flush_deferred_completion(event.plan_id)
            return
        self._apply_references(new_plan)
        if request is not None:
            # The retry now owns the delegation. The original plan's
            # completion — already in hand or still to arrive — is
            # superseded and must stay silent, or the user gets told twice.
            self._delegations.pop(event.plan_id, None)
            self._deferred_completions.pop(event.plan_id, None)
            self._delegations[new_plan.id] = request
        self._bus.emit(
            PlanCreated(
                plan=_plan_to_dict(new_plan),
                session_id=event.session_id,
            )
        )

    # ── public helpers (also used by tests) ───────────────────────

    async def plan_from_request(
        self, request: TaskDispatched, session_id: str
    ) -> ExecutionPlan | None:
        """Build an ExecutionPlan for a delegated :class:`TaskDispatched`."""
        # Fast path: a single atomic action we can map without an LLM.
        fast = _build_single_task_plan(request, session_id)
        if fast is not None:
            return fast

        # Slow path: ask the LLM to decompose the instruction.
        context = await self._gather_context(request.instruction, session_id)
        delegation = _format_delegation_context(request)
        if delegation:
            context = f"{context}\n\n{delegation}" if context else delegation
        return await self._ask_llm_for_plan(
            user_request=request.instruction,
            extra_context=context,
            session_id=session_id,
        )

    async def replan(
        self,
        original_plan: ExecutionPlan,
        failed_task: PlannedTask,
        reason: str,
    ) -> ExecutionPlan | None:
        """Public API for callers that hold a handle to the original plan."""
        return await self._ask_llm_for_plan(
            user_request=original_plan.user_request,
            extra_context=(
                f"Previous plan failed at task {failed_task.id} "
                f"({failed_task.description}): {reason}. "
                f"Skip that task and continue with the rest."
            ),
            session_id=original_plan.session_id,
        )

    # ── internals ─────────────────────────────────────────────────

    async def _ask_llm_for_plan(
        self,
        user_request: str,
        extra_context: str = "",
        session_id: str = "default",
    ) -> ExecutionPlan | None:
        prompt = build_planner_prompt(user_request, extra_context)
        try:
            result = await self._llm.generate_json(
                prompt=f"User request:\n\n{user_request}",
                schema_description=(
                    "An object with a 'tasks' key holding a list of task "
                    "objects. Each task has: id (str), description (str), "
                    "tool (str, must exist in the catalog), arguments "
                    "(object), depends_on (list of task ids), output_refs "
                    "(object mapping arg names to '<<task_id:key>>' "
                    "placeholders)."
                ),
                system=prompt,
                hedge_width=2,
                call_site="planner",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Planner LLM call failed: %s", exc)
            return None

        if not result:
            logger.warning("Planner LLM returned empty result for: %s", user_request)
            return None

        return self._parse_plan_json(result, user_request, session_id)

    def _parse_plan_json(
        self,
        raw: dict[str, Any],
        user_request: str,
        session_id: str,
    ) -> ExecutionPlan | None:
        tasks_raw = raw.get("tasks")
        if not isinstance(tasks_raw, list) or not tasks_raw:
            logger.warning("Planner JSON missing 'tasks' list: %r", raw)
            return None

        if len(tasks_raw) > self._max_tasks:
            logger.warning(
                "Plan has %d tasks; truncating to %d",
                len(tasks_raw),
                self._max_tasks,
            )
            tasks_raw = tasks_raw[: self._max_tasks]

        # Normalize ids — the LLM occasionally repeats t1.
        seen_ids: set[str] = set()
        tasks: list[PlannedTask] = []
        for i, t in enumerate(tasks_raw):
            tid = str(t.get("id") or f"t{i + 1}").strip()
            if not tid or tid in seen_ids:
                tid = f"t{i + 1}"
            seen_ids.add(tid)
            tool = str(t.get("tool") or "").strip()
            if tool not in TASK_REGISTRY:
                logger.warning(
                    "Planner returned unknown tool %r — dropping task %s",
                    tool,
                    tid,
                )
                continue
            args = dict(t.get("arguments") or {})
            ok, reason = validate_task(tool, args)
            if not ok:
                logger.warning(
                    "Planner task %s failed validation (%s) — dropping",
                    tid,
                    reason,
                )
                continue
            deps = tuple(str(d) for d in (t.get("depends_on") or ()))
            tasks.append(
                PlannedTask(
                    id=tid,
                    description=str(t.get("description") or tid),
                    tool=tool,
                    arguments=args,
                    depends_on=deps,
                    retryable=bool(t.get("retryable", True)),
                    max_retries=int(t.get("max_retries", 1)),
                    status=TaskStatus.PENDING,
                    output_refs=dict(t.get("output_refs") or {}),
                )
            )

        if not tasks:
            return None

        # Validate the graph; drop cyclic/unknown-dep tasks.
        ids = {t.id for t in tasks}
        clean_tasks: list[PlannedTask] = []
        for t in tasks:
            if all(d in ids for d in t.depends_on):
                clean_tasks.append(t)
            else:
                logger.warning(
                    "Planner task %s has unresolved deps %s — dropping",
                    t.id,
                    t.depends_on,
                )

        if not clean_tasks:
            return None

        return ExecutionPlan(
            id=f"plan-{uuid.uuid4().hex[:8]}",
            user_request=user_request,
            tasks=clean_tasks,
            session_id=session_id,
            status=PlanStatus.CREATED,
        )

    def _apply_references(self, plan: ExecutionPlan) -> None:
        """Run the ReferenceResolver over each task's arguments."""
        prior: list[TaskArtifact] = []
        for t in plan.tasks:
            t.arguments, new_refs = self._resolver.resolve(t.arguments, prior)
            for arg_name, ref in new_refs.items():
                t.output_refs[arg_name] = ref
            prior.append(
                TaskArtifact(
                    task_id=t.id,
                    tool=t.tool,
                    description=t.description,
                    result=t.result,  # empty until executed
                )
            )
        plan.references_resolved = True

    async def _gather_context(self, user_request: str, session_id: str) -> str:
        """Build a short context block for the Planner prompt."""
        try:
            facts = await self._memory.get_all_facts()
        except Exception:  # noqa: BLE001
            facts = []
        recent = self._memory.short_term.get_recent(limit=4)
        recent_lines = [f"- {m['role']}: {m['content']}" for m in recent]
        fact_lines = [
            f"- {f['key']}: {f['value']}" for f in facts if "key" in f and "value" in f
        ]
        parts: list[str] = []
        if recent_lines:
            parts.append("Recent conversation:\n" + "\n".join(recent_lines))
        if fact_lines:
            parts.append("Durable user facts:\n" + "\n".join(fact_lines))
        return "\n\n".join(parts)
