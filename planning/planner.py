"""Planner — convert a user request into an :class:`ExecutionPlan`.

The Planner subscribes to :class:`IntentIdentified` and, when the intent
is a task (or potentially multi-task), asks the LLM to decompose it
into atomic tool tasks. Each task is validated against
:data:`tasks.registry.TASK_REGISTRY`, references are resolved, and the
plan is emitted as a :class:`PlanCreated` event for the Scheduler.

The Planner is **only** a strategist. It never runs tools itself. On
permanent task failure the Scheduler emits
:class:`PlanReplanRequested`; the Planner's :meth:`replan` rebuilds a
new plan starting from the remaining work.
"""

from __future__ import annotations

import json
import logging
import re
import uuid

from typing import Any

from core.bus import EventBus
from core.events import (
    Intent,
    IntentIdentified,
    PlanCreated,
    PlanReplanRequested,
    ResponseReady,
)
from memory.manager import MemoryManager
from memory.token_log import TokenLog
from planning.models import (
    ExecutionPlan,
    PlanStatus,
    PlannedTask,
    TaskStatus,
)
from planning.prompts import REPLANNER_SYSTEM_PROMPT, build_planner_prompt
from planning.reference import ReferenceResolver, TaskArtifact
from reasoning.llm_client import GeminiClient
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
    """True if *text* probably contains two or more distinct actions."""
    if not text:
        return False
    has_conjunction = _MULTISTEP_RE.search(text) is not None
    verb_count = len(_VERBS_RE.findall(text))
    if has_conjunction and verb_count >= 2:
        return True
    # More than one enumerator ("first ... second ...", numbered list).
    if re.search(r"\b(?:first|1\.|1\))\s.*\b(?:second|2\.|2\))\s", text, re.IGNORECASE | re.DOTALL):
        return True
    return False


def _strip_session_default(intent: IntentIdentified) -> str:
    """The IntentIdentified.session_id is set by the bridge; if the
    planner is invoked directly (e.g. from tests), fall back to
    'default'."""
    return intent.session_id or "default"


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


def _build_single_task_plan(
    intent: IntentIdentified,
    session_id: str,
) -> ExecutionPlan | None:
    """Fast path: when NLU already produced a concrete tool decision
    **and** the raw request looks like a single atomic action, skip
    the LLM and build a one-task plan directly.

    The plan is functionally identical to what the old
    ``ReasoningCoordinator`` did, but it goes through the same
    Scheduler so retries, replan, and parallel execution still work.

    If the raw input contains conjunctions ("open firefox **and**
    create a folder") or enumerations, we deliberately refuse the
    fast path: NLU's regex often latches onto the first verb and
    silently drops the rest. The Planner's LLM call is the only
    thing that can decompose multi-step requests.
    """
    if not intent.requires_task:
        return None
    if not intent.task_type:
        return None
    if intent.task_type not in TASK_REGISTRY:
        logger.warning(
            "Fast-path: unknown task_type %s in intent; falling back to LLM",
            intent.task_type,
        )
        return None

    if _looks_multistep(intent.raw_input or ""):
        logger.info(
            "Fast-path skipped — request looks multi-step: %r",
            intent.raw_input,
        )
        return None

    # Strip our private keys out of the params before validation so we
    # don't confuse the registry's type checker.
    params = dict(intent.task_params or {})
    ok, reason = validate_task(intent.task_type, params)
    if not ok:
        logger.warning("Fast-path validation failed: %s", reason)
        return None

    plan_id = f"plan-{uuid.uuid4().hex[:8]}"
    task = PlannedTask(
        id="t1",
        description=f"{intent.task_type}({params})",
        tool=intent.task_type,
        arguments=params,
        depends_on=(),
        retryable=True,
        max_retries=1,
        status=TaskStatus.PENDING,
    )
    return ExecutionPlan(
        id=plan_id,
        user_request=intent.raw_input,
        tasks=[task],
        session_id=session_id,
        status=PlanStatus.CREATED,
    )


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

    # ── registration ──────────────────────────────────────────────

    def register(self) -> None:
        self._bus.subscribe(IntentIdentified, self.on_intent)
        self._bus.subscribe(PlanReplanRequested, self.on_replan_requested)

    # ── public handlers ───────────────────────────────────────────

    async def on_intent(self, event: IntentIdentified) -> None:
        """Handle an intent that may require one or more tool tasks."""
        # Conversational / query intents don't need planning — leave
        # them for the existing ReasoningCoordinator. (The coordinator
        # is still subscribed and will emit ResponseReady for these.)
        if event.intent in (Intent.CONVERSATIONAL, Intent.QUERY):
            return
        if not event.requires_task:
            return

        session_id = _strip_session_default(event)
        plan = await self.plan_from_intent(event, session_id)
        if plan is None:
            # Fallback: just route the single task through the existing
            # TaskExecutionRequested path by emitting a "raw" intent the
            # coordinator already understands. To keep things simple we
            # emit a ResponseReady with a clarification message.
            self._bus.emit(
                ResponseReady(
                    text=(
                        "I couldn't break that request down into steps. "
                        "Could you rephrase it?"
                    ),
                    session_id=session_id,
                )
            )
            return

        # Apply reference resolution pass.
        self._apply_references(plan)

        self._bus.emit(
            PlanCreated(
                plan=_plan_to_dict(plan),
                session_id=session_id,
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
        new_plan = await self._ask_llm_for_plan(
            user_request=(
                event.remaining_tasks[0].get("user_request", "")
                if event.remaining_tasks
                else ""
            ),
            extra_context=(
                f"Previously failed task: {event.failed_task_id} ({event.reason}). "
                "Skip it and continue with the rest of the request."
            ),
        )
        if new_plan is None:
            logger.warning(
                "Replan for plan %s produced no plan; giving up.",
                event.plan_id,
            )
            return
        self._apply_references(new_plan)
        self._bus.emit(
            PlanCreated(
                plan=_plan_to_dict(new_plan),
                session_id=event.session_id,
            )
        )

    # ── public helpers (also used by tests) ───────────────────────

    async def plan_from_intent(
        self, event: IntentIdentified, session_id: str
    ) -> ExecutionPlan | None:
        """Build an ExecutionPlan for an IntentIdentified event."""
        # Fast path: single task the NLU already nailed.
        fast = _build_single_task_plan(event, session_id)
        if fast is not None:
            return fast

        # Slow path: ask the LLM to decompose.
        context = await self._gather_context(event.raw_input, session_id)
        return await self._ask_llm_for_plan(
            user_request=event.raw_input,
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
            tid = str(t.get("id") or f"t{i+1}").strip()
            if not tid or tid in seen_ids:
                tid = f"t{i+1}"
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
            f"- {f['key']}: {f['value']}"
            for f in facts
            if "key" in f and "value" in f
        ]
        parts: list[str] = []
        if recent_lines:
            parts.append("Recent conversation:\n" + "\n".join(recent_lines))
        if fact_lines:
            parts.append("Durable user facts:\n" + "\n".join(fact_lines))
        return "\n\n".join(parts)
