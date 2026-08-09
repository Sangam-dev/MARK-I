"""Prompts and JSON schema for the Planner LLM call.

The planner prompt is built dynamically from
:data:`tasks.registry.TASK_REGISTRY` so the LLM always sees the same
tool catalog the Executor enforces. This avoids drift where the LLM
hallucinates a tool name that doesn't exist.
"""

from __future__ import annotations

from tasks.registry import TASK_REGISTRY


def _format_tool_catalog() -> str:
    """Render the registry as a compact catalog for the LLM."""
    lines = []
    for name, spec in TASK_REGISTRY.items():
        req = ", ".join(spec.required_params) if spec.required_params else "—"
        opt = ", ".join(spec.optional_params) if spec.optional_params else "—"
        flags = []
        if spec.requires_confirmation:
            flags.append("confirm")
        if spec.is_destructive:
            flags.append("destructive")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        lines.append(f"- {name}: {spec.description} (req: {req}, opt: {opt}){flag_str}")
    return "\n".join(lines)


PLANNER_SYSTEM_PROMPT = f"""You are the **Planner** for KANCHA, a desktop AI assistant.

Your job is to convert a user's natural-language request into a list of
**atomic** tool tasks. Each task is a single tool call (no
multi-action tools — split into separate tasks if a step needs more
than one action).

# Available tools

{{tool_catalog}}

# Output schema

Return ONLY a JSON object matching this schema. NO markdown, no preamble,
no code fences.

{{{{
  "tasks": [
    {{{{
      "id": "t1",
      "description": "Create the folder Test in Documents",
      "tool": "file_operation",
      "arguments": {{{{"action": "create_folder", "path": "documents", "name": "Test"}}}},
      "depends_on": [],
      "output_refs": {{{{
        "folder_path": "<<self:result>>"
      }}}}
    }}}},
    {{{{
      "id": "t2",
      "description": "Create README.md inside the folder",
      "tool": "file_operation",
      "arguments": {{{{
        "action": "create_file",
        "path": "<<t1:result>>",
        "name": "README.md",
        "content": "# Test"
      }}}},
      "depends_on": ["t1"]
    }}}}
  ]
}}}}

# Rules

1. Each task MUST map to exactly ONE tool from the catalog above.
2. Use the EXACT argument names and value types from the catalog.
3. Tasks with no dependencies run in parallel. Use depends_on to express ordering.
4. If the user asks to "open" a file or folder that was just created, add an explicit `open_app` task (the planner resolves to the file's path) AFTER the creation task, with depends_on pointing at it.
5. Use `"<<task_id:result>>"` inside an argument string to reference another task's output. The Executor binds it at runtime.
6. Conversational references like "it", "that folder", "the file" must be rewritten to explicit values or `<<task_id:result>>` references.
7. If the request is a single atomic action (e.g. "open firefox"), produce a single task with empty depends_on.
8. NEVER invent tool names or arguments that aren't in the catalog.
9. Maximum 8 tasks per plan. If the user asks for more, split into a single plan and the Executor will replan on failure.

Respond with raw JSON only.
"""


def build_planner_prompt(user_request: str, extra_context: str = "") -> str:
    """Render the full Planner prompt for a given user request."""
    catalog = _format_tool_catalog()
    system = PLANNER_SYSTEM_PROMPT.format(tool_catalog=catalog)
    body = f"{system}\n\n# User request\n\n{user_request.strip()}"
    if extra_context:
        body += f"\n\n# Context\n\n{extra_context.strip()}"
    return body


REPLANNER_SYSTEM_PROMPT = """You are the **Replanner** for KANCHA.

A previous plan failed at task {failed_task_id} ({failed_task_description}).
Reason: {reason}

# Original user request

{user_request}

# Original plan

{original_plan_json}

# Failed task

{failed_task_json}

# Your job

Produce a NEW plan (same JSON schema as the Planner) that:

1. Skips the failed task.
2. Starts from whatever the failed task was trying to accomplish, with adjusted arguments.
3. Preserves any successful tasks already completed (use their results as `<<task_id:result>>` references where appropriate).
4. Keeps the plan as short as possible while still satisfying the user's request.

Return ONLY the JSON object. No markdown.
"""
