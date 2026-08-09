"""Resolve conversational references (``"it"``, ``"that folder"``) inside
planned task arguments.

The Planner may emit an argument like ``"open it"`` or ``"the folder I
just created"``. The Resolver rewrites these to ``<<task_id:result>>``
placeholders that the Executor binds at dispatch time.

The Resolver has two passes:

1. **Pronouns** — replace bare ``it`` / ``its`` with the result of the
   most recent task that produced an artifact (folder, file, alarm).
2. **Demonstrative noun phrases** — replace ``that folder``, ``the
   file``, ``that alarm`` etc. with a typed reference, choosing the
   best prior task by tool type and description.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ── Reference patterns ──────────────────────────────────────────────────

# Matches a *bare* pronoun "it" or "its" as a whole argument value.
# We deliberately avoid matching inside long arguments; those are usually
# already concrete.
_BARE_PRONOUN_RE = re.compile(r"^\s*(it|its)\s*$", re.IGNORECASE)

# Matches demonstrative noun phrases like "that folder", "the file",
# "this alarm". Captures the noun so we can pick the right prior task.
_DEMONSTRATIVE_RE = re.compile(
    r"\b(?:that|the|this)\s+(?P<noun>folder|directory|file|alarm|"
    r"reminder|app|application|note|notes|document)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class TaskArtifact:
    """Metadata about a previously completed task's output."""

    task_id: str
    tool: str
    description: str
    result: str


class ReferenceResolver:
    """Stateless reference resolver.

    The caller provides a list of prior task outputs (typically the
    tasks already completed in the current plan). The resolver returns
    rewritten arguments and a list of ``output_refs`` that the
    Executor should bind.
    """

    # Heuristic mapping from a demonstrative noun to the tool most
    # likely to have produced that artifact. If a prior task matches,
    # its result is preferred as the reference target.
    _NOUN_TO_TOOLS: dict[str, tuple[str, ...]] = {
        "folder": ("file_operation",),
        "directory": ("file_operation",),
        "file": ("file_operation",),
        "document": ("file_operation",),
        "alarm": ("set_alarm",),
        "reminder": ("set_alarm",),
        "app": ("open_app",),
        "application": ("open_app",),
        "note": ("file_operation",),
        "notes": ("file_operation",),
    }

    def resolve(
        self,
        arguments: dict[str, Any],
        prior_outputs: list[TaskArtifact],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Rewrite *arguments* in place-style.

        Returns ``(new_arguments, output_refs)``. ``new_arguments`` is a
        shallow copy with string substitutions applied. ``output_refs``
        maps ``arg_name -> "task_id:result"`` for the bindings the
        Executor should apply at dispatch time.
        """
        if not prior_outputs:
            return arguments, {}

        new_args: dict[str, Any] = {}
        refs: dict[str, str] = {}

        for arg_name, value in arguments.items():
            if not isinstance(value, str):
                new_args[arg_name] = value
                continue

            rewritten, ref = self._rewrite_string(value, prior_outputs)
            new_args[arg_name] = rewritten
            if ref is not None:
                refs[arg_name] = ref

        return new_args, refs

    # ── internals ───────────────────────────────────────────────────

    def _rewrite_string(
        self,
        value: str,
        prior_outputs: list[TaskArtifact],
    ) -> tuple[str, str | None]:
        """Rewrite a single string argument.

        Returns ``(rewritten, ref_or_None)``. ``ref_or_None`` is the
        ``"task_id:result"`` binding the Executor should apply; if the
        value did not require any rewrite, it's ``None``.
        """

        # 1. Bare pronoun — most recent artifact wins.
        if _BARE_PRONOUN_RE.match(value):
            target = self._most_recent_artifact(prior_outputs)
            if target is None:
                return value, None
            placeholder = f"<<{target.task_id}:result>>"
            return placeholder, f"{target.task_id}:result"

        # 2. Demonstrative noun — pick the prior task whose tool
        #    matches the noun's expected tool class.
        match = _DEMONSTRATIVE_RE.search(value)
        if match:
            noun = match.group("noun").lower()
            target = self._artifact_for_noun(noun, prior_outputs)
            if target is None:
                target = self._most_recent_artifact(prior_outputs)
            if target is not None:
                # Rewrite just the noun phrase so the surrounding
                # context is preserved.
                placeholder = f"<<{target.task_id}:result>>"
                rewritten = _DEMONSTRATIVE_RE.sub(placeholder, value, count=1)
                return rewritten, f"{target.task_id}:result"

        return value, None

    def _most_recent_artifact(
        self, prior_outputs: list[TaskArtifact]
    ) -> TaskArtifact | None:
        return prior_outputs[-1] if prior_outputs else None

    def _artifact_for_noun(
        self, noun: str, prior_outputs: list[TaskArtifact]
    ) -> TaskArtifact | None:
        preferred = self._NOUN_TO_TOOLS.get(noun, ())
        # Walk backwards so the most recent match wins.
        for artifact in reversed(prior_outputs):
            if artifact.tool in preferred:
                return artifact
        return None
