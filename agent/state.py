"""Persistence for delegated OpenCode sessions.

The assistant's memory of its projects lives in ``projects/agent-state.json``
under the configured project root — one file per install, atomically written,
so a restart between turns loses nothing: the session ids, directories,
objectives and last-known state of every project the assistant has ever
delegated to.

The sessions themselves live on OpenCode's side
(``~/.local/share/opencode/opencode.db``); this file is only the mapping that
says "which label is which session, and where does it work". Loading it back
in lets :class:`agent.tool.OpenCodeTool` attach to a session id again instead
of opening a fresh one, which is exactly what "continue working on X" means
after the app has restarted.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("kancha.agent.state")

#: Keep a bounded tail of instructions so "what were we doing" has an
#: answer even without re-reading the whole session.
LAST_INSTRUCTIONS_KEPT = 6


@dataclass(slots=True)
class ProjectRecord:
    """Everything the assistant needs to resume a project by name."""

    label: str
    title: str = ""
    session_id: str = ""
    directory: str = ""
    objective: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    turns: int = 0
    last_instructions: list[str] = field(default_factory=list)
    last_summary: str = ""
    state: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectRecord":
        return cls(
            label=str(data.get("label") or ""),
            title=str(data.get("title") or ""),
            session_id=str(data.get("session_id") or ""),
            directory=str(data.get("directory") or ""),
            objective=str(data.get("objective") or ""),
            created_at=float(data.get("created_at") or 0.0),
            updated_at=float(data.get("updated_at") or 0.0),
            turns=int(data.get("turns") or 0),
            last_instructions=[
                str(instruction)
                for instruction in (data.get("last_instructions") or [])[
                    :LAST_INSTRUCTIONS_KEPT
                ]
            ],
            last_summary=str(data.get("last_summary") or ""),
            state=str(data.get("state") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "title": self.title,
            "session_id": self.session_id,
            "directory": self.directory,
            "objective": self.objective,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turns": self.turns,
            "last_instructions": self.last_instructions[-LAST_INSTRUCTIONS_KEPT:],
            "last_summary": self.last_summary,
            "state": self.state,
        }


class AgentStateStore:
    """Load and atomically persist the project registry.

    The store is deliberately dumb: it reads and writes the JSON and merges
    fresh snapshots over whatever it last knew, so a session evicted from the
    in-memory window (``max_sessions``) is still remembered and can be
    resurrected by name. :class:`agent.tool.OpenCodeTool` owns the merge
    policy; this class only makes persistence not forget things.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[str, ProjectRecord] = {}
        self.active: str = ""
        self.load()

    def load(self) -> None:
        """Read the registry, tolerating a missing or corrupt file."""
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            logger.exception("Ignoring unreadable project state %s", self.path)
            return
        if not isinstance(raw, dict):
            logger.warning("Project state %s is not a mapping — ignoring.", self.path)
            return
        projects = raw.get("projects") or {}
        self.records = {
            str(key): ProjectRecord.from_dict(value)
            for key, value in projects.items()
            if isinstance(value, dict) and str(key)
        }
        active = str(raw.get("active") or "")
        self.active = active if active in self.records else ""

    def save(
        self,
        sessions: dict[str, ProjectRecord],
        active: str,
        drop: set[str] | None = None,
    ) -> None:
        """Merge the live sessions over stored ones and write atomically.

        ``sessions`` is keyed by label and holds the current snapshot. Anything
        stored that is not in it (an evicted project) is kept, so only an
        explicit ``drop`` — ``end_session`` — removes a record for good.
        """
        merged = dict(self.records)
        merged.update(sessions)
        for label in drop or ():
            merged.pop(label, None)
        self.records = merged
        self.active = active if active in merged else ""

        payload: dict[str, Any] = {
            "version": 1,
            "active": self.active,
            "projects": {
                label: record.to_dict() for label, record in sorted(merged.items())
            },
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            # The registry lives in ``projects/``; a delegation pointed at
            # a directory outside it must not make the registry unwritable.
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, self.path)
        except OSError:
            logger.exception("Could not write project state %s", self.path)
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)