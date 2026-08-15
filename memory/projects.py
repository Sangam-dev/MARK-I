"""Project registry and activity log — the assistant's memory of the
projects it builds and what it last did in each.

Layer 2 of navigation. Layer 1 (the fuzzy index in
:mod:`actions.file_controller`) answers "which *file*"; this answers
"which *project*, and what were we doing in it" — which is what turns
"hey, let's continue working on our previous project" into a concrete
directory plus context the delegated agent can resume from, instead of a
vague instruction handed to a blank agent.

Persistence is a small JSON file next to the RAG data (atomic replace),
and the workspace (``~/kancha-workspace`` by default) is re-scanned on
demand so folders the agent creates are discovered without any
registration ceremony. Nothing here imports from ``actions/`` — the
executor glues the two layers together.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("kancha.memory.projects")

_PROJECTS_FILE = Path(__file__).resolve().parent / "data" / "projects.json"
_DEFAULT_WORKSPACE = Path.home() / "kancha-workspace"

#: Cap on the activity log — it is a recent-work trail, not an archive.
_MAX_ACTIVITY = 500

#: Spoken references that mean "the most recently used project".
_MOST_RECENT_RE = re.compile(
    r"\b(?:our|the|this|that|my|last|current|previous|latest|most\s+recent)\s+"
    r"(?:active\s+)?(?:project|work|codebase|repo|repository)\b",
    re.IGNORECASE,
)

#: A name wrapped in a project skeleton: "the kancha project", "our
#: study plan work". The name inside is matched against registered
#: projects, so a reference with a determiner never falls through to the
#: whole-sentence fuzzy match (which would drown the name in verbs).
_SKELETON_RE = re.compile(
    r"\b(?:the|our|my|this|that|a|an)\s+(?P<name>[\w ._+-]+?)\s+"
    r"(?:project|repo|repository|codebase|work)\b",
    re.IGNORECASE,
)

#: Tokens that add nothing when matching a project name by voice.
_PROJECT_STOPWORDS = frozenset(
    {
        "the", "a", "an", "my", "our", "your", "his", "her", "its", "their",
        "of", "in", "on", "at", "for", "to", "with", "by", "from", "into",
        "and", "or", "project", "projects", "folder", "directory", "dir",
        "called", "named", "which", "that", "this", "please", "inside",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(
        t
        for t in _TOKEN_RE.findall((text or "").lower())
        if len(t) > 1 and t not in _PROJECT_STOPWORDS
    )


@dataclass(slots=True)
class Project:
    """One registered project. ``root`` is an absolute path."""

    name: str
    root: str
    created_at: float
    last_used: float
    description: str = ""
    key_files: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Activity:
    """One recorded delegation: what was asked, where, and when."""

    ts: float
    session: str
    project: str
    task: str
    files: list[str] = field(default_factory=list)


def _description_for(root: Path) -> str:
    """A one-line description from the project's README, if it has one."""
    for candidate in ("README.md", "readme.md", "README.txt"):
        readme = root / candidate
        if not readme.is_file():
            continue
        try:
            for line in readme.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip().lstrip("#").strip()
                if line and not line.startswith(("![", "<")):
                    return line[:160]
        except OSError:
            pass
    return ""


def _newest_mtime(root: Path) -> float | None:
    newest: float | None = None
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "node_modules"]
            for fname in filenames:
                if fname.startswith("."):
                    continue
                try:
                    mtime = os.stat(Path(dirpath) / fname).st_mtime
                except OSError:
                    continue
                if newest is None or mtime > newest:
                    newest = mtime
    except OSError:
        pass
    return newest


class ProjectStore:
    """JSON-backed registry + activity log, with workspace auto-discovery."""

    def __init__(
        self,
        path: str | Path | None = None,
        workspace: str | Path | None = None,
    ) -> None:
        self._path = Path(path) if path else _PROJECTS_FILE
        self._workspace = Path(workspace) if workspace else _DEFAULT_WORKSPACE
        self._lock = threading.Lock()

    @property
    def workspace(self) -> Path:
        return self._workspace

    # ── persistence ───────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("projects"), dict):
                return data
        except (OSError, ValueError):
            pass
        return {"version": 1, "projects": {}, "activity": []}

    def _save(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ── discovery ─────────────────────────────────────────────────────

    def ensure_indexed(self) -> None:
        """Register every top-level folder in the workspace as a project.

        Idempotent and cheap: existing projects keep their description,
        and a project's ``last_used`` is only advanced here when its
        folder has newer files than anything recorded (so a folder that
        was touched by hand still counts).
        """
        if not self._workspace.is_dir():
            return
        with self._lock:
            data = self._load()
            changed = False
            try:
                children = sorted(
                    p for p in self._workspace.iterdir() if p.is_dir() and not p.name.startswith(".")
                )
            except OSError:
                return
            for child in children:
                if child.name in data["projects"]:
                    continue
                newest = _newest_mtime(child)
                data["projects"][child.name] = {
                    "name": child.name,
                    "root": str(child.resolve()),
                    "created_at": newest or time.time(),
                    "last_used": newest or time.time(),
                    "description": _description_for(child),
                    "key_files": [],
                }
                changed = True
                logger.info("Registered project '%s' at %s", child.name, child)
            if changed:
                self._save(data)

    # ── queries ───────────────────────────────────────────────────────

    def list_projects(self) -> list[Project]:
        data = self._load()
        projects = [
            Project(
                name=p["name"],
                root=p["root"],
                created_at=p.get("created_at", 0.0),
                last_used=p.get("last_used", 0.0),
                description=p.get("description", ""),
                key_files=list(p.get("key_files", [])),
            )
            for p in data["projects"].values()
        ]
        projects.sort(key=lambda p: p.last_used, reverse=True)
        return projects

    def resolve_project(self, reference: str) -> Project | None:
        """Turn a spoken *reference* into a project, or ``None``.

        ``"our previous project"`` / ``"last project"`` / ``"the
        project"`` mean the most recently used one. Anything else is
        matched as a fuzzy name: "study plan" resolves a project whose
        name contains study/plan tokens. Ambiguous or absent → ``None``,
        so the caller leaves the request untouched rather than guessing.
        """
        self.ensure_indexed()
        projects = self.list_projects()
        if not projects:
            return None

        text = (reference or "").strip()
        if not text:
            return None
        if _MOST_RECENT_RE.search(text):
            return projects[0]

        # "the <name> project" — match the name on its own first.
        skeleton = _SKELETON_RE.search(text)
        if skeleton:
            hint = skeleton.group("name").strip()
            by_name = self._match_name(hint, projects)
            if by_name is not None:
                return by_name

        query = set(_tokenize(text))
        if not query:
            return None

        scored: list[tuple[float, Project]] = []
        for project in projects:
            name_tokens = set(_tokenize(project.name))
            if not name_tokens:
                continue
            hits = len(query & name_tokens)
            if hits == 0:
                continue
            score = hits / max(len(query), len(name_tokens))
            scored.append((score, project))

        if not scored:
            return None
        scored.sort(key=lambda t: (t[0], t[1].last_used), reverse=True)
        top, second = scored[0], (scored[1] if len(scored) > 1 else None)
        # Only act on a clear winner: the name matched at least half the
        # query, and nothing else matched nearly as well.
        if top[0] < 0.5:
            return None
        if second is not None and top[0] - second[0] < 0.25:
            return None
        return top[1]

    @staticmethod
    def _match_name(hint: str, projects: list[Project]) -> Project | None:
        """Best project for a bare *hint* name, or None when unclear."""
        query = set(_tokenize(hint))
        if not query:
            return None
        scored: list[tuple[float, Project]] = []
        for project in projects:
            name_tokens = set(_tokenize(project.name))
            if not name_tokens:
                continue
            hits = len(query & name_tokens)
            if hits == 0:
                continue
            score = hits / max(len(query), len(name_tokens))
            scored.append((score, project))
        if not scored:
            return None
        scored.sort(key=lambda t: (t[0], t[1].last_used), reverse=True)
        top, second = scored[0], (scored[1] if len(scored) > 1 else None)
        if top[0] < 0.5:
            return None
        if second is not None and top[0] - second[0] < 0.25:
            return None
        return top[1]

    def recent_activity(self, project: str, limit: int = 4) -> list[Activity]:
        data = self._load()
        out: list[Activity] = []
        for entry in reversed(data.get("activity", [])):
            if entry.get("project") != project:
                continue
            out.append(
                Activity(
                    ts=entry.get("ts", 0.0),
                    session=entry.get("session", ""),
                    project=entry.get("project", ""),
                    task=entry.get("task", ""),
                    files=list(entry.get("files", [])),
                )
            )
            if len(out) >= limit:
                break
        return out

    def project_context(self, project: Project, limit: int = 3) -> str:
        """A short text block to hand a delegated agent as context."""
        lines = [
            f"Project: {project.name}",
            f"Location: {project.root}",
        ]
        if project.description:
            lines.append(f"What it is: {project.description}")
        recent = self.recent_activity(project.name, limit=limit)
        if recent:
            lines.append("Last activity:")
            for entry in recent:
                when = time.strftime("%b %d %H:%M", time.localtime(entry.ts))
                lines.append(f"  - {when}: {entry.task[:160]}")
        return "\n".join(lines)

    def recent_files(self, project: Project, limit: int = 5) -> list[str]:
        """Most recently modified file names under *project*'s root.

        Used to enrich the activity summary indexed into semantic memory
        ("which files does this project revolve around") without needing
        a per-build snapshot-diff.
        """
        root = Path(project.root)
        if not root.is_dir():
            return []
        scored: list[tuple[float, str]] = []
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [
                    d
                    for d in dirnames
                    if not d.startswith(".")
                    and d not in ("node_modules", ".venv", "venv", "__pycache__")
                ]
                for fname in filenames:
                    if fname.startswith("."):
                        continue
                    try:
                        scored.append((os.stat(Path(dirpath) / fname).st_mtime, fname))
                    except OSError:
                        continue
        except OSError:
            pass
        scored.sort(reverse=True)
        return [name for _, name in scored[:limit]]

    def activity_summary(
        self, project: Project, task: str, limit_files: int = 5
    ) -> str:
        """A condensed, self-contained record of one delegation.

        This is the unit that goes into semantic memory — small by
        construction (a few lines, not the transcript), so retrieval is
        cheap and never crowds anything out.
        """
        lines = [
            f"Project: {project.name}",
            f"Location: {project.root}",
            f"Task: {(task or '').strip()}",
        ]
        files = self.recent_files(project, limit=limit_files)
        if files:
            lines.append("Files: " + ", ".join(files))
        return "\n".join(lines)

    # ── writes ────────────────────────────────────────────────────────

    def record_activity(
        self,
        session: str,
        task: str,
        project: str,
        files: list[str] | None = None,
    ) -> None:
        """Append one activity entry and mark *project* as just used."""
        if not project:
            return
        with self._lock:
            data = self._load()
            data["activity"].append(
                {
                    "ts": time.time(),
                    "session": session,
                    "project": project,
                    "task": (task or "")[:300],
                    "files": list(files or []),
                }
            )
            del data["activity"][:-_MAX_ACTIVITY]
            if project in data["projects"]:
                data["projects"][project]["last_used"] = time.time()
            self._save(data)


# ── shared instance (executor uses this; tests swap it) ───────────────

_project_store: ProjectStore | None = None


def get_shared_project_store() -> ProjectStore:
    global _project_store
    if _project_store is None:
        _project_store = ProjectStore()
    return _project_store


def set_shared_project_store(store: ProjectStore | None) -> None:
    global _project_store
    _project_store = store
