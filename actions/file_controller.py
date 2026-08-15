import os
import re
import shutil
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

try:
    import send2trash
    _SEND2TRASH = True
except ImportError:
    _SEND2TRASH = False

_OS = platform.system()  # "Windows" | "Darwin" | "Linux"

_SAFE_ROOTS: list[Path] = [
    Path.home(),
]

def _is_safe_path(target: Path) -> bool:
    """Verilen path _SAFE_ROOTS içinde mi? Değilse işlemi reddet."""
    try:
        resolved = target.resolve()
        return any(
            resolved == root.resolve() or resolved.is_relative_to(root.resolve())
            for root in _SAFE_ROOTS
        )
    except Exception:
        return False

def _get_desktop() -> Path:
    if _OS == "Linux":
        xdg = os.environ.get("XDG_DESKTOP_DIR", "")
        if xdg and Path(xdg).exists():
            return Path(xdg)
    return Path.home() / "Desktop"

def _get_downloads() -> Path:
    if _OS == "Linux":
        xdg = os.environ.get("XDG_DOWNLOAD_DIR", "")
        if xdg and Path(xdg).exists():
            return Path(xdg)
    return Path.home() / "Downloads"

def _get_documents() -> Path:
    if _OS == "Linux":
        xdg = os.environ.get("XDG_DOCUMENTS_DIR", "")
        if xdg and Path(xdg).exists():
            return Path(xdg)
    return Path.home() / "Documents"

def _get_pictures() -> Path:
    if _OS == "Linux":
        xdg = os.environ.get("XDG_PICTURES_DIR", "")
        if xdg and Path(xdg).exists():
            return Path(xdg)
    return Path.home() / "Pictures"

def _get_music() -> Path:
    if _OS == "Linux":
        xdg = os.environ.get("XDG_MUSIC_DIR", "")
        if xdg and Path(xdg).exists():
            return Path(xdg)
    return Path.home() / "Music"

def _get_videos() -> Path:
    if _OS == "Linux":
        xdg = os.environ.get("XDG_VIDEOS_DIR", "")
        if xdg and Path(xdg).exists():
            return Path(xdg)
    return Path.home() / "Videos"


# Spoken names for the standard user directories. Matched
# case-insensitively against a whole path segment, so "Documents/projects"
# and "documents/projects" both land in the same place.
_SHORTCUT_ALIASES: dict[str, str] = {
    "desktop": "desktop",
    "my desktop": "desktop",
    "downloads": "downloads",
    "download": "downloads",
    "documents": "documents",
    "document": "documents",
    "docs": "documents",
    "pictures": "pictures",
    "picture": "pictures",
    "photos": "pictures",
    "images": "pictures",
    "music": "music",
    "songs": "music",
    "videos": "videos",
    "video": "videos",
    "movies": "videos",
    "home": "home",
    "~": "home",
    "home folder": "home",
    "home directory": "home",
    "my home": "home",
}

# Words a speaker tacks onto a directory name ("the kancha folder") that
# are never part of the name itself.
_PATH_NOISE_RE = re.compile(
    r"^(?:the|my)\s+|\s+(?:folder|directory|dir)$", re.IGNORECASE
)


def _shortcut_bases() -> dict[str, Path]:
    return {
        "desktop":   _get_desktop(),
        "downloads": _get_downloads(),
        "documents": _get_documents(),
        "pictures":  _get_pictures(),
        "music":     _get_music(),
        "videos":    _get_videos(),
        "home":      Path.home(),
    }


def _shortcut_for(segment: str) -> Path | None:
    """The standard directory *segment* names, or None if it names none."""
    key = _PATH_NOISE_RE.sub("", segment.strip()).strip().lower()
    canonical = _SHORTCUT_ALIASES.get(key)
    if canonical is None:
        return None
    return _shortcut_bases()[canonical]


def _split_segments(text: str) -> list[str]:
    """Path text to segments, tolerating either slash and dropping noise."""
    parts = []
    for segment in re.split(r"[\\/]+", text):
        segment = _PATH_NOISE_RE.sub("", segment.strip()).strip()
        if segment and segment != ".":
            parts.append(segment)
    return parts


def _walk_case_insensitively(base: Path, segments: list[str]) -> Path | None:
    """Follow *segments* under *base*, matching names ignoring case.

    Speech and LLMs both hand us "Kancha" for a directory called "kancha";
    an exact-case lookup misses it, and the caller then falls back to its
    default location — which is how "create it in kancha" ended up on the
    Desktop.
    """
    current = base
    for segment in segments:
        direct = current / segment
        if direct.exists():
            current = direct
            continue
        if not current.is_dir():
            return None
        match = None
        lowered = segment.lower()
        try:
            for child in current.iterdir():
                if child.name.lower() == lowered:
                    match = child
                    break
        except (OSError, PermissionError):
            return None
        if match is None:
            return None
        current = match
    return current


def _resolve_path(raw: str) -> Path:
    """Turn whatever the caller said into a concrete absolute path.

    Accepts a standard-directory name ("downloads"), a nested one
    ("documents/projects/notes"), an absolute path, a ``~`` path, and a
    bare directory name ("kancha"). A bare or relative name is anchored
    at the home directory — never at the process's working directory,
    which is wherever the assistant happened to be started from.
    """
    text = str(raw or "").strip().strip("\"'")
    if not text:
        return Path.home()

    text = os.path.expandvars(text)

    if text.startswith("~"):
        return Path(text).expanduser()

    if Path(text).is_absolute():
        return Path(text)

    # A multi-word shortcut ("my home", "home folder") before segmenting,
    # since those contain spaces rather than separators.
    whole = _shortcut_for(text)
    if whole is not None:
        return whole

    segments = _split_segments(text)
    if not segments:
        return Path.home()

    base = _shortcut_for(segments[0])
    if base is not None:
        rest = segments[1:]
        if not rest:
            return base
        found = _walk_case_insensitively(base, rest)
        return found if found is not None else base.joinpath(*rest)

    # Relative to home, then to the standard directories — so "projects/api"
    # finds ~/Documents/projects/api when that is where it actually lives.
    for root in (Path.home(), *_shortcut_bases().values()):
        found = _walk_case_insensitively(root, segments)
        if found is not None:
            return found

    # Nothing there yet (a folder about to be created): anchor at home.
    return Path.home().joinpath(*segments)


def _join_name(base: Path, name: str) -> Path:
    """``base / name`` where *name* may itself be nested or absolute."""
    text = str(name or "").strip().strip("\"'")
    if not text:
        return base
    if text.startswith("~") or Path(text).is_absolute():
        return _resolve_path(text)
    segments = [seg for seg in re.split(r"[\\/]+", text) if seg and seg != "."]
    if not segments:
        return base
    return base.joinpath(*segments)

def _pretty(target: Path) -> str:
    """A short, speakable form of *target* — "~/kancha/notes"."""
    try:
        return "~/" + str(target.resolve().relative_to(Path.home()))
    except ValueError:
        return str(target)


def _format_size(b: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


# ── Fuzzy name resolution ───────────────────────────────────────────────
#
# Finding things by *spoken* names — "the study plan file", "our project"
# — is a token-match problem, not a substring one: "study plan" will never
# be a substring of "final_assessment_study_plan.md". The old find matched
# ``name.lower() in item.name.lower()`` and therefore missed every fuzzy
# reference, which is what turned one "open the study plan" request into a
# three-turn find/open saga. This section indexes a root once (cached,
# bounded) and ranks candidates by token overlap, with recently-touched
# files weighted higher — the things people ask about are the things they
# just built.

# Directories that never hold something a user wants located by name:
# VCS internals, dependency trees, virtualenvs and build caches.
_INDEX_EXCLUDED_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", ".bzr", "node_modules", ".venv", "venv",
        "env", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        ".cache", ".local", ".config", ".idea", ".vscode", "dist", "build",
        ".next", ".turbo", ".nuxt", "coverage", "htmlcov", ".tox",
        "site-packages", "jarvis_frontend/electron-out", "electron-out",
    }
)

# Tokens that carry no distinguishing signal in a spoken file name.
_FIND_STOPWORDS = frozenset(
    {
        "the", "a", "an", "my", "our", "your", "his", "her", "its",
        "their", "of", "in", "on", "at", "for", "to", "with", "by",
        "from", "into", "inside", "under", "over", "and", "or", "file",
        "files", "folder", "folders", "directory", "dir", "document",
        "documents", "project", "projects", "thing", "things", "one",
        "some", "called", "named", "which", "that", "this", "please",
    }
)

# Index entries go stale as builds land new files, so cache briefly.
_INDEX_TTL_S = 10.0
_MAX_INDEX_DIRS = 2000
_MAX_INDEX_ENTRIES = 6000
_MAX_INDEX_DEPTH = 5

# Below this a match is noise, not a hit: "study plan" against a paper
# that merely lives in the same folder scores ~0.25 and must not flood a
# find listing.
_FIND_MIN_SCORE = 0.30

#: Above this score a fuzzy match is treated as "the thing" and used
#: without asking; above the floor but below it, the caller offers the
#: top candidates instead of guessing.
_RESOLVE_CONFIDENT = 0.55
_RESOLVE_AMBIGUOUS = 0.30


@dataclass(slots=True)
class _IndexedEntry:
    path: Path
    name: str
    stem_tokens: tuple[str, ...]
    dir_tokens: tuple[str, ...]
    mtime: float
    is_dir: bool = False
    size: int = 0


# root -> (built_at, entries)
_index_cache: dict[Path, tuple[float, list[_IndexedEntry]]] = {}


def _tokenize(text: str) -> tuple[str, ...]:
    """Meaningful lowercase tokens from *text* — no stopwords, no single
    letters, no punctuation. Splitting on non-alphanumerics is what lets
    "study plan" and "final_assessment_study_plan.md" share tokens."""
    return tuple(
        t
        for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(t) > 1 and t not in _FIND_STOPWORDS
    )


def _make_index_entry(path: Path, stat: os.stat_result, *, is_dir: bool) -> _IndexedEntry:
    stem = path.name.rsplit(".", 1)[0] if not is_dir else path.name
    dir_tokens = _tokenize(path.parent.name) if not is_dir else ()
    return _IndexedEntry(
        path=path,
        name=path.name,
        stem_tokens=_tokenize(stem),
        dir_tokens=dir_tokens,
        mtime=stat.st_mtime,
        is_dir=is_dir,
        size=stat.st_size,
    )


def _index_root(root: Path) -> list[_IndexedEntry]:
    """Recursively index *root*, cached for a few seconds.

    The walk prunes excluded directories (``node_modules``, ``.git``,
    virtualenvs, build output) so indexing home does not descend into
    megabytes of dependencies, and caps both directory count and entry
    count so a cold walk stays bounded.
    """
    root = root.resolve()
    now = time.time()
    cached = _index_cache.get(root)
    if cached is not None and now - cached[0] < _INDEX_TTL_S:
        return cached[1]

    entries: list[_IndexedEntry] = []
    dirs_seen = 0
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            # Hidden entries are lock files, editor swaps and config
            # noise — never what a spoken name points at.
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d not in _INDEX_EXCLUDED_DIRS and not d.startswith(".")
            )
            filenames = [f for f in filenames if not f.startswith(".")]
            try:
                depth = len(Path(dirpath).relative_to(root).parts)
            except ValueError:
                depth = 0
            if depth > _MAX_INDEX_DEPTH:
                dirnames[:] = []
                continue
            dirs_seen += 1
            if dirs_seen > _MAX_INDEX_DIRS:
                break
            if depth > 0:
                try:
                    entries.append(_make_index_entry(Path(dirpath), os.stat(dirpath), is_dir=True))
                except OSError:
                    pass
            for fname in filenames:
                if len(entries) >= _MAX_INDEX_ENTRIES:
                    break
                try:
                    fp = Path(dirpath) / fname
                    entries.append(_make_index_entry(fp, os.stat(fp), is_dir=False))
                except OSError:
                    continue
    except OSError:
        pass

    _index_cache[root] = (now, entries)
    return entries


def _fuzzy_score(query: tuple[str, ...], entry: _IndexedEntry) -> float:
    """Rank how well *query* names *entry*, in 0.0..1.0.

    The stem (filename without extension) counts at full weight, the
    immediate parent directory at half — so "kancha workspace" helps
    without letting the location drown the name. Files touched in the
    last day get a 1.3x boost, the last week 1.15x: people ask about what
    they just built.
    """
    if not query:
        return 0.0
    q = frozenset(query)
    stem = frozenset(entry.stem_tokens)
    dirt = frozenset(entry.dir_tokens)

    hit_stem = len(q & stem)
    hit_dir = len(q & dirt)
    weighted = hit_stem + 0.5 * hit_dir
    if weighted == 0:
        return 0.0

    recall = weighted / len(q)
    precision = weighted / max(1.0, len(stem | (dirt & q)))
    score = 0.6 * recall + 0.4 * precision

    # Every query token lives in the name itself — "study plan" against
    # study_plan.md. That is the clearest possible hit, floor it high.
    if q <= stem:
        score = max(score, 0.85)
    if q == stem:
        score = 1.0

    age_days = max(0.0, (time.time() - entry.mtime) / 86400.0)
    if age_days <= 1.0:
        score *= 1.3
    elif age_days <= 7.0:
        score *= 1.15
    return min(score, 1.0)


def _fuzzy_find(
    name: str,
    search_root: Path,
    *,
    extension: str = "",
    max_results: int = 20,
    include_dirs: bool = False,
) -> list[_IndexedEntry]:
    """Ranked matches for *name* under *search_root* (best first)."""
    query = _tokenize(name)
    scored: list[tuple[float, _IndexedEntry]] = []
    for entry in _index_root(search_root):
        if entry.is_dir and not include_dirs:
            continue
        if extension and entry.path.suffix.lower() != extension.lower():
            continue
        score = _fuzzy_score(query, entry) if query else 1.0
        if score < _FIND_MIN_SCORE:
            continue
        scored.append((score, entry))
    scored.sort(key=lambda t: (t[0], t[1].mtime), reverse=True)
    return [entry for _, entry in scored[:max_results]]


def _fuzzy_search_roots() -> list[Path]:
    """Where a fuzzy name is looked for when the user gave no location:
    the assistant's own workspace first (that is where delegated builds
    land), then the standard folders."""
    roots = [Path.home() / "kancha-workspace"]
    for name in ("Desktop", "Documents", "Downloads", "Pictures", "Music", "Videos"):
        roots.append(Path.home() / name)
    return roots


def resolve_fuzzy_name(
    name: str, *, max_results: int = 3
) -> tuple[str | None, list[tuple[str, str]]]:
    """Resolve a spoken *name* to an absolute path.

    Returns ``(path, [])`` when the top match is confident enough to act
    on, ``(None, [(name, path), ...])`` when several candidates tie (the
    caller turns those into a clarifying question), and ``(None, [])``
    when nothing matched — the caller then lets the tool's own not-found
    message surface.
    """
    text = str(name or "").strip()
    if not text:
        return None, []

    # A URL is a target, not a *name* — its path tokens ("example.com",
    # "pdf") only ever match some unrelated file on disk. Leave it to the
    # tool, same as the executor's open-routing does.
    if "://" in text or text.startswith("www."):
        return None, []

    # Already a real path? Not our job — the tool resolves and opens it.
    try:
        if _resolve_path(text).exists():
            return None, []
    except Exception:  # noqa: BLE001
        pass

    query = _tokenize(text)
    if not query:
        return None, []

    best: list[tuple[float, _IndexedEntry]] = []
    seen: set[Path] = set()
    for root in _fuzzy_search_roots():
        if not root.exists():
            continue
        for entry in _index_root(root):
            if entry.path in seen:
                continue
            seen.add(entry.path)
            score = _fuzzy_score(query, entry)
            if score <= 0:
                continue
            best.append((score, entry))

    if not best:
        return None, []

    best.sort(key=lambda t: (t[0], t[1].mtime), reverse=True)
    top = best[0][0]
    if top >= _RESOLVE_CONFIDENT:
        return str(best[0][1].path), []
    if top >= _RESOLVE_AMBIGUOUS:
        return None, [(e.name, str(e.path)) for _, e in best[:max_results]]
    return None, []

def _safe_trash(target: Path) -> str:

    if not _SEND2TRASH:
        return (
            "send2trash is not installed. "
            "Run: pip install send2trash — "
            "Permanent deletion is disabled for safety."
        )
    send2trash.send2trash(str(target))
    return f"Moved to Trash: {target.name}"


def list_files(path: str = "desktop", show_hidden: bool = False) -> str:
    try:
        target = _resolve_path(path)
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        if not target.exists():
            return f"Path not found: {target}"
        if not target.is_dir():
            return f"Not a directory: {target}"

        items = []
        for item in sorted(target.iterdir()):
            if not show_hidden and item.name.startswith("."):
                continue
            if item.is_dir():
                items.append(f"📁 {item.name}/")
            else:
                size = _format_size(item.stat().st_size)
                items.append(f"📄 {item.name} ({size})")

        if not items:
            return f"Directory is empty: {_pretty(target)}"

        return f"Contents of {_pretty(target)} ({len(items)} items):\n" + "\n".join(items)

    except PermissionError:
        return f"Permission denied: {path}"
    except Exception as e:
        return f"Error listing files: {e}"


def create_file(path: str, name: str = "", content: str = "") -> str:
    try:
        base   = _resolve_path(path)
        target = _join_name(base, name)
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"File created: {target.name} in {_pretty(target.parent)}"
    except Exception as e:
        return f"Could not create file: {e}"


def create_folder(path: str, name: str = "") -> str:
    try:
        base   = _resolve_path(path)
        target = _join_name(base, name)
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        if target.is_dir():
            return f"Folder already exists: {_pretty(target)}"
        target.mkdir(parents=True, exist_ok=True)
        # Report the location, not just the leaf name — "Folder created:
        # Test" gave no way to notice it had landed somewhere unintended.
        return f"Folder created: {target.name} in {_pretty(target.parent)}"
    except Exception as e:
        return f"Could not create folder: {e}"


def delete_file(path: str, name: str = "") -> str:
    try:
        base   = _resolve_path(path)
        target = _join_name(base, name)
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        if not target.exists():
            return f"Not found: {target.name}"

        # Güvenli dizin kontrolü — kritik kullanıcı klasörlerini koru
        protected = {
            _get_desktop(), _get_downloads(), _get_documents(),
            _get_pictures(), _get_music(), _get_videos(), Path.home()
        }
        if target.resolve() in {p.resolve() for p in protected}:
            return f"Protected directory, cannot delete: {target.name}"

        return _safe_trash(target)

    except PermissionError:
        return f"Permission denied: {path}"
    except Exception as e:
        return f"Could not delete: {e}"


def move_file(path: str, name: str = "", destination: str = "") -> str:
    try:
        base   = _resolve_path(path)
        src    = _join_name(base, name)
        dst    = _resolve_path(destination) if destination else None

        if not src.exists():
            return f"Source not found: {src.name}"
        if dst is None:
            return "No destination specified."
        if not _is_safe_path(src):
            return f"Access denied (source): {src}"
        if not _is_safe_path(dst):
            return f"Access denied (destination): {dst}"

        if dst.is_dir():
            dst = dst / src.name

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"Moved: {src.name} → {dst.parent.name}/"

    except Exception as e:
        return f"Could not move: {e}"


def copy_file(path: str, name: str = "", destination: str = "") -> str:
    try:
        base = _resolve_path(path)
        src  = _join_name(base, name)
        dst  = _resolve_path(destination) if destination else None

        if not src.exists():
            return f"Source not found: {src.name}"
        if dst is None:
            return "No destination specified."
        if not _is_safe_path(src):
            return f"Access denied (source): {src}"
        if not _is_safe_path(dst):
            return f"Access denied (destination): {dst}"

        if dst.is_dir():
            dst = dst / src.name

        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.is_dir():
            shutil.copytree(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))

        return f"Copied: {src.name} → {dst.parent.name}/"

    except Exception as e:
        return f"Could not copy: {e}"


def rename_file(path: str, name: str = "", new_name: str = "") -> str:
    try:
        base     = _resolve_path(path)
        target   = _join_name(base, name)
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        if not target.exists():
            return f"Not found: {target.name}"
        if not new_name:
            return "No new name provided."

        new_path = target.parent / new_name
        if new_path.exists():
            return f"A file named '{new_name}' already exists here."

        target.rename(new_path)
        return f"Renamed: {target.name} → {new_name}"

    except Exception as e:
        return f"Could not rename: {e}"


def read_file(path: str, name: str = "", max_chars: int = 4000) -> str:
    try:
        base   = _resolve_path(path)
        target = _join_name(base, name)
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        if not target.exists():
            return f"File not found: {target.name}"
        if not target.is_file():
            return f"Not a file: {target.name}"

        content = target.read_text(encoding="utf-8", errors="ignore")
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n[Truncated — {len(content)} total chars]"
        return content

    except Exception as e:
        return f"Could not read file: {e}"


def write_file(path: str, name: str = "", content: str = "",
               append: bool = False) -> str:
    try:
        base   = _resolve_path(path)
        target = _join_name(base, name)
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(target, mode, encoding="utf-8") as f:
            f.write(content)
        action = "Appended to" if append else "Written to"
        return f"{action}: {target.name}"
    except Exception as e:
        return f"Could not write file: {e}"


def find_files(name: str = "", extension: str = "",
               path: str = "home", max_results: int = 20) -> str:
    try:
        search_path = _resolve_path(path)
        if not _is_safe_path(search_path):
            return f"Access denied: {search_path}"
        if not search_path.exists():
            return f"Search path not found: {path}"

        matches = _fuzzy_find(
            name, search_path, extension=extension, max_results=max_results
        )
        if not matches:
            query = name or extension or "files"
            return f"No {query} found in {search_path.name}/"

        lines = [
            f"📄 {entry.name} ({_format_size(entry.size)}) — {entry.path.parent}"
            for entry in matches
        ]
        return f"Found {len(matches)} file(s):\n" + "\n".join(lines)

    except Exception as e:
        return f"Search error: {e}"


def get_largest_files(path: str = "downloads", count: int = 10) -> str:
    count = min(count, 50)  # maksimum 50
    try:
        search_path = _resolve_path(path)
        if not _is_safe_path(search_path):
            return f"Access denied: {search_path}"
        if not search_path.exists():
            return f"Path not found: {path}"

        files = []
        for item in search_path.rglob("*"):
            if item.is_file():
                try:
                    files.append((item.stat().st_size, item))
                except Exception:
                    continue

        files.sort(reverse=True)
        top = files[:count]

        if not top:
            return "No files found."

        lines = [f"Top {len(top)} largest files in {search_path.name}/:"]
        for size, f in top:
            lines.append(f"  {_format_size(size):>10}  {f.name}  ({f.parent})")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: {e}"


def get_disk_usage(path: str = "home") -> str:
    try:
        target = _resolve_path(path)
        usage  = shutil.disk_usage(target)
        pct    = usage.used / usage.total * 100
        return (
            f"Disk usage ({target}):\n"
            f"  Total : {_format_size(usage.total)}\n"
            f"  Used  : {_format_size(usage.used)} ({pct:.1f}%)\n"
            f"  Free  : {_format_size(usage.free)}"
        )
    except Exception as e:
        return f"Could not get disk usage: {e}"


def organize_desktop() -> str:
    type_map = {
        "Images":    {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg", ".ico", ".heic"},
        "Documents": {".pdf", ".doc", ".docx", ".txt", ".xls", ".xlsx",
                      ".ppt", ".pptx", ".csv", ".odt", ".ods", ".odp"},
        "Videos":    {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v"},
        "Music":     {".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a"},
        "Archives":  {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"},
        "Code":      {".py", ".js", ".ts", ".html", ".css", ".json", ".xml",
                      ".cpp", ".java", ".cs", ".go", ".rs", ".sh"},
    }

    desktop = _get_desktop()
    moved, skipped = [], []

    try:
        for item in desktop.iterdir():
            # Klasörlere, gizli dosyalara ve organize klasörlerine dokunma
            if item.is_dir() or item.name.startswith("."):
                continue
            if item.name in {k for k in type_map}:
                continue

            ext        = item.suffix.lower()
            target_dir = desktop / "Others"
            for folder, exts in type_map.items():
                if ext in exts:
                    target_dir = desktop / folder
                    break

            target_dir.mkdir(exist_ok=True)
            new_path = target_dir / item.name

            if new_path.exists():
                skipped.append(item.name)
                continue

            shutil.move(str(item), str(new_path))
            moved.append(f"{item.name} → {target_dir.name}/")

        result = f"Desktop organized: {len(moved)} files moved."
        if moved:
            preview = moved[:8]
            result += "\n" + "\n".join(preview)
            if len(moved) > 8:
                result += f"\n... and {len(moved) - 8} more."
        if skipped:
            result += f"\n{len(skipped)} file(s) skipped (name conflict)."
        return result

    except Exception as e:
        return f"Could not organize desktop: {e}"


def get_file_info(path: str, name: str = "") -> str:
    try:
        base   = _resolve_path(path)
        target = _join_name(base, name)
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        if not target.exists():
            return f"Not found: {target.name}"

        stat = target.stat()
        info = {
            "Name":      target.name,
            "Type":      "Folder" if target.is_dir() else "File",
            "Size":      _format_size(stat.st_size),
            "Location":  str(target.parent),
            "Created":   datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M"),
            "Modified":  datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            "Extension": target.suffix or "—",
        }
        return "\n".join(f"  {k}: {v}" for k, v in info.items())

    except Exception as e:
        return f"Could not get file info: {e}"

def file_controller(
    parameters: dict = None,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    params = parameters or {}
    action = params.get("action", "").lower().strip()
    path   = params.get("path", "desktop")
    name   = params.get("name", "")

    if player:
        player.write_log(f"[file] {action} {name or path}")

    try:
        if action == "list":
            return list_files(path)

        elif action == "create_file":
            return create_file(path, name=name, content=params.get("content", ""))

        elif action == "create_folder":
            return create_folder(path, name=name)

        elif action == "delete":
            return delete_file(path, name=name)

        elif action == "move":
            return move_file(path, name=name, destination=params.get("destination", ""))

        elif action == "copy":
            return copy_file(path, name=name, destination=params.get("destination", ""))

        elif action == "rename":
            return rename_file(path, name=name, new_name=params.get("new_name", ""))

        elif action == "read":
            return read_file(path, name=name)

        elif action == "write":
            return write_file(
                path, name=name,
                content=params.get("content", ""),
                append=params.get("append", False)
            )

        elif action == "find":
            return find_files(
                name=name or params.get("name", ""),
                extension=params.get("extension", ""),
                path=path,
                max_results=min(int(params.get("max_results", 20)), 50),
            )

        elif action == "largest":
            return get_largest_files(
                path=path,
                count=int(params.get("count", 10)),
            )

        elif action == "disk_usage":
            return get_disk_usage(path)

        elif action == "organize_desktop":
            return organize_desktop()

        elif action == "info":
            return get_file_info(path, name=name)

        else:
            return f"Unknown action: '{action}'"

    except Exception as e:
        return f"File controller error ({action}): {e}"