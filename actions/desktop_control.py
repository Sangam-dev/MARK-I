"""
desktop.py — Linux desktop automation module for Jarvis.

Supported actions:
    wallpaper        : set wallpaper from local file path
    wallpaper_url    : set wallpaper from URL
    current_wallpaper: get current wallpaper path
    organize         : organize desktop files by type or date
    clean            : archive all desktop files to a dated folder
    list             : list desktop contents
    stats            : desktop file/folder stats
    list_windows     : list all open windows with their current desktop
    focus            : bring a window to the front        (app=)
    close_window     : gracefully close a window          (app=)
    minimize         : minimize a window                  (app=)
    maximize         : toggle maximize a window           (app=)
    list_workspaces  : list all virtual desktops
    switch_workspace : switch active desktop              (target=)
    move_to_workspace: move app to a desktop              (app=, target=, follow=False)
    window_workspace : which desktop is this app on?      (app=)
    task             : AI-powered action via Gemini (sandboxed exec)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("kancha.actions.desktop_control")

# ── Optional deps ────────────────────────────────────────────────────────────
try:
    import pyautogui as _pyautogui
    _HAS_PYAUTOGUI = True
except Exception:
    _HAS_PYAUTOGUI = False

try:
    import keyring as _keyring
    _HAS_KEYRING = True
except ImportError:
    _HAS_KEYRING = False


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


def _get_api_key() -> str:
    """
    Load a Gemini API key.

    Preference order:
      1. The shared key pool (reasoning.llm_client_mulapi) — so desktop
         automation rides the same rotation/cooldown state as chat.
      2. Raw env vars (GEMINI_API_KEY_1..9, GEMINI_API_KEY, GOOGLE_API_KEY).
      3. OS keyring  (service="jarvis", username="gemini_api_key")
      4. config/api_keys.json  (fallback)
    """
    try:
        from reasoning.llm_client_mulapi import get_pool  # noqa: PLC0415

        for entry in get_pool().entries():
            if entry.is_available:
                return entry.key
    except Exception as exc:  # noqa: BLE001
        logger.debug("Key pool unavailable for desktop_control (%s) — using env", exc)

    for i in range(1, 10):
        key = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
        if key:
            return key
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        key = os.getenv(name, "").strip()
        if key:
            return key

    if _HAS_KEYRING:
        val = _keyring.get_password("jarvis", "gemini_api_key")
        if val:
            return val
    path = _base_dir() / "config" / "api_keys.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)["gemini_api_key"]


def _get_desktop() -> Path:
    xdg = os.environ.get("XDG_DESKTOP_DIR", "")
    if xdg:
        p = Path(xdg)
        if p.is_dir():
            return p
    return Path.home() / "Desktop"


def _detect_de() -> str:
    """Return lowercase desktop-environment string."""
    return os.environ.get("XDG_CURRENT_DESKTOP", "").lower()


# ─────────────────────────────────────────────────────────────────────────────
# Wallpaper
# ─────────────────────────────────────────────────────────────────────────────

_SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_wallpaper(image_path: str) -> str:
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        return f"Image not found: {image_path}"
    if path.suffix.lower() not in _SUPPORTED_IMAGE_EXTS:
        return f"Unsupported format: {path.suffix}. Use jpg, png, bmp, or webp."

    uri = f"file://{path}"
    de  = _detect_de()

    try:
        if "gnome" in de or "unity" in de or "budgie" in de or "pantheon" in de:
            _run(["gsettings", "set", "org.gnome.desktop.background", "picture-uri",      uri])
            _run(["gsettings", "set", "org.gnome.desktop.background", "picture-uri-dark", uri])

        elif "kde" in de:
            script = (
                f'var d=desktops();for(var i=0;i<d.length;i++){{'
                f'd[i].wallpaperPlugin="org.kde.image";'
                f'd[i].currentConfigGroup=["Wallpaper","org.kde.image","General"];'
                f'd[i].writeConfig("Image","file://{path}");}}'
            )
            _run(["qdbus", "org.kde.plasmashell", "/PlasmaShell",
                  "org.kde.PlasmaShell.evaluateScript", script])

        elif "xfce" in de:
            _run(["xfconf-query", "-c", "xfce4-desktop",
                  "-p", "/backdrop/screen0/monitor0/workspace0/last-image",
                  "-s", str(path)])

        elif "mate" in de:
            _run(["gsettings", "set", "org.mate.background", "picture-filename", str(path)])

        elif "cinnamon" in de:
            _run(["gsettings", "set", "org.cinnamon.desktop.background", "picture-uri", uri])

        elif "lxde" in de or "lxqt" in de:
            _run(["pcmanfm", "--set-wallpaper", str(path)])

        else:
            # Generic fallback — feh works on most WMs (i3, openbox, bspwm, etc.)
            result = subprocess.run(
                ["feh", "--bg-scale", str(path)],
                capture_output=True
            )
            if result.returncode != 0:
                return (
                    f"Could not set wallpaper for DE '{de}'. "
                    "Install 'feh' or set it manually."
                )

        return f"Wallpaper set: {path.name}"

    except FileNotFoundError as e:
        return f"Required tool not found: {e.filename}. Install it and retry."
    except Exception as e:
        return f"Could not set wallpaper: {e}"


def set_wallpaper_from_url(url: str) -> str:
    suffix = Path(url.split("?")[0]).suffix or ".jpg"
    if suffix.lower() not in _SUPPORTED_IMAGE_EXTS:
        return f"Unsupported image format in URL: {suffix}"

    # Save to a stable cache dir so the path stays valid after the call
    cache_dir = _base_dir() / "cache" / "wallpapers"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"wallpaper_{datetime.now().strftime('%Y%m%d%H%M%S')}{suffix}"

    try:
        urllib.request.urlretrieve(url, str(dest))
    except Exception as e:
        return f"Could not download wallpaper: {e}"

    return set_wallpaper(str(dest))


def get_current_wallpaper() -> str:
    de = _detect_de()
    try:
        if "gnome" in de or "unity" in de or "budgie" in de or "pantheon" in de:
            out = _run_output(["gsettings", "get",
                               "org.gnome.desktop.background", "picture-uri"])
            return f"Current wallpaper: {out.strip("' ")}"

        elif "mate" in de:
            out = _run_output(["gsettings", "get",
                               "org.mate.background", "picture-filename"])
            return f"Current wallpaper: {out.strip("' ")}"

        elif "cinnamon" in de:
            out = _run_output(["gsettings", "get",
                               "org.cinnamon.desktop.background", "picture-uri"])
            return f"Current wallpaper: {out.strip("' ")}"

        elif "xfce" in de:
            out = _run_output(["xfconf-query", "-c", "xfce4-desktop",
                               "-p", "/backdrop/screen0/monitor0/workspace0/last-image"])
            return f"Current wallpaper: {out.strip()}"

        else:
            return f"Wallpaper query not supported for DE: '{de or 'unknown'}'."

    except Exception as e:
        return f"Could not get wallpaper: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Window management
# ─────────────────────────────────────────────────────────────────────────────

# Common app name aliases — maps what a user might say to wmctrl window title keywords
_APP_ALIASES: dict[str, list[str]] = {
    "brave":     ["brave", "brave-browser"],
    "chrome":    ["chrome", "google chrome", "chromium"],
    "firefox":   ["firefox", "mozilla firefox"],
    "terminal":  ["terminal", "konsole", "gnome-terminal", "alacritty",
                  "kitty", "xterm", "tilix", "urxvt"],
    "vscode":    ["visual studio code", "vscode", "code"],
    "files":     ["files", "nautilus", "dolphin", "thunar", "nemo", "pcmanfm"],
    "spotify":   ["spotify"],
    "discord":   ["discord"],
    "telegram":  ["telegram"],
    "slack":     ["slack"],
    "zoom":      ["zoom"],
    "vlc":       ["vlc"],
    "gimp":      ["gimp"],
    "inkscape":  ["inkscape"],
}


def _check_tool(name: str) -> bool:
    """Return True if a CLI tool is available on PATH."""
    return shutil.which(name) is not None


def list_windows() -> str:
    """Return a formatted list of all open windows."""
    if not _check_tool("wmctrl"):
        return "wmctrl is not installed. Run: sudo apt install wmctrl"

    out = _run_output(["wmctrl", "-l"])
    if not out:
        return "No windows found (or wmctrl returned nothing)."

    lines = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) >= 4:
            win_id, desktop_num, host, title = parts
            lines.append(f"  [{desktop_num}] {title}  (id={win_id})")
        elif len(parts) == 3:
            lines.append(f"  {parts[2]}  (id={parts[0]})")

    return f"Open windows ({len(lines)}):\n" + "\n".join(lines)


def focus_window(app_name: str) -> str:
    """
    Bring a window matching `app_name` to the front.
    Tries wmctrl first, falls back to xdotool.
    """
    if not app_name:
        return "No app name provided."

    name_lower = app_name.lower().strip()

    # Resolve alias → list of title keywords to try
    keywords: list[str] = _APP_ALIASES.get(name_lower, [name_lower])

    # ── wmctrl path ──────────────────────────────────────────────────────────
    if _check_tool("wmctrl"):
        for keyword in keywords:
            result = subprocess.run(
                ["wmctrl", "-a", keyword],
                capture_output=True
            )
            if result.returncode == 0:
                return f"Focused: {app_name} (matched '{keyword}' via wmctrl)"

        # wmctrl -a matches by title substring — if none matched, list windows
        # so the user knows what names are available
        windows = _run_output(["wmctrl", "-l"])
        matched = _find_best_window(windows, keywords)
        if matched:
            win_id = matched.split()[0]
            subprocess.run(["wmctrl", "-ia", win_id], capture_output=True)
            title = matched.split(None, 3)[-1] if len(matched.split(None, 3)) >= 4 else matched
            return f"Focused: {title.strip()} (via wmctrl id)"

    # ── xdotool fallback ─────────────────────────────────────────────────────
    if _check_tool("xdotool"):
        for keyword in keywords:
            out = _run_output([
                "xdotool", "search", "--onlyvisible", "--name", keyword
            ])
            if out:
                win_id = out.splitlines()[0].strip()
                subprocess.run(
                    ["xdotool", "windowactivate", "--sync", win_id],
                    capture_output=True
                )
                return f"Focused: {app_name} (matched '{keyword}' via xdotool)"

    # ── nothing worked ────────────────────────────────────────────────────────
    if not _check_tool("wmctrl") and not _check_tool("xdotool"):
        return (
            "Neither wmctrl nor xdotool is installed.\n"
            "Install one: sudo apt install wmctrl   OR   sudo apt install xdotool"
        )

    return (
        f"No open window matched '{app_name}'. "
        f"Tried keywords: {keywords}.\n"
        f"Use 'list_windows' action to see exact window titles."
    )


def _find_best_window(wmctrl_output: str, keywords: list[str]) -> str:
    """Find the best-matching wmctrl -l line for the user's app query.

    Two-pass strategy:

    1. **Exact substring** — try each ``keyword`` as a substring of the
       window title. This preserves the old fast path: if the user says
       ``"chrome"`` and any window contains the literal word ``"chrome"``,
       that wins immediately.

    2. **Fuzzy token overlap** — when no title contains any keyword
       verbatim (e.g. user said ``"tools.py"`` but the open window is
       ``"tool_voice.py - kancha - Visual Studio Code"``), score every
       title by how many tokens it shares with the user's query. The
       best score above :data:`_FUZZY_THRESHOLD` wins.

    Returns the matching wmctrl line (so callers can extract ``win_id``
    and ``title``), or ``""`` if nothing matches.
    """
    lines = [
        ln for ln in wmctrl_output.splitlines()
        if len(ln.split(None, 3)) >= 4
    ]
    if not lines:
        return ""

    # ── Pass 1: exact substring ────────────────────────────────────────
    # Skip keywords shorter than 2 chars — "a" would match the letter "a"
    # anywhere in any title (kancha, assistant, …) and focus the wrong window.
    for line in lines:
        title = line.split(None, 3)[3].lower()
        for kw in keywords:
            if kw and len(kw) >= 2 and kw in title:
                return line

    # ── Pass 2: fuzzy token overlap ────────────────────────────────────
    best_line, best_score = "", 0.0
    for line in lines:
        title_lower = line.split(None, 3)[3].lower()
        # Strip file-extension punctuation and split into tokens.
        # Title punctuation varies wildly — keep alphanumerics + dots.
        title_tokens = {
            t for t in _TITLE_TOKEN_RE.findall(title_lower) if len(t) >= 2
        }
        if not title_tokens:
            continue
        for kw in keywords:
            if not kw:
                continue
            kw_tokens = {
                t for t in _TITLE_TOKEN_RE.findall(kw.lower()) if len(t) >= 2
            }
            if not kw_tokens:
                continue
            # Match tokens on exact equality AND on prefix (so "tool"
            # can match "tools.py", "tool_voice", "toolbar").
            matched = 0
            for kt in kw_tokens:
                if kt in title_tokens:
                    matched += 1
                else:
                    # Prefix match: title token starts with kw token
                    # (e.g. user said "tool", title has "tool_voice")
                    for tt in title_tokens:
                        if tt.startswith(kt) or kt.startswith(tt):
                            matched += 1
                            break
            # Normalize: fraction of kw tokens matched, weighted toward
            # the smaller side so a short query ("chrome") against a
            # long title ("Google Chrome - foo bar baz") still scores
            # high when the single token matches.
            denom = min(len(kw_tokens), len(title_tokens)) or 1
            score = matched / denom
            if score > best_score:
                best_score = score
                best_line = line

    return best_line if best_score >= _FUZZY_THRESHOLD else ""


# Token regex: split at any non-alphanumeric character (including . _ -).
# "tools.py" → {"tools", "py"}; "tool_voice.py" → {"tool", "voice", "py"};
# "Visual Studio Code" → {"visual", "studio", "code"}. Splitting on . and _
# is the key to matching "tools.py" against a window whose title has
# "tool_voice.py" — both share {"py"} as a token, and "tool" prefixes
# "tools".
_TITLE_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Below this score, fuzzy matching refuses to guess — better to tell
# the user "no match" than to focus the wrong window.
_FUZZY_THRESHOLD = 0.5


def _resolve_window_id(app_name: str) -> tuple[str, str, str]:
    """Look up ``app_name`` and return ``(win_id, title, matched_keyword)``.

    Tries wmctrl exact-match first (fast), then xdotool search, then
    fuzzy token-overlap via :func:`_find_best_window`. Returns
    ``("", "", "")`` if nothing matches.

    Used by close / minimize / focus / etc. so they all benefit from
    the fuzzy upgrade without copy-pasting the resolution logic.
    """
    if not app_name or not _check_tool("wmctrl"):
        return "", "", ""

    name_lower = app_name.lower().strip()
    keywords   = _APP_ALIASES.get(name_lower, [name_lower])

    windows_out = _run_output(["wmctrl", "-l"])
    matched     = _find_best_window(windows_out, keywords)
    if not matched:
        return "", "", ""

    parts = matched.split(None, 3)
    win_id = parts[0]
    title  = parts[3] if len(parts) >= 4 else app_name
    # Find which keyword actually matched — purely for the response message.
    matched_kw = next(
        (kw for kw in keywords if kw in matched.lower() or any(kw in t.lower() for t in [title])),
        keywords[0],
    )
    return win_id, title, matched_kw


def close_window(app_name: str) -> str:
    """Gracefully close a window matching app_name."""
    if not app_name:
        return "No app name provided."

    name_lower = app_name.lower().strip()
    keywords   = _APP_ALIASES.get(name_lower, [name_lower])

    # Pass 1: wmctrl exact substring — fast path.
    if _check_tool("wmctrl"):
        for keyword in keywords:
            result = subprocess.run(
                ["wmctrl", "-c", keyword],
                capture_output=True
            )
            if result.returncode == 0:
                return f"Closed: {app_name} (matched '{keyword}')"

    # Pass 2: fuzzy match + xdotool windowclose.
    win_id, title, matched_kw = _resolve_window_id(app_name)
    if win_id and _check_tool("xdotool"):
        subprocess.run(["xdotool", "windowclose", win_id], capture_output=True)
        return f"Closed: {title} (matched '{matched_kw}')"

    # Pass 3: xdotool search by name (covers windows wmctrl didn't list,
    # e.g. un-decorated ones).
    if _check_tool("xdotool"):
        for keyword in keywords:
            out = _run_output([
                "xdotool", "search", "--onlyvisible", "--name", keyword
            ])
            if out:
                subprocess.run(
                    ["xdotool", "windowclose", out.splitlines()[0].strip()],
                    capture_output=True,
                )
                return f"Closed: {app_name} (via xdotool)"

    return f"Could not close '{app_name}' — window not found or no tool available."


def minimize_window(app_name: str) -> str:
    """Minimize a window matching app_name."""
    if not _check_tool("xdotool"):
        return "xdotool is required for minimize. Install: sudo apt install xdotool"

    name_lower = app_name.lower().strip()
    keywords   = _APP_ALIASES.get(name_lower, [name_lower])

    # Pass 1: xdotool exact name search.
    for keyword in keywords:
        out = _run_output([
            "xdotool", "search", "--onlyvisible", "--name", keyword
        ])
        if out:
            subprocess.run(
                ["xdotool", "windowminimize", out.splitlines()[0].strip()],
                capture_output=True
            )
            return f"Minimized: {app_name} (matched '{keyword}')"

    # Pass 2: fuzzy match via wmctrl list + xdotool close-by-id.
    win_id, title, matched_kw = _resolve_window_id(app_name)
    if win_id:
        subprocess.run(["xdotool", "windowminimize", win_id], capture_output=True)
        return f"Minimized: {title} (matched '{matched_kw}')"

    return f"Could not minimize '{app_name}' — window not found."


def maximize_window(app_name: str) -> str:
    """Toggle maximize on a window matching app_name."""
    if not _check_tool("wmctrl"):
        return "wmctrl is required for maximize. Install: sudo apt install wmctrl"

    name_lower = app_name.lower().strip()
    keywords   = _APP_ALIASES.get(name_lower, [name_lower])

    windows = _run_output(["wmctrl", "-l"])
    matched = _find_best_window(windows, keywords)
    if matched:
        win_id = matched.split()[0]
        subprocess.run(
            ["wmctrl", "-ir", win_id, "-b", "toggle,maximized_vert,maximized_horz"],
            capture_output=True
        )
        return f"Toggled maximize: {app_name}"

    return f"Could not maximize '{app_name}' — window not found."


# ─────────────────────────────────────────────────────────────────────────────
# Virtual desktop (workspace) management
# ─────────────────────────────────────────────────────────────────────────────

# Human-readable workspace name aliases → 0-based index
# Users say "desktop 1" or "workspace 1" — internally wmctrl uses 0-based index
_WORKSPACE_ALIASES: dict[str, int] = {
    "1": 0, "one":   0, "first":  0, "desktop 1": 0, "workspace 1": 0,
    "2": 1, "two":   1, "second": 1, "desktop 2": 1, "workspace 2": 1,
    "3": 2, "three": 2, "third":  2, "desktop 3": 2, "workspace 3": 2,
    "4": 3, "four":  3, "fourth": 3, "desktop 4": 3, "workspace 4": 3,
    "5": 4, "five":  4, "fifth":  4, "desktop 5": 4, "workspace 5": 4,
    "6": 5, "six":   5, "sixth":  5, "desktop 6": 5, "workspace 6": 5,
}


def _resolve_workspace(name: str) -> int | None:
    """
    Convert a human workspace name/number to a 0-based int.
    Accepts: "1", "desktop 2", "workspace 3", "second", etc.
    Returns None if unresolvable.
    """
    cleaned = name.lower().strip()
    if cleaned in _WORKSPACE_ALIASES:
        return _WORKSPACE_ALIASES[cleaned]
    # bare integer string like "0" or "3"
    try:
        return int(cleaned)
    except ValueError:
        return None


def list_workspaces() -> str:
    """List all virtual desktops and which one is currently active."""
    if not _check_tool("wmctrl"):
        return "wmctrl is not installed. Run: sudo apt install wmctrl"

    out = _run_output(["wmctrl", "-d"])
    if not out:
        return "Could not retrieve workspace info."

    lines = []
    for line in out.splitlines():
        parts = line.split(None, 9)
        if len(parts) < 2:
            continue
        idx    = parts[0]
        active = parts[1] == "*"
        name   = parts[-1] if len(parts) >= 10 else f"Workspace {int(idx) + 1}"
        marker = " ◀ current" if active else ""
        lines.append(f"  Desktop {int(idx) + 1} — {name}{marker}")

    return "Virtual desktops:\n" + "\n".join(lines)


def switch_workspace(target: str) -> str:
    """Switch the active virtual desktop to `target`."""
    if not _check_tool("wmctrl"):
        return "wmctrl is not installed. Run: sudo apt install wmctrl"

    idx = _resolve_workspace(target)
    if idx is None:
        return (
            f"Could not understand workspace '{target}'. "
            "Use: '1', '2', 'desktop 3', 'workspace 4', etc."
        )

    result = subprocess.run(
        ["wmctrl", "-s", str(idx)],
        capture_output=True
    )
    if result.returncode == 0:
        return f"Switched to desktop {idx + 1}."
    return f"Could not switch to desktop {idx + 1} — does it exist? Use 'list_workspaces' to check."


def move_window_to_workspace(app_name: str, target: str) -> str:
    """
    Move a window matching `app_name` to virtual desktop `target`.
    Optionally also switches focus to that desktop.

    Examples:
        move_window_to_workspace("brave", "2")
        move_window_to_workspace("terminal", "desktop 4")
    """
    if not _check_tool("wmctrl"):
        return "wmctrl is not installed. Run: sudo apt install wmctrl"

    if not app_name:
        return "No app name provided."

    idx = _resolve_workspace(target)
    if idx is None:
        return (
            f"Could not understand workspace '{target}'. "
            "Use: '1', '2', 'desktop 3', 'workspace 4', etc."
        )

    name_lower = app_name.lower().strip()
    keywords   = _APP_ALIASES.get(name_lower, [name_lower])

    # Get full window list so we can match by title
    windows_out = _run_output(["wmctrl", "-l"])
    matched     = _find_best_window(windows_out, keywords)

    if not matched:
        return (
            f"No open window matched '{app_name}'. "
            f"Tried: {keywords}. Use 'list_windows' to see exact titles."
        )

    win_id = matched.split()[0]
    title  = matched.split(None, 3)[-1].strip() if len(matched.split(None, 3)) >= 4 else app_name

    # wmctrl -ir <win_id> -t <desktop_index>  →  move window to desktop
    result = subprocess.run(
        ["wmctrl", "-ir", win_id, "-t", str(idx)],
        capture_output=True
    )

    if result.returncode != 0:
        return f"Failed to move '{title}' to desktop {idx + 1}."

    return f"Moved '{title}' to desktop {idx + 1}."


def move_window_to_workspace_and_follow(app_name: str, target: str) -> str:
    """Move window to workspace AND switch focus there — so you follow the app."""
    move_result = move_window_to_workspace(app_name, target)
    if "Moved" not in move_result:
        return move_result  # something failed, return the error

    switch_result = switch_workspace(target)
    return f"{move_result}\n{switch_result}"


def get_window_workspace(app_name: str) -> str:
    """Report which virtual desktop a window is currently on."""
    if not _check_tool("wmctrl"):
        return "wmctrl is not installed. Run: sudo apt install wmctrl"

    name_lower = app_name.lower().strip()
    keywords   = _APP_ALIASES.get(name_lower, [name_lower])

    windows_out = _run_output(["wmctrl", "-l"])
    matched     = _find_best_window(windows_out, keywords)

    if not matched:
        return f"No open window matched '{app_name}'."

    parts          = matched.split(None, 3)
    desktop_idx    = int(parts[1])
    title          = parts[3].strip() if len(parts) >= 4 else app_name
    desktop_human  = desktop_idx + 1

    # -1 means the window is "sticky" (visible on all desktops)
    if desktop_idx == -1:
        return f"'{title}' is sticky — visible on all desktops."

    return f"'{title}' is on desktop {desktop_human}."


# ─────────────────────────────────────────────────────────────────────────────
# Desktop file management
# ─────────────────────────────────────────────────────────────────────────────

FILE_TYPE_MAP: dict[str, set[str]] = {
    "Images":      {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
                    ".svg", ".ico", ".heic", ".tiff"},
    "Documents":   {".pdf", ".doc", ".docx", ".txt", ".xls", ".xlsx",
                    ".ppt", ".pptx", ".csv", ".odt", ".ods", ".odp", ".md"},
    "Videos":      {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv",
                    ".webm", ".m4v"},
    "Music":       {".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a"},
    "Archives":    {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"},
    "Code":        {".py", ".js", ".ts", ".html", ".css", ".json", ".xml",
                    ".cpp", ".c", ".h", ".java", ".cs", ".go", ".rs",
                    ".sh", ".php", ".rb", ".lua", ".toml", ".yaml", ".yml"},
    "Executables": {".sh", ".appimage", ".deb", ".rpm", ".run"},
}

# Linux desktop launchers — skip these when organizing/cleaning
_SKIP_EXTENSIONS = {".desktop"}


def organize_desktop(mode: str = "by_type") -> str:
    desktop = _get_desktop()
    moved, skipped = [], []

    for item in sorted(desktop.iterdir()):
        if item.is_dir() or item.name.startswith("."):
            continue
        if item.suffix.lower() in _SKIP_EXTENSIONS:
            continue

        if mode == "by_date":
            mtime       = datetime.fromtimestamp(item.stat().st_mtime)
            folder_name = mtime.strftime("%Y-%m")
        else:
            ext         = item.suffix.lower()
            folder_name = "Others"
            for folder, exts in FILE_TYPE_MAP.items():
                if ext in exts:
                    folder_name = folder
                    break

        target_dir = desktop / folder_name
        target_dir.mkdir(exist_ok=True)
        new_path = target_dir / item.name

        if new_path.exists():
            skipped.append(item.name)
            continue

        shutil.move(str(item), str(new_path))
        moved.append(f"  {item.name} → {folder_name}/")

    lines = [f"Desktop organized ({mode}): {len(moved)} file(s) moved."]
    if moved:
        lines += moved[:10]
        if len(moved) > 10:
            lines.append(f"  ... and {len(moved) - 10} more.")
    if skipped:
        lines.append(f"{len(skipped)} file(s) skipped (name conflict).")
    return "\n".join(lines)


def clean_desktop() -> str:
    desktop     = _get_desktop()
    today       = datetime.now().strftime("%Y-%m-%d")
    archive_dir = desktop / f"Desktop Archive {today}"
    archive_dir.mkdir(exist_ok=True)
    moved = 0

    for item in desktop.iterdir():
        if item.is_dir() or item.name.startswith("."):
            continue
        if item.suffix.lower() in _SKIP_EXTENSIONS:
            continue
        dest = archive_dir / item.name
        if not dest.exists():
            shutil.move(str(item), str(dest))
            moved += 1

    return f"Desktop cleaned: {moved} file(s) archived to '{archive_dir.name}'."


def list_desktop() -> str:
    desktop = _get_desktop()
    items   = []

    for item in sorted(desktop.iterdir()):
        if item.name.startswith("."):
            continue
        if item.is_dir():
            try:
                count = sum(1 for _ in item.iterdir())
            except PermissionError:
                count = "?"
            items.append(f"📁 {item.name}/ ({count} items)")
        else:
            size = item.stat().st_size
            size_str = (
                f"{size / 1_048_576:.1f} MB" if size >= 1_048_576
                else f"{size / 1024:.1f} KB"
            )
            items.append(f"📄 {item.name} ({size_str})")

    if not items:
        return "Desktop is empty."
    return f"Desktop — {len(items)} item(s):\n" + "\n".join(items)


def get_desktop_stats() -> str:
    desktop = _get_desktop()
    files   = [i for i in desktop.iterdir() if i.is_file()]
    folders = [i for i in desktop.iterdir() if i.is_dir()]
    total   = sum(f.stat().st_size for f in files if f.exists())
    size_str = (
        f"{total / 1_048_576:.1f} MB" if total >= 1_048_576
        else f"{total / 1024:.1f} KB"
    )
    de = _detect_de() or "unknown"
    return (
        f"Desktop stats:\n"
        f"  DE      : {de}\n"
        f"  Files   : {len(files)}\n"
        f"  Folders : {len(folders)}\n"
        f"  Size    : {size_str}\n"
        f"  Path    : {desktop}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# AI-powered task execution (sandboxed)
# ─────────────────────────────────────────────────────────────────────────────

class _JarvisInput:
    """
    Thin pyautogui wrapper exposed to AI-generated code.
    Only the actions Jarvis actually needs — no raw screen access.
    """
    def __init__(self) -> None:
        if not _HAS_PYAUTOGUI:
            raise RuntimeError("pyautogui is not installed.")

    def click(self, x: int, y: int) -> None:
        _pyautogui.click(x, y)

    def type_text(self, text: str, interval: float = 0.05) -> None:
        _pyautogui.typewrite(text, interval=interval)

    def hotkey(self, *keys: str) -> None:
        _pyautogui.hotkey(*keys)

    def screenshot(self) -> Any:
        return _pyautogui.screenshot()


def _build_sandbox() -> dict:
    import time  # noqa: PLC0415

    safe_builtins = {
        k: v for k, v in __builtins__.items()   # type: ignore[union-attr]
        if k in {
            "print", "len", "str", "int", "float", "bool",
            "list", "dict", "tuple", "set", "range", "enumerate",
            "sorted", "isinstance", "hasattr", "getattr",
            "max", "min", "sum", "abs", "zip", "map", "filter",
            "round", "any", "all",
        }
    } if isinstance(__builtins__, dict) else {
        name: getattr(__builtins__, name)
        for name in [
            "print", "len", "str", "int", "float", "bool",
            "list", "dict", "tuple", "set", "range", "enumerate",
            "sorted", "isinstance", "hasattr", "getattr",
            "max", "min", "sum", "abs", "zip", "map", "filter",
            "round", "any", "all",
        ]
        if hasattr(__builtins__, name)
    }

    sandbox: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "Path": Path,
        "time": time,
        "shutil": type("_Shutil", (), {
            "copy2":      staticmethod(shutil.copy2),
            "copytree":   staticmethod(shutil.copytree),
            "disk_usage": staticmethod(shutil.disk_usage),
        })(),
        "os_path": os.path,
        "desktop": str(_get_desktop()),
    }

    if _HAS_PYAUTOGUI:
        sandbox["jarvis_input"] = _JarvisInput()

    return sandbox


def _strip_fences(code: str) -> str:
    if code.startswith("```"):
        lines = code.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return code.strip()


def _execute_sandboxed(code: str, player=None) -> str:
    code = _strip_fences(code)
    if not code or code == "UNSAFE":
        return "This action cannot be performed safely with the available tools."

    sandbox      = _build_sandbox()
    output_lines: list[str] = []
    sandbox["__builtins__"]["print"] = lambda *a, **_: output_lines.append(
        " ".join(str(x) for x in a)
    )

    try:
        exec(compile(code, "<jarvis_desktop>", "exec"), sandbox)  # noqa: S102
        return "\n".join(output_lines) if output_lines else "Done."
    except Exception as e:
        # Log truncated code for debugging — never log user data / keys
        _log(f"[Desktop] Exec error: {e} | code snippet: {code[:120]!r}")
        return f"Execution error: {e}"


def _ask_gemini(task: str) -> str:
    from google import genai as _genai  # noqa: PLC0415

    client  = _genai.Client(api_key=_get_api_key())
    desktop = str(_get_desktop())
    de      = _detect_de() or "unknown"

    prompt = f"""You are a Linux desktop automation assistant for Jarvis.
Desktop environment : {de}
Desktop path        : {desktop}

Generate safe Python code to accomplish the task below.

Allowed names (pre-injected, no imports needed):
  Path          — pathlib.Path  (read + copy only, NO unlink/rmdir)
  shutil        — .copy2(), .copytree(), .disk_usage()  (NO move, NO rmtree)
  os_path       — os.path (read-only)
  time          — time.sleep only
  jarvis_input  — .click(x,y), .type_text(str), .hotkey(*keys)  [if needed]
  desktop       — str path to the desktop folder

Hard rules:
  - NO import statements
  - NO file deletion (no unlink, rmdir, rmtree, remove)
  - NO subprocess or os.system calls
  - NO exec() or eval() inside the generated code
  - NO write operations unless the task explicitly requests saving a file
  - If the task cannot be done safely, output exactly: UNSAFE

Output ONLY valid Python code. No explanation, no markdown, no backticks.

Task: {task}"""

    try:
        resp = client.models.generate_content(
            model=os.getenv(
                "GEMINI_MODEL",
                os.getenv("gemini-flash-lite-latest")
                .split(",")[0],
            ),
            contents=prompt,
        )
        return resp.text.strip()
    except Exception as e:
        return f"ERROR: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, capture_output=True, check=False)


def _run_output(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.stdout.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def desktop_control(
    parameters: dict | None = None,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    """
    Main entry point called by Jarvis orchestration layer.

    parameters keys:
        action  : one of the action strings listed in the module docstring
        path    : local image path  (for 'wallpaper')
        url     : image URL         (for 'wallpaper_url')
        mode    : 'by_type' | 'by_date'  (for 'organize')
        task    : natural-language description  (for 'task')
        confirm : bool — if True, skip confirmation prompt for AI tasks
    """
    params = parameters or {}
    action = params.get("action", "").lower().strip()
    task   = params.get("task", "").strip()

    if player:
        player.write_log(f"[desktop] action={action or 'task'}")

    try:
        if action == "wallpaper":
            path = params.get("path", "")
            return set_wallpaper(path) if path else "No image path provided."

        if action == "wallpaper_url":
            url = params.get("url", "")
            return set_wallpaper_from_url(url) if url else "No URL provided."

        if action == "current_wallpaper":
            return get_current_wallpaper()

        if action == "organize":
            return organize_desktop(params.get("mode", "by_type"))

        if action == "clean":
            return clean_desktop()

        if action == "list":
            return list_desktop()

        if action == "stats":
            return get_desktop_stats()

        if action == "list_windows":
            return list_windows()

        if action == "focus":
            app = params.get("app", "") or task
            return focus_window(app)

        if action == "close_window":
            app = params.get("app", "") or task
            return close_window(app)

        if action == "minimize":
            app = params.get("app", "") or task
            return minimize_window(app)

        if action == "maximize":
            app = params.get("app", "") or task
            return maximize_window(app)

        if action == "list_workspaces":
            return list_workspaces()

        if action == "switch_workspace":
            target = params.get("target", "") or params.get("workspace", "") or task
            return switch_workspace(target)

        if action == "move_to_workspace":
            app    = params.get("app", "")
            target = params.get("target", "") or params.get("workspace", "")
            follow = params.get("follow", False)   # if True, switch to that desktop too
            if not app or not target:
                return "Provide both 'app' and 'target' (e.g. app='brave', target='2')."
            if follow:
                return move_window_to_workspace_and_follow(app, target)
            return move_window_to_workspace(app, target)

        if action == "window_workspace":
            app = params.get("app", "") or task
            return get_window_workspace(app)

        if action == "task" or task:
            actual_task = task or params.get("description", "")
            if not actual_task:
                return "Please describe what you want to do on the desktop."

            _log(f"[Desktop] Gemini task: {actual_task}")
            if player:
                player.write_log("[Desktop] Generating action...")

            # Optional confirmation gate (set confirm=False to require approval)
            if not params.get("confirm", True):
                if player and hasattr(player, "confirm"):
                    if not player.confirm(f"Run desktop task: {actual_task}?"):
                        return "Task cancelled by user."

            code = _ask_gemini(actual_task)
            return _execute_sandboxed(code, player=player)

        # Fallback: treat unknown action string as a natural-language task
        if action:
            code = _ask_gemini(action)
            return _execute_sandboxed(code, player=player)

        return "No action or task specified."

    except Exception as e:
        _log(f"[Desktop] Unhandled error: {e}")
        return f"Desktop control error: {e}"