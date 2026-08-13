import re
import time
import subprocess
import platform
import shutil

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

_SYSTEM = platform.system()

_APP_ALIASES: dict[str, dict[str, str]] = {

    "chrome":             {"Windows": "chrome",                  "Darwin": "Google Chrome",        "Linux": "google-chrome"},
    "google chrome":      {"Windows": "chrome",                  "Darwin": "Google Chrome",        "Linux": "google-chrome"},
    "firefox":            {"Windows": "firefox",                 "Darwin": "Firefox",              "Linux": "firefox"},
    "edge":               {"Windows": "msedge",                  "Darwin": "Microsoft Edge",       "Linux": "microsoft-edge"},
    "brave":              {"Windows": "brave",                   "Darwin": "Brave Browser",        "Linux": "brave-browser"},
    "safari":             {"Windows": "msedge",                  "Darwin": "Safari",               "Linux": "firefox"},
    "opera":              {"Windows": "opera",                   "Darwin": "Opera",                "Linux": "opera"},
    "whatsapp":           {"Windows": "WhatsApp",                "Darwin": "WhatsApp",             "Linux": "whatsapp"},
    "telegram":           {"Windows": "Telegram",                "Darwin": "Telegram",             "Linux": "telegram"},
    "discord":            {"Windows": "Discord",                 "Darwin": "Discord",              "Linux": "discord"},
    "slack":              {"Windows": "Slack",                   "Darwin": "Slack",                "Linux": "slack"},
    "zoom":               {"Windows": "Zoom",                    "Darwin": "zoom.us",              "Linux": "zoom"},
    "teams":              {"Windows": "msteams",                 "Darwin": "Microsoft Teams",      "Linux": "teams"},
    "skype":              {"Windows": "skype",                   "Darwin": "Skype",                "Linux": "skype"},
    "signal":             {"Windows": "signal",                  "Darwin": "Signal",               "Linux": "signal"},
    "spotify":            {"Windows": "Spotify",                 "Darwin": "Spotify",              "Linux": "spotify"},
    "vlc":                {"Windows": "vlc",                     "Darwin": "VLC",                  "Linux": "vlc"},
    "netflix":            {"Windows": "Netflix",                 "Darwin": "Netflix",              "Linux": "firefox"},
    "vscode":             {"Windows": "code",                    "Darwin": "Visual Studio Code",   "Linux": "code"},
    "vs code":            {"Windows": "code",                    "Darwin": "Visual Studio Code",   "Linux": "code"},
    "visual studio code": {"Windows": "code",                    "Darwin": "Visual Studio Code",   "Linux": "code"},
    "code":               {"Windows": "code",                    "Darwin": "Visual Studio Code",   "Linux": "code"},
    "terminal":           {"Windows": "wt",                      "Darwin": "Terminal",             "Linux": "gnome-terminal"},
    "cmd":                {"Windows": "cmd.exe",                 "Darwin": "Terminal",             "Linux": "bash"},
    "powershell":         {"Windows": "powershell.exe",          "Darwin": "Terminal",             "Linux": "bash"},
    "postman":            {"Windows": "Postman",                 "Darwin": "Postman",              "Linux": "postman"},
    "git":                {"Windows": "git-bash",                "Darwin": "Terminal",             "Linux": "bash"},
    "figma":              {"Windows": "Figma",                   "Darwin": "Figma",                "Linux": "figma"},
    "blender":            {"Windows": "blender",                 "Darwin": "Blender",              "Linux": "blender"},
    "word":               {"Windows": "winword",                 "Darwin": "Microsoft Word",       "Linux": "libreoffice --writer"},
    "excel":              {"Windows": "excel",                   "Darwin": "Microsoft Excel",      "Linux": "libreoffice --calc"},
    "powerpoint":         {"Windows": "powerpnt",                "Darwin": "Microsoft PowerPoint", "Linux": "libreoffice --impress"},
    "libreoffice":        {"Windows": "soffice",                 "Darwin": "LibreOffice",          "Linux": "libreoffice"},
    "notepad":            {"Windows": "notepad.exe",             "Darwin": "TextEdit",             "Linux": "gedit"},
    "textedit":           {"Windows": "notepad.exe",             "Darwin": "TextEdit",             "Linux": "gedit"},
    "explorer":           {"Windows": "explorer.exe",            "Darwin": "Finder",               "Linux": "nautilus"},
    "file explorer":      {"Windows": "explorer.exe",            "Darwin": "Finder",               "Linux": "nautilus"},
    "finder":             {"Windows": "explorer.exe",            "Darwin": "Finder",               "Linux": "nautilus"},
    "files":              {"Windows": "explorer.exe",            "Darwin": "Finder",               "Linux": "nautilus"},
    "file manager":       {"Windows": "explorer.exe",            "Darwin": "Finder",               "Linux": "nautilus"},
    "file browser":       {"Windows": "explorer.exe",            "Darwin": "Finder",               "Linux": "nautilus"},
    "nautilus":           {"Windows": "explorer.exe",            "Darwin": "Finder",               "Linux": "nautilus"},
    "task manager":       {"Windows": "taskmgr.exe",             "Darwin": "Activity Monitor",     "Linux": "gnome-system-monitor"},
    "settings":           {"Windows": "ms-settings:",            "Darwin": "System Preferences",   "Linux": "gnome-control-center"},
    "calculator":         {"Windows": "calc.exe",                "Darwin": "Calculator",           "Linux": "gnome-calculator"},
    "paint":              {"Windows": "mspaint.exe",             "Darwin": "Preview",              "Linux": "gimp"},
    "instagram":          {"Windows": "Instagram",               "Darwin": "Instagram",            "Linux": "firefox"},
    "tiktok":             {"Windows": "TikTok",                  "Darwin": "TikTok",               "Linux": "firefox"},
    "notion":             {"Windows": "Notion",                  "Darwin": "Notion",               "Linux": "notion"},
    "obsidian":           {"Windows": "Obsidian",                "Darwin": "Obsidian",             "Linux": "obsidian"},
    "capcut":             {"Windows": "CapCut",                  "Darwin": "CapCut",               "Linux": "capcut"},
    "steam":              {"Windows": "steam",                   "Darwin": "Steam",                "Linux": "steam"},
    "epic":               {"Windows": "EpicGamesLauncher",       "Darwin": "Epic Games Launcher",  "Linux": "legendary"},
    "epic games":         {"Windows": "EpicGamesLauncher",       "Darwin": "Epic Games Launcher",  "Linux": "legendary"},
}


def _normalize(raw: str) -> str:
    key = " ".join(raw.lower().split())

    if key in _APP_ALIASES:
        return _APP_ALIASES[key].get(_SYSTEM, raw)

    # Longest whole-word match wins, so "google chrome browser" resolves
    # through "google chrome" rather than whichever alias happened to be
    # first, and a fragment like "co" no longer resolves to "code".
    best_key: str | None = None
    best_map: dict[str, str] | None = None
    for alias_key, os_map in _APP_ALIASES.items():
        if re.search(rf"\b{re.escape(alias_key)}\b", key):
            if best_key is None or len(alias_key) > len(best_key):
                best_key, best_map = alias_key, os_map
    if best_map is not None:
        return best_map.get(_SYSTEM, raw)

    return raw


def _launch_windows(app_name: str, args: list[str] | None = None) -> bool:
    args = list(args or [])

    binary = shutil.which(app_name) or shutil.which(app_name.split(".")[0])
    if binary:
        try:
            # List form, never a shell string: a target path may contain
            # spaces, and quoting it back into a command line is exactly
            # the sort of thing that silently drops the argument.
            subprocess.Popen(
                [binary, *args],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(1.5)
            return True
        except Exception as e:
            print(f"[open_app] subprocess failed: {e}")

    if ":" in app_name and not args:
        try:
            subprocess.Popen(["cmd", "/c", "start", "", app_name])
            time.sleep(1.0)
            return True
        except Exception:
            pass

    # The Start Menu fallback types a name; it has no way to carry a path,
    # so a request that named one must fail rather than open a blank app.
    if args:
        return False

    try:
        import pyautogui
        pyautogui.PAUSE = 0.1
        pyautogui.press("win")
        time.sleep(0.7)
        pyautogui.write(app_name, interval=0.05)
        time.sleep(0.9)
        pyautogui.press("enter")
        time.sleep(2.5)
        return True
    except Exception as e:
        print(f"[open_app] Start Menu search failed: {e}")

    return False


def _launch_macos(app_name: str, args: list[str] | None = None) -> bool:
    args = list(args or [])

    for name in (app_name, f"{app_name}.app"):
        try:
            result = subprocess.run(
                ["open", "-a", name, *args],
                capture_output=True, timeout=8
            )
            if result.returncode == 0:
                time.sleep(1.0)
                return True
        except Exception:
            pass

    binary = shutil.which(app_name) or shutil.which(app_name.lower())
    if binary:
        try:
            subprocess.Popen(
                [binary, *args],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(1.0)
            return True
        except Exception:
            pass

    if args:
        return False

    try:
        import pyautogui
        pyautogui.hotkey("command", "space")
        time.sleep(0.6)
        pyautogui.write(app_name, interval=0.05)
        time.sleep(0.8)
        pyautogui.press("enter")
        time.sleep(1.5)
        return True
    except Exception as e:
        print(f"[open_app] Spotlight failed: {e}")

    return False


def _resolve_linux_binary(app_name: str) -> tuple[str, list[str]] | None:
    """The executable for *app_name*, plus any flags the alias carries.

    Some aliases are a command line rather than a binary
    ("libreoffice --writer"), which ``shutil.which`` can never find as a
    single name.
    """
    tokens = app_name.split()
    if len(tokens) > 1 and tokens[1].startswith("-"):
        found = shutil.which(tokens[0])
        if found:
            return found, tokens[1:]

    for candidate in (
        app_name,
        app_name.lower(),
        app_name.lower().replace(" ", "-"),
        app_name.lower().replace(" ", "_"),
    ):
        found = shutil.which(candidate)
        if found:
            return found, []
    return None


def _launch_linux(app_name: str, args: list[str] | None = None) -> bool:
    args = list(args or [])

    resolved = _resolve_linux_binary(app_name)
    if resolved is not None:
        binary, flags = resolved
        try:
            subprocess.Popen(
                [binary, *flags, *args],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                # Detach: the app outlives the assistant process.
                start_new_session=True,
            )
            time.sleep(1.0)
            return True
        except Exception:
            pass

    # .desktop entries — gtk-launch passes trailing arguments to the app.
    for desktop_name in [
        app_name.lower(),
        app_name.lower().replace(" ", "-"),
        app_name.lower().replace(" ", ""),
    ]:
        try:
            result = subprocess.run(
                ["gtk-launch", desktop_name, *args],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass

    # Last resort: hand the target to whatever the desktop has registered
    # for it. With no target there is nothing meaningful to open — passing
    # the app's *name* to xdg-open only ever looked like it worked.
    if args:
        try:
            result = subprocess.run(
                ["xdg-open", args[0]], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            pass

    return False


_OS_LAUNCHERS = {
    "Windows": _launch_windows,
    "Darwin":  _launch_macos,
    "Linux":   _launch_linux,
}

from dataclasses import dataclass

@dataclass(slots=True)
class AppResult:
    success: bool
    message: str

_URL_RE = re.compile(r"^(?:https?://|www\.)", re.IGNORECASE)

# A bare domain — "google.com", "news.ycombinator.com/newest". Checked
# only after the path lookup fails, so a real file called notes.io wins.
_DOMAIN_RE = re.compile(
    r"^[\w-]+(?:\.[\w-]+)*\."
    r"(?:com|org|net|io|dev|ai|co|edu|gov|app|me|tv|in|np|uk|us|xyz|info|shop)"
    r"(?:/\S*)?$",
    re.IGNORECASE,
)


def _resolve_target(raw: str) -> tuple[str | None, str | None]:
    """Turn a spoken target into something to hand the app.

    Returns ``(argument, error)``. A URL is passed through; anything else
    is resolved as a path the same way the file tools resolve one, and
    must exist — opening an editor on a directory that isn't there is a
    silent no-op the user reads as "it just opened a blank window".
    """
    text = str(raw or "").strip().strip("\"'")
    if not text:
        return None, None

    if _URL_RE.match(text):
        return text if "://" in text else f"https://{text}", None

    # Same resolver the file tools use, so "kancha", "documents/notes"
    # and "~/kancha" all mean here what they mean there.
    from .file_controller import _resolve_path  # local: avoids a cycle

    path = _resolve_path(text)
    if path.exists():
        return str(path), None

    if _DOMAIN_RE.match(text):
        return f"https://{text}", None

    return None, f"I couldn't find {text} — {path} does not exist."


def _open_with_default_handler(target: str) -> bool:
    """Open *target* in whatever the desktop has registered for it."""
    opener = {"Windows": ["explorer"], "Darwin": ["open"]}.get(_SYSTEM, ["xdg-open"])
    try:
        result = subprocess.run(
            [*opener, target], capture_output=True, timeout=8
        )
        return result.returncode == 0
    except Exception:
        return False


def open_app(
    parameters=None,
    response=None,
    player=None,
    session_memory=None,
) -> AppResult:
    raw_target = ""
    if isinstance(parameters, dict):
        app_name = str(parameters.get("app_name") or "").strip()
        raw_target = str(
            parameters.get("target")
            or parameters.get("path")
            or parameters.get("location")
            or parameters.get("url")
            or ""
        ).strip()
    elif isinstance(parameters, str):
        app_name = parameters.strip()
    else:
        app_name = ""

    if not app_name:
        return AppResult(False, "No application name provided.")

    launcher = _OS_LAUNCHERS.get(_SYSTEM)
    if launcher is None:
        return AppResult(False, f"Unsupported operating system: {_SYSTEM}")

    argument, error = _resolve_target(raw_target)
    if error:
        return AppResult(False, error)
    args = [argument] if argument else []

    normalized = _normalize(app_name)
    print(f"[open_app] Launching: '{app_name}' → '{normalized}' {args} ({_SYSTEM})")

    if player:
        player.write_log(f"[open_app] {app_name} {argument or ''}".rstrip())

    where = f" with {argument}" if argument else ""

    try:
        if launcher(normalized, args):
            return AppResult(True, f"Launch command sent for {app_name}{where}.")
        if normalized.lower() != app_name.lower():
            if launcher(app_name, args):
                return AppResult(True, f"Launch command sent for {app_name}{where}.")

        # No such application — but "open kancha" often means a folder,
        # not a program. If the name is a real location, hand it to the
        # desktop's default handler instead of reporting a dead end.
        if not args:
            as_path, _ = _resolve_target(app_name)
            if as_path:
                if _open_with_default_handler(as_path):
                    return AppResult(True, f"Opened {as_path}.")

        return AppResult(
            False,
            f"Could not confirm that {app_name} launched. "
            f"It may still be loading, or it might not be installed."
        )
    except Exception as e:
        print(f"[open_app] Error: {e}")
        return AppResult(False, f"Failed to open {app_name}: {e}")
