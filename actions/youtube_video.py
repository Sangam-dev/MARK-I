from __future__ import annotations

import logging
import random
import re
import shutil
import subprocess
from typing import Any

logger = logging.getLogger("kancha.actions.youtube_video")

# How many results to ask YouTube for. One is enough to play the top
# hit; "random" needs a pool to choose from, and ten is deep enough to
# feel varied while still being one cheap request.
SEARCH_RESULTS = 10

# yt-dlp will happily wait a long time on a slow network. This sits in
# the assistant's response path, so cap it.
SEARCH_TIMEOUT_S = 20

# Leading filler that is instruction, not search terms. "Play me a
# song by X" and "song by X" should search for the same thing.
_LEAD_RE = re.compile(
    r"^\s*(?:hey\s+|ok(?:ay)?\s+)?(?:kancha|jarvis)?[,\s]*"
    r"(?:can\s+you\s+|could\s+you\s+|please\s+|)"
    r"(?:play|put\s+on|start|open|search\s+for|find|queue(?:\s+up)?)\s+"
    r"(?:me\s+|us\s+)?(?:a\s+|an\s+|the\s+|some\s+)?",
    re.IGNORECASE,
)

# Trailing filler: "…on youtube", "…please".
_TRAIL_RE = re.compile(
    r"\s*(?:on\s+youtube|in\s+youtube|from\s+youtube|please|for\s+me)\s*[.!?]*\s*$",
    re.IGNORECASE,
)

# "random" is an instruction about *which* result to take, not part of
# what to search for — searching for the word "random" would poison the
# query.
_RANDOM_RE = re.compile(r"\b(?:random|any|some\s+random|shuffle)\b", re.IGNORECASE)

_YOUTUBE_RE = re.compile(r"\b(?:on|from|in)\s+youtube\b", re.IGNORECASE)


def _error(message: str, **extra: Any) -> dict[str, Any]:
    """Structured failure. Same keys as success so callers need no branch."""
    logger.info("youtube_video: %s", message)
    return {
        "success": False,
        "error": message,
        "title": "",
        "url": "",
        "channel": "",
        **extra,
    }


def extract_query(request: str) -> tuple[str, bool]:
    """Turn a spoken request into (search query, wants a random pick).

    Deliberately light-handed. "song by The Weeknd" is a better YouTube
    query than "The Weeknd" — stripping down to a bare artist name loses
    the intent — so only the words that are pure instruction come out.
    """
    text = (request or "").strip()
    if not text:
        return "", False

    wants_random = bool(_RANDOM_RE.search(text))

    query = _LEAD_RE.sub("", text)
    query = _YOUTUBE_RE.sub(" ", query)
    query = _TRAIL_RE.sub("", query)
    if wants_random:
        query = _RANDOM_RE.sub(" ", query)
    # Articles left stranded by the removals ("play a random song" ->
    # " song") and doubled spaces.
    query = re.sub(r"^\s*(?:a|an|the|some)\s+", "", query, flags=re.IGNORECASE)
    query = re.sub(r"\s{2,}", " ", query).strip(" ,.!?")

    return query, wants_random


def _search(query: str, limit: int = SEARCH_RESULTS) -> list[dict[str, Any]]:
    """Return flat YouTube search results. Never downloads."""
    from yt_dlp import YoutubeDL  # noqa: PLC0415 — import cost only when used

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        # Metadata only: yt-dlp does not resolve each video page, so the
        # search is one request and no media is ever touched.
        "extract_flat": True,
        "socket_timeout": SEARCH_TIMEOUT_S,
        "noplaylist": True,
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)

    entries = (info or {}).get("entries") or []
    return [e for e in entries if isinstance(e, dict)]


def _usable(entry: dict[str, Any]) -> bool:
    """Skip results that would not actually play something."""
    if not entry.get("id"):
        return False
    # Flat search results carry a duration of None for live streams and
    # for entries that are really channels or playlists.
    if entry.get("_type") in ("playlist", "channel"):
        return False
    return True


def _url_for(entry: dict[str, Any]) -> str:
    url = entry.get("url") or entry.get("webpage_url") or ""
    if url.startswith("http"):
        return url
    return f"https://www.youtube.com/watch?v={entry['id']}"


def _open_in_browser(url: str) -> tuple[bool, str]:
    """Hand the URL to the desktop's default handler."""
    opener = shutil.which("xdg-open")
    if not opener:
        return False, "xdg-open is not available on this system"
    try:
        subprocess.Popen(
            [opener, url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        return False, f"could not open the browser: {exc}"
    return True, ""


def youtube_video(request: str) -> dict[str, Any]:
    """Search YouTube for *request* and open the best match in the browser.

    Returns ``{"success": True, "title", "url", "channel"}`` on success,
    or ``{"success": False, "error", ...}`` on any failure — network,
    no results, or no way to open a browser.
    """
    query, wants_random = extract_query(request)
    if not query:
        return _error("no search terms in that request")

    logger.info(
        "youtube_video: searching %r%s", query, " (random pick)" if wants_random else ""
    )

    try:
        entries = _search(query)
    except ImportError:
        return _error("yt-dlp is not installed — run 'uv add yt-dlp'")
    except Exception as exc:  # noqa: BLE001 — yt-dlp raises a wide family
        return _error(f"YouTube search failed: {exc}")

    candidates = [e for e in entries if _usable(e)]
    if not candidates:
        return _error(f"nothing found on YouTube for {query!r}")

    # "Random" picks from the top results rather than the whole page, so
    # a request for a random song by an artist still returns that artist.
    chosen = random.choice(candidates[:SEARCH_RESULTS]) if wants_random else candidates[0]

    url = _url_for(chosen)
    opened, failure = _open_in_browser(url)
    if not opened:
        return _error(
            failure,
            title=chosen.get("title") or "",
            url=url,
            channel=chosen.get("uploader") or chosen.get("channel") or "",
        )

    title = chosen.get("title") or "Untitled"
    channel = chosen.get("uploader") or chosen.get("channel") or ""
    logger.info("youtube_video: playing %r by %s", title, channel or "unknown")

    return {
        "success": True,
        "title": title,
        "url": url,
        "channel": channel,
    }
