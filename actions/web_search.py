"""actions/web_search.py

Live web search via Playwright (DuckDuckGo HTML) + Groq synthesis.

Flow
----
1. Check in-memory TTL cache.
2. Launch headless Chromium, search DuckDuckGo HTML, collect result
   titles + snippets.
3. Pass snippets to Groq (llama-3.1-8b-instant) which synthesises a
   short, plain-English answer. Falls back to the best raw snippet if
   Groq is unavailable.
4. Cache the answer for 5 minutes.

Threading note
--------------
sync_playwright() is greenlet-bound. A process-wide browser singleton
breaks when asyncio.to_thread picks a different worker thread
("Cannot switch to a different thread"). A fresh context is created
per call — always correct regardless of which thread runs the action.
"""

from __future__ import annotations

import glob
import hashlib
import logging
import os
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("kancha.actions.web_search")

# ── TTL cache ─────────────────────────────────────────────────────────────────

_CACHE_TTL_S: float = 300.0  # 5 minutes
_CACHE_MAX: int = 200

_cache: dict[str, tuple[str, float]] = {}  # key -> (answer, expires_monotonic)


def _ck(query: str) -> str:
    return hashlib.md5(" ".join(query.lower().split()).encode()).hexdigest()


def _cache_get(query: str) -> str | None:
    k = _ck(query)
    entry = _cache.get(k)
    if entry is None:
        return None
    answer, expires = entry
    if time.monotonic() < expires:
        return answer
    _cache.pop(k, None)
    return None


def _cache_put(query: str, answer: str) -> None:
    if len(_cache) >= _CACHE_MAX:
        _cache.pop(min(_cache, key=lambda k: _cache[k][1]), None)
    _cache[_ck(query)] = (answer, time.monotonic() + _CACHE_TTL_S)


# ── Groq singleton ────────────────────────────────────────────────────────────

_groq_client: Any = None
_groq_init_done: bool = False

_GROQ_SYSTEM = (
    "You are a concise factual assistant. "
    "Answer in 1-2  for simple and long for complex in plain English sentences using only the provided search snippets. "
    "State the answer directly — no markdown, no lists, no source citations, no preamble."
)


def _init_groq() -> None:
    global _groq_client, _groq_init_done
    if _groq_init_done:
        return

    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        logger.warning("web_search: no GROQ_API_KEY — raw snippets will be used")
        _groq_init_done = True
        return

    try:
        from groq import Groq  # noqa: PLC0415

        _groq_client = Groq(api_key=key)
        _groq_init_done = True
        logger.debug("web_search: Groq client ready")
    except Exception as exc:  # noqa: BLE001
        logger.warning("web_search: Groq init failed: %s", exc)
        _groq_init_done = True


def _synthesize_with_groq(query: str, results: list[dict[str, str]]) -> str | None:
    """Synthesise a short plain-English answer from raw search snippets.

    Uses Groq llama-3.1-8b-instant (<1 s on the free tier, 14.4K RPD,
    500K TPD — well within normal daily use). Falls back to the best raw
    snippet when Groq is unavailable.
    """
    _init_groq()

    lines: list[str] = []
    for r in results[:5]:
        title = r.get("title", "").strip()
        snippet = r.get("snippet", "").strip()
        if snippet:
            lines.append(f"{title}: {snippet}" if title else snippet)

    if not lines:
        return None

    if _groq_client is None:
        # No Groq — return the first snippet that looks like prose.
        for line in lines:
            if len(line) >= 30 and line[0].isupper():
                return line[:450]
        return lines[0][:450] if lines else None

    context = "\n".join(lines)
    try:
        resp = _groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _GROQ_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\nSearch snippets:\n{context}\n\nAnswer:"
                    ),
                },
            ],
            max_tokens=120,
            temperature=0.0,
        )
        answer = resp.choices[0].message.content.strip()
        return answer or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("web_search: Groq synthesis failed: %s — using raw snippet", exc)
        return lines[0][:450] if lines else None


# ── Playwright search ─────────────────────────────────────────────────────────


def _chrome_exe() -> str | None:
    """Find the best available Chromium executable in the Playwright cache."""
    shell = glob.glob(
        os.path.expanduser(
            "~/.cache/ms-playwright/chromium_headless_shell-*/"
            "chrome-headless-shell-linux64/chrome-headless-shell"
        )
    )
    full = glob.glob(
        os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome")
    )
    return (shell or full or [None])[0]


def _search_playwright(query: str) -> str | None:
    """Search DuckDuckGo HTML with Playwright and synthesise the answer with Groq."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415
    except ImportError:
        logger.warning("web_search: playwright not installed")
        return None

    # df=m restricts results to the past month so stale cached pages
    # (e.g. old Wikipedia snapshots) don't outrank current news.
    # Appending the year further biases DDG toward recent content.
    year = time.strftime("%Y")
    fresh_query = f"{query} {year}" if year not in query else query
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode(
        {"q": fresh_query, "df": "m"}
    )
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                executable_path=_chrome_exe(),
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            try:
                page = browser.new_page(
                    user_agent=(
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    )
                )
                page.goto(url, timeout=10_000, wait_until="domcontentloaded")

                results: list[dict[str, str]] = []
                for el in page.query_selector_all(".result.results_links")[:6]:
                    title_el = el.query_selector(".result__a")
                    snippet_el = el.query_selector(".result__snippet")
                    title = title_el.inner_text().strip() if title_el else ""
                    snippet = snippet_el.inner_text().strip() if snippet_el else ""
                    if snippet:
                        results.append({"title": title, "snippet": snippet})

                if not results:
                    logger.info("web_search: no snippets found for %r", query[:60])
                    return None

                logger.debug("web_search: collected %d snippets", len(results))
                return _synthesize_with_groq(query, results)
            finally:
                browser.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("web_search: Playwright failed: %s", exc)
        return None


# ── Public interface ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class SearchResult:
    success: bool
    message: str


def web_search(query: str) -> SearchResult:
    """Search the web and return a short plain-English answer.

    Always returns a SearchResult; never raises.
    """
    query = " ".join(query.strip().split())
    if not query:
        return SearchResult(False, "I need a search query, sir.")

    cached = _cache_get(query)
    if cached:
        logger.debug("web_search: cache hit for %r", query[:60])
        return SearchResult(True, cached)

    answer = _search_playwright(query)
    if answer:
        logger.info("web_search: answered (%d chars)", len(answer))
        _cache_put(query, answer)
        return SearchResult(True, answer)

    return SearchResult(
        False,
        "I couldn't find current results for that right now, sir. Try again in a moment.",
    )
