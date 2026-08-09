from __future__ import annotations

import glob
import hashlib
import logging
import os
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any

from numpy import str_

logger = logging.getLogger("kancha.actions.web_search")

# ── TTL cache ─────────────────────────────────────────────────────────────────

_CACHE_TTL_S: float = 60.0  # 1 minutes
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
    "Answer in 1-2  for simple and long for complex in plain English sentences using only the provided search snippets." \
    "respond like a human(jarvis), not an AI. "
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


def _synthesize_with_groq(query: str, results: list[dict[str, str]] | str) -> str | None:
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


# ── Gemini (Google Search grounding), rotating across 8 keys ────────────────

# Reads GEMINI_API_KEY_1 .. GEMINI_API_KEY_8 (skips any that are unset).
_GEMINI_KEY_ENV_VARS: list[str] = [f"GEMINI_API_KEY_{i}" for i in range(7, 0, -1)]

_gemini_last_good_idx: int = 0

_genai_module: Any = None
_genai_import_failed: bool = False


def _get_gemini_keys() -> list[str]:
    keys = []
    for var in _GEMINI_KEY_ENV_VARS:
        val = os.getenv(var, "").strip()
        if val:
            keys.append(val)
    return keys


def _get_genai_module() -> Any:
    global _genai_module, _genai_import_failed
    if _genai_module is not None or _genai_import_failed:
        return _genai_module
    try:
        from google import genai  # noqa: PLC0415

        _genai_module = genai
    except ImportError:
        logger.warning("web_search: google-genai not installed — skipping Gemini")
        _genai_import_failed = True
    return _genai_module


def _gemini_search(query: str) -> str | None:
    global _gemini_last_good_idx

    keys = _get_gemini_keys()
    if not keys:
        logger.debug(
            "web_search: no GEMINI_API_KEY_1.._8 set — skipping Gemini"
        )
        return None

    genai = _get_genai_module()
    if genai is None:
        return None

    n = len(keys)
    start = _gemini_last_good_idx % n

    for attempt in range(n):
        idx = (start + attempt) % n
        key = keys[idx]
        try:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=query,
                config={"tools": [{"google_search": {}}]},

            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "web_search: Gemini key #%d failed (%s) — trying next key",
                idx + 1,
                exc,
            )
            continue

        try:
            text = (response.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "web_search: Gemini key #%d response.text failed: %s",
                idx + 1,
                exc,
            )
            continue

        if not text:
            logger.info(
                "web_search: Gemini key #%d returned empty text — trying next key",
                idx + 1,
            )
            continue

        # This key worked — remember it so next call tries it first.
        _gemini_last_good_idx = idx
        #return text[:2000]  # hard cap — answers should be short
        return _synthesize_with_groq(query, text)

    logger.warning("web_search: all %d Gemini keys failed or returned empty", n)
    return None


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

    year = time.strftime("%Y")
    fresh_query = f"{query} {year}" if year not in query else query
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode(
        {"q": fresh_query, "df": "d"}
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

    # Primary: Gemini with Google Search grounding, rotating across keys.
    # answer = _gemini_search(query)
    # if answer:
    #     logger.info("web_search: answered by Gemini (%d chars)", len(answer))
    #     _cache_put(query, answer)
    #     return SearchResult(True, answer)

    # Fallback: Playwright + Groq synthesis.
    answer = _search_playwright(query)
    if answer:
        logger.info("web_search: answered by Playwright fallback (%d chars)", len(answer))
        _cache_put(query, answer)
        return SearchResult(True, answer)

    return SearchResult(
        False,
        "I couldn't find current results for that right now, sir. Try again in a moment.",
    )