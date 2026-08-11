#!/usr/bin/env python3
"""Probe every Gemini API key in .env and report its health.

Why this exists
---------------
KANCHA's multi-key pool (``reasoning/llm_client_mulapi.py``) rotates
across keys only when one fails, so a key that is invalid or shares a
project with another key silently weakens the pool. This script answers:

  * Is the key valid at all?
  * Which configured models can it actually serve?
  * Does it still serve embeddings (RAG depends on this)?
  * Is it currently rate-limited / quota-exhausted?
  * Can we tell whether two keys belong to the SAME project?

The last point is the important one: per-minute and per-day rate limits
are per PROJECT, not per key. Two keys under one project share one
budget — rotating between them gains nothing. When a quota error
includes a project number we print it, so keys reporting the same
project id are visibly coupled.

Usage
-----
    python key_health.py                 # probe all keys on configured models
    python key_health.py --no-embed      # skip embedding probes
    python key_health.py --model gemini-2.5-flash   # probe an extra model

It makes a handful of tiny "hi" calls per key (well within free tier).
It never prints key material — only length and the first 4 characters.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from reasoning.llm_client_mulapi import (  # noqa: E402
    ALL_MODELS,
    REQUEST_TIMEOUT,
    _is_invalid_key,
    _is_not_found,
    _is_quota_exhausted,
    _is_retryable,
    get_pool,
)

# text-embedding-004 was retired (NOT_FOUND on every key as of Aug 2026).
EMBEDDING_MODELS = ["gemini-embedding-001", "gemini-embedding-2"]

# Gemini quota errors often embed the project id, e.g.
#   "GenerateRequestsPerDayPerProject limit exceeded for project 123456789"
_PROJECT_RE = re.compile(r"project(?:s/| )?(\d{6,})", re.IGNORECASE)


def _mask(key: str) -> str:
    return f"{key[:4]}…({len(key)} chars)"


def _classify(exc: Exception) -> str:
    if _is_invalid_key(exc):
        return "INVALID_KEY"
    if _is_not_found(exc):
        return "NOT_FOUND"
    if _is_quota_exhausted(exc):
        return "QUOTA_EXHAUSTED"
    if _is_retryable(exc):
        return "RATE_LIMITED"
    return f"OTHER: {type(exc).__name__}"


def _project_hint(exc: Exception) -> str:
    match = _PROJECT_RE.search(str(exc))
    return f" (project {match.group(1)})" if match else ""


async def _probe(
    entry: Any, model: str, timeout: float, embed: bool
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        if embed:
            await asyncio.wait_for(
                asyncio.to_thread(
                    entry.client.models.embed_content,
                    model=model,
                    contents="hi",
                ),
                timeout=timeout,
            )
        else:
            await asyncio.wait_for(
                asyncio.to_thread(
                    entry.client.models.generate_content,
                    model=model,
                    contents="hi",
                ),
                timeout=timeout,
            )
        return {"ok": True, "ms": round((time.perf_counter() - t0) * 1000), "err": ""}
    except Exception as exc:  # noqa: BLE001 - diagnostics must not crash
        return {
            "ok": False,
            "ms": round((time.perf_counter() - t0) * 1000),
            "err": _classify(exc) + _project_hint(exc),
        }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-embed", action="store_true", help="skip embedding probes"
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="extra model to probe (repeatable); defaults to configured models",
    )
    parser.add_argument(
        "--timeout", type=float, default=REQUEST_TIMEOUT, help="per-call timeout"
    )
    args = parser.parse_args()

    pool = get_pool()
    entries = pool.entries()
    models = list(dict.fromkeys([*ALL_MODELS, *args.model]))
    do_embed = not args.no_embed

    print(f"Probing {len(entries)} key(s) × {len(models)} model(s)"
          f"{' + embeddings' if do_embed else ''} (timeout {args.timeout:.0f}s)\n")
    healthy = 0
    for entry in entries:
        tag = "💀 dead" if entry.dead else (
            "🔴 cooling" if not entry.is_available else "✅ available"
        )
        print(f"key[{entry.index}] {_mask(entry.key)}  {tag}")

        any_ok = False
        for model in models:
            res = await _probe(entry, model, args.timeout, embed=False)
            mark = "✅" if res["ok"] else "❌"
            print(f"    {mark} {model:<32} {res['ms']:>6}ms  {res['err']}")
            if res["ok"]:
                any_ok = True
                entry.mark_supported(model)

        if do_embed:
            for model in EMBEDDING_MODELS:
                res = await _probe(entry, model, args.timeout, embed=True)
                mark = "✅" if res["ok"] else "❌"
                print(f"    {mark} {model:<32} {res['ms']:>6}ms  {res['err']} (embed)")

        if any_ok:
            healthy += 1
        print()

    print(f"\nSummary: {healthy}/{len(entries)} keys served at least one model.")
    print(
        "Project coupling: keys whose quota errors show the SAME project id "
        "share one rate-limit budget. Keys that never error cannot be "
        "compared this way — verify in AI Studio that each belongs to a "
        "separate project for the multi-key split to multiply limits."
    )
    return 0 if healthy == len(entries) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
