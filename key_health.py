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

Logging
-------
Every probe (success or failure) is logged with full detail:
  * Console: human-readable one-liner per probe (as before).
  * Log file (default ``key_health.log``): structured JSON lines, one
    per probe attempt, including the full exception type, message,
    and traceback for failures — nothing is swallowed or summarized
    away. Use ``--log-file -`` to disable file logging and
    ``--verbose`` to also print full tracebacks to the console as
    they happen (not just at the end).

Usage
-----
    python key_health.py                          # probe all keys on configured models
    python key_health.py --no-embed                # skip embedding probes
    python key_health.py --model gemini-2.5-flash  # probe an extra model
    python key_health.py --verbose                 # print tracebacks to console too
    python key_health.py --log-file runs/probe.jsonl

It makes a handful of tiny "hi" calls per key (well within free tier).
It never prints key material — only length and the first 4 characters.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
import traceback
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

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# `log` is the structured/file logger (JSON lines, one per probe attempt).
# Console human-readable output is handled separately by print() calls in
# main(), exactly as before — the two are independent so piping stdout
# doesn't lose structure and reading the log file doesn't need parsing
# printed text.
log = logging.getLogger("key_health")


def _setup_logging(log_file: str, verbose: bool) -> None:
    log.setLevel(logging.DEBUG)
    log.propagate = False

    if verbose:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(logging.DEBUG)
        console.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
        log.addHandler(console)

    if log_file and log_file != "-":
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(file_handler)


def _log_json(event: str, **fields: Any) -> None:
    """Emit one structured JSON line. Never raises — logging must not crash probing."""
    try:
        record = {"ts": time.time(), "event": event, **fields}
        log.debug(json.dumps(record, default=str))
    except Exception:  # noqa: BLE001
        log.debug("log-serialization-failed event=%s", event)


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


def _exc_details(exc: Exception) -> dict[str, Any]:
    """Full, un-summarized detail for one exception — for the log file, not
    the terse console line. Grabs anything the underlying SDK exception
    exposes (status code, response body) without assuming which fields
    exist, since different failure modes populate different attributes."""
    details: dict[str, Any] = {
        "exc_type": type(exc).__name__,
        "exc_module": type(exc).__module__,
        "exc_str": str(exc),
        "exc_repr": repr(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }
    for attr in ("status_code", "code", "reason", "response", "details", "message"):
        if hasattr(exc, attr):
            try:
                details[f"attr_{attr}"] = str(getattr(exc, attr))
            except Exception:  # noqa: BLE001
                pass
    if exc.__cause__ is not None:
        details["cause"] = repr(exc.__cause__)
    return details


async def _probe(
    entry: Any, model: str, timeout: float, embed: bool
) -> dict[str, Any]:
    t0 = time.perf_counter()
    key_masked = _mask(entry.key)
    _log_json(
        "probe_start",
        key=key_masked,
        key_index=getattr(entry, "index", None),
        model=model,
        embed=embed,
        timeout=timeout,
    )
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
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        _log_json(
            "probe_ok",
            key=key_masked,
            model=model,
            embed=embed,
            ms=elapsed_ms,
        )
        return {"ok": True, "ms": elapsed_ms, "err": ""}
    except Exception as exc:  # noqa: BLE001 - diagnostics must not crash
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        classified = _classify(exc)
        hint = _project_hint(exc)
        details = _exc_details(exc)
        if not details["exc_str"] and isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            # str(TimeoutError()) is '' — synthesize something actually useful
            # instead of shipping a blank error line.
            synthesized = (
                f"no response within {timeout:.0f}s timeout "
                f"(request ran {elapsed_ms}ms before being cancelled — "
                f"the call was still pending, this is a hang/no-response, "
                f"not a fast rejection)"
            )
            details["exc_str"] = synthesized
            details["exc_synthesized"] = True
        _log_json(
            "probe_error",
            key=key_masked,
            model=model,
            embed=embed,
            ms=elapsed_ms,
            classified=classified,
            project_hint=hint.strip(" ()") or None,
            **details,
        )
        return {
            "ok": False,
            "ms": elapsed_ms,
            "err": classified + hint,
            "raw_err": details["exc_str"],
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
    parser.add_argument(
        "--log-file",
        default="key_health.log",
        help="structured JSON-lines log path (one line per probe). "
        "Pass '-' to disable file logging.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="also print full tracebacks to the console as errors happen, "
        "not just the terse per-probe line",
    )
    args = parser.parse_args()

    _setup_logging(args.log_file, args.verbose)

    pool = get_pool()
    entries = pool.entries()
    models = list(dict.fromkeys([*ALL_MODELS, *args.model]))
    do_embed = not args.no_embed

    print(f"Probing {len(entries)} key(s) × {len(models)} model(s)"
          f"{' + embeddings' if do_embed else ''} (timeout {args.timeout:.0f}s)")
    if args.log_file and args.log_file != "-":
        print(f"Full structured logs (every request, every error) → {args.log_file}\n")
    else:
        print()

    healthy = 0
    error_count = 0
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
            if not res["ok"]:
                error_count += 1
                print(f"        ↳ {res['raw_err']}")
            if res["ok"]:
                any_ok = True
                entry.mark_supported(model)

        if do_embed:
            for model in EMBEDDING_MODELS:
                res = await _probe(entry, model, args.timeout, embed=True)
                mark = "✅" if res["ok"] else "❌"
                print(f"    {mark} {model:<32} {res['ms']:>6}ms  {res['err']} (embed)")
                if not res["ok"]:
                    error_count += 1
                    print(f"        ↳ {res['raw_err']}")

        if any_ok:
            healthy += 1
        print()

    print(f"\nSummary: {healthy}/{len(entries)} keys served at least one model.")
    print(f"Total probe errors this run: {error_count}.")
    if args.log_file and args.log_file != "-":
        print(f"Full details (tracebacks, status codes, project ids) in {args.log_file}")
    print(
        "Project coupling: keys whose quota errors show the SAME project id "
        "share one rate-limit budget. Keys that never error cannot be "
        "compared this way — verify in AI Studio that each belongs to a "
        "separate project for the multi-key split to multiply limits."
    )
    return 0 if healthy == len(entries) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))