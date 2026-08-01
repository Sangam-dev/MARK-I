from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable


class TokenLog:
    """Simple per-session token usage logger for LLM calls."""

    def __init__(self, session_id: str, sink: Callable[[dict], None] | None = None) -> None:
        self.session_id = session_id
        self._entries: list[dict] = []
        self._sink = sink

    def record(
        self,
        call_site: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
    ) -> None:
        entry = {
            "session_id": self.session_id,
            "call_site": call_site,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._entries.append(entry)
        if self._sink is not None:
            self._sink(entry)

    @property
    def total_input_tokens(self) -> int:
        return sum(e["input_tokens"] for e in self._entries)

    @property
    def total_output_tokens(self) -> int:
        return sum(e["output_tokens"] for e in self._entries)

    @property
    def entries(self) -> list[dict]:
        return list(self._entries)

    def by_call_site(self) -> dict[str, dict[str, int]]:
        totals: dict[str, dict[str, int]] = {}
        for entry in self._entries:
            site = entry["call_site"]
            totals.setdefault(site, {"input": 0, "output": 0})
            totals[site]["input"] += entry["input_tokens"]
            totals[site]["output"] += entry["output_tokens"]
        return totals


def jsonl_sink(path: Path) -> Callable[[dict], None]:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _write(entry: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    return _write
