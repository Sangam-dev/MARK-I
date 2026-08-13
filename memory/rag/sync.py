"""Boot-time replay of ``memory/rag*.txt`` into the vector index.

Why this exists
---------------
``MemoryManager.append_rag`` writes every ``rag`` entry the model emits to
``memory/rag.txt`` as a human-readable audit log. That file is the record
of what the assistant decided was worth remembering — but a plain text
file is not searchable by meaning, so on its own it can never answer
"what did we decide about X last week?".

This module reads that log at startup and replays it through the RAG
Manager, so everything ever written to it becomes semantically
searchable. After the first boot the work is nearly free: the manager's
content-hash dedupe rejects already-indexed entries *before* spending an
embedding call, so a boot with no new entries costs one SQL query.

Rotated logs
------------
The scan is a glob (``rag*.txt``), not a single filename, so an
archive/rotation scheme — ``rag.2026-07.txt``, ``rag.1.txt`` — is picked
up automatically alongside the live file. That only matters when the
index is rebuilt (new machine, deleted ``vectors.db``, changed embedding
model): in steady state everything already lives in the vector store. But
a rebuild is precisely when losing the archives would hurt most, so they
are replayed, oldest first, with the live ``rag.txt`` last.

The vector index remains authoritative. These files are the input to a
one-way sync and nothing more — this module never writes to them.

Format
------
Written by :meth:`memory.manager.MemoryManager._format_rag_entry`::

    ----------------------------------------
    Timestamp: 2026-08-04T09:15:22.481239
    Type: debugging
    Title: Fixed the WebSocket drop

    The socket closed after 60s because the proxy idle timeout was
    shorter than the heartbeat interval.

    ----------------------------------------

Entries are appended back-to-back, so separator lines routinely appear in
pairs. The parser tolerates that, missing headers, unknown headers, blank
runs, and a truncated final entry (a crash mid-write must not poison
every future boot).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import RAGConfig
from .manager import RAGManager

logger = logging.getLogger("kancha.memory.rag.sync")

# A separator line is a run of dashes on its own line. MemoryManager uses
# exactly 40; we accept 3+ so a hand-edited file still parses.
_MIN_SEPARATOR_DASHES = 3

_KNOWN_HEADERS = {"timestamp", "type", "title"}


def _is_separator(line: str) -> bool:
    stripped = line.strip()
    return (
        len(stripped) >= _MIN_SEPARATOR_DASHES
        and stripped.count("-") == len(stripped)
    )


@dataclass(slots=True)
class RagFileEntry:
    """One parsed block from ``rag.txt``."""

    title: str
    content: str
    entry_type: str = "note"
    timestamp: str = ""

    def to_entry_dict(self) -> dict[str, Any]:
        """Shape expected by :meth:`RAGManager.index_conversation_entries`."""
        return {
            "type": self.entry_type,
            "title": self.title,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class SyncReport:
    """Outcome of one boot sync."""

    file_found: bool = False
    files_scanned: int = 0
    files_read: list[str] = field(default_factory=list)
    entries_parsed: int = 0
    entries_indexed: int = 0
    entries_skipped: int = 0
    entries_failed: int = 0
    chunks_indexed: int = 0
    malformed: int = 0
    duration_ms: float = 0.0
    error: str = ""
    details: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.error:
            return f"rag.txt sync failed: {self.error}"
        if not self.file_found:
            return "no rag*.txt found — nothing to replay"
        scope = (
            f"{self.files_scanned} file(s)"
            if self.files_scanned != 1
            else (self.files_read[0] if self.files_read else "1 file")
        )
        parts = [
            f"rag.txt sync ({scope}): {self.entries_parsed} entr(ies) parsed",
            f"{self.entries_indexed} newly indexed ({self.chunks_indexed} chunks)",
            f"{self.entries_skipped} already known",
        ]
        # Anything that was neither indexed nor a known duplicate is a
        # problem worth surfacing — silently folding it into "skipped"
        # is how a dropped memory goes unnoticed.
        if self.entries_failed:
            parts.append(f"{self.entries_failed} FAILED")
        if self.malformed:
            parts.append(f"{self.malformed} malformed")
        parts.append(f"{self.duration_ms:.0f}ms")
        return ", ".join(parts)


def parse_rag_file(text: str) -> tuple[list[RagFileEntry], int]:
    """Parse the audit log into entries.

    Returns ``(entries, malformed_count)``. Never raises: a corrupt block
    is counted and skipped so one bad write cannot block the rest of the
    user's history from loading.
    """
    entries: list[RagFileEntry] = []
    malformed = 0

    # Split into blocks on separator lines. Consecutive separators
    # produce empty blocks, which are simply dropped.
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in (text or "").splitlines():
        if _is_separator(line):
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)

    for block in blocks:
        # Skip blocks that are entirely blank.
        if not any(line.strip() for line in block):
            continue

        headers: dict[str, str] = {}
        body_start = 0

        for index, line in enumerate(block):
            stripped = line.strip()
            if not stripped:
                # Blank line ends the header section.
                body_start = index + 1
                break
            key, separator, value = stripped.partition(":")
            name = key.strip().lower()
            if separator and (name in _KNOWN_HEADERS or headers):
                # A recognised header, or an unrecognised one sitting in a
                # header block we have already started reading.
                #
                # Bailing out on the first unknown key used to lose every
                # header after it: entries written as
                # ``Timestamp/Type/Category/Title`` stopped at ``Category``
                # and indexed as "Untitled note", because Title came last.
                # An unfamiliar key is not a reason to distrust the block —
                # it is just a field we do not use.
                headers[name] = value.strip()
                continue
            # Nothing header-shaped at the very first line: this block has
            # no header section at all, so it is content from the top.
            body_start = index
            break
        else:
            # Ran off the end with no body.
            body_start = len(block)

        content = "\n".join(block[body_start:]).strip()
        if not content:
            malformed += 1
            continue

        entries.append(
            RagFileEntry(
                title=headers.get("title", "").strip() or "Untitled note",
                content=content,
                entry_type=headers.get("type", "").strip() or "note",
                timestamp=headers.get("timestamp", "").strip(),
            )
        )

    return entries, malformed


class RagFileSync:
    """Replays ``rag.txt`` into the vector index."""

    def __init__(self, config: RAGConfig, rag_manager: RAGManager) -> None:
        self._config = config
        self._rag = rag_manager

    @property
    def paths(self) -> list[Path]:
        """Audit logs to replay, oldest first (see ``RAGConfig.rag_file_paths``)."""
        return self._config.rag_file_paths

    async def run(self, session_id: str = "default") -> SyncReport:
        """Parse and index every audit log. Never raises."""
        started = time.perf_counter()
        report = SyncReport()

        paths = self.paths
        report.files_scanned = len(paths)
        if not paths:
            report.duration_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "No rag*.txt in %s — skipping boot sync", self._config.memory_dir
            )
            return report

        report.file_found = True

        # Collect across all files before indexing, so a duplicate that
        # appears in both an archive and the live file is resolved once,
        # by the hash pass, rather than once per file.
        entries: list[RagFileEntry] = []
        for path in paths:
            try:
                raw = await asyncio.to_thread(
                    path.read_text, encoding="utf-8", errors="replace"
                )
            except Exception as exc:  # noqa: BLE001
                # One unreadable archive must not cost the user every
                # other file's history.
                report.details.append(f"{path.name}: unreadable ({exc})")
                logger.warning("Could not read %s: %s", path, exc)
                continue

            file_entries, malformed = parse_rag_file(raw)
            report.malformed += malformed
            report.files_read.append(path.name)
            entries.extend(file_entries)
            logger.debug(
                "Parsed %d entr(ies) from %s (%d malformed)",
                len(file_entries),
                path.name,
                malformed,
            )

        report.entries_parsed = len(entries)

        if not entries:
            report.duration_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "rag*.txt contained no usable entries across %d file(s) (%d malformed)",
                len(paths),
                report.malformed,
            )
            return report

        logger.info(
            "Replaying %d entr(ies) from %d file(s) [%s] into the vector index…",
            len(entries),
            len(report.files_read),
            ", ".join(report.files_read),
        )

        try:
            # Semantic dedupe is deliberately OFF here. The hash pass
            # already rejects verbatim re-runs, and a semantic search per
            # entry would turn a 500-entry backfill into 500 extra vector
            # searches for no benefit — these entries were already
            # de-duplicated when they were first written.
            results = await self._rag.index_conversation_entries(
                (entry.to_entry_dict() for entry in entries),
                session_id=session_id,
                semantic_dedupe=False,
                origin="rag_file",
            )
        except Exception as exc:  # noqa: BLE001
            report.error = str(exc)
            report.duration_ms = (time.perf_counter() - started) * 1000
            logger.warning("rag.txt replay failed: %s", exc)
            return report

        for result in results:
            if result.indexed:
                report.entries_indexed += 1
                report.chunks_indexed += result.chunks_indexed
                continue

            reason = (result.skipped_reason or "").lower()
            if "already" in reason or "duplicate" in reason:
                report.entries_skipped += 1
            else:
                # Not a duplicate and not indexed — the entry was lost.
                report.entries_failed += 1
                detail = f"{result.title!r}: {result.skipped_reason}"
                report.details.append(detail)
                logger.warning("rag.txt entry not indexed — %s", detail)

        # index_conversation_entries drops entries it cannot process at all
        # (non-dict, blank content) without returning a result, so anything
        # missing from the results list is unaccounted for.
        unaccounted = report.entries_parsed - len(results)
        if unaccounted > 0:
            report.entries_failed += unaccounted
            logger.warning(
                "%d rag.txt entr(ies) were dropped before indexing", unaccounted
            )

        report.duration_ms = (time.perf_counter() - started) * 1000
        logger.info(report.summary())
        return report
