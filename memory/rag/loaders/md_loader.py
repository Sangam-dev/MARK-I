"""Markdown loader.

Markdown is kept largely intact rather than rendered to plain text: the
heading structure is a strong retrieval signal, and the chunker splits on
blank lines, which Markdown already uses to separate blocks.

Two things *are* extracted: YAML front matter (so ``title:`` becomes the
document title instead of leaking into the embedded body) and the first
H1, used as a title fallback.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..models import Document
from .base import DocumentLoader, LoaderError

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*\n", re.DOTALL)
_H1_RE = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.MULTILINE)
_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.+?)\s*$", re.MULTILINE)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split ``---`` front matter off the top of a Markdown file.

    Deliberately a flat ``key: value`` scan rather than a YAML parse —
    it avoids a PyYAML dependency and handles the shape actually used by
    notes and docs. Nested YAML degrades to being ignored, not to a crash.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    fields: dict[str, str] = {}
    for line in match.group("body").splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and value:
            fields[key] = value

    return fields, text[match.end() :]


class MarkdownLoader(DocumentLoader):
    """Loads ``.md``/``.markdown`` files, preserving heading structure."""

    extensions = (".md", ".markdown", ".mdown", ".mkd")
    name = "markdown"

    async def load(
        self,
        path: Path,
        *,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        raw = await self._read_bytes(path)
        text = self.decode_text(raw, path)

        frontmatter, body = _parse_frontmatter(text)
        if not body.strip():
            raise LoaderError(f"{path.name} contains no readable text")

        heading_match = _H1_RE.search(body)
        resolved_title = (
            title
            or frontmatter.get("title")
            or (heading_match.group("title") if heading_match else "")
            or path.stem
        )

        headings = [m.group("text") for m in _HEADING_RE.finditer(body)]

        extra: dict[str, Any] = {"headings": headings[:20]}
        # Front-matter keys are namespaced so they can never collide with
        # the metadata keys the chunker and store rely on.
        extra.update({f"fm_{k}": v for k, v in frontmatter.items()})

        return self.build_document(
            path=path,
            content=body,
            title=resolved_title,
            metadata=metadata,
            extra=extra,
        )
