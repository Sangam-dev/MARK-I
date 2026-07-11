"""Short-term conversational memory — the RAM buffer.

Holds the last N conversation turns for injection into every LLM prompt.
Does NOT persist — cleared on restart.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Turn:
    """A single conversation turn."""

    role: Literal["user", "assistant", "system"]
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to OpenAI-compatible message format."""
        return {"role": self.role, "content": self.content}


class ConversationContext:
    """Short-term conversation buffer with rolling eviction."""

    def __init__(self, max_turns: int = 10) -> None:
        self._buffer: deque[Turn] = deque(maxlen=max_turns)
        self._lock = asyncio.Lock()

    @property
    def max_turns(self) -> int:
        """Maximum number of turns to retain."""
        return self._buffer.maxlen or 10

    async def add_turn(self, role: Literal["user", "assistant", "system"], content: str) -> None:
        """Add a turn to the buffer, skipping empty content."""
        if not content or not content.strip():
            return
        async with self._lock:
            self._buffer.append(Turn(role=role, content=content.strip()))

    async def as_messages(self) -> list[dict[str, str]]:
        """Return buffer as OpenAI-compatible message list."""
        async with self._lock:
            return [turn.to_dict() for turn in self._buffer]

    async def clear(self) -> None:
        """Clear the conversation buffer."""
        async with self._lock:
            self._buffer.clear()

    async def token_estimate(self) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        async with self._lock:
            total_chars = sum(len(turn.content) for turn in self._buffer)
            return max(1, total_chars // 4)

    def __repr__(self) -> str:
        return f"ConversationContext(max_turns={self.max_turns}, current_turns={len(self._buffer)})"