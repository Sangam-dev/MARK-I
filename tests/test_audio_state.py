from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.audio_state import AudioState


async def main() -> None:
    state = AudioState(tts_cooldown_secs=0.05)

    assert not state.is_speaking
    assert not state.is_audio_input_blocked

    state.speaking_started()
    state.speaking_started()
    assert state.is_speaking
    assert state.is_audio_input_blocked

    state.speaking_finished()
    assert state.is_speaking, "overlapping speech should keep mic blocked"

    state.speaking_finished()
    assert not state.is_speaking
    assert state.is_audio_input_blocked, "cooldown should block immediate re-listen"

    started = time.monotonic()
    await state.wait_until_idle()
    elapsed = time.monotonic() - started

    assert elapsed >= 0.04, f"cooldown ended too quickly: {elapsed:.3f}s"
    assert not state.is_audio_input_blocked

    # note_spoken feeds the self-echo guard: only non-empty text is kept,
    # the list is bounded, and timestamps are monotonic.
    assert state.recent_spoken == []
    state.note_spoken("   ")
    assert state.recent_spoken == [], "blank text must not be recorded"
    state.note_spoken("Hello there.")
    state.note_spoken("How can I help?")
    assert [t for _, t in state.recent_spoken] == ["Hello there.", "How can I help?"]
    assert all(at <= time.monotonic() for at, _ in state.recent_spoken)
    for _ in range(20):
        state.note_spoken(f"Sentence number {_}.")
    assert len(state.recent_spoken) <= state._max_recent_spoken

    print("audio_state verified")


if __name__ == "__main__":
    asyncio.run(main())
