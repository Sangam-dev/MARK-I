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

    print("audio_state verified")


if __name__ == "__main__":
    asyncio.run(main())
