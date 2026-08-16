from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.audio_state import audio_state
from input.stt import (
    _is_self_echo,
    _is_probable_silence_hallucination,
    _text_similarity,
    _word_tokens,
)


def _set_recent(*sentences: str) -> None:
    now = time.monotonic()
    audio_state.recent_spoken = [
        (now - 1.0 - i * 0.5, sentence) for i, sentence in enumerate(sentences)
    ]


def main() -> None:
    quiet_short_audio = np.zeros(8000, dtype=np.float32)
    clear_audio = np.full(32000, 0.08, dtype=np.float32)

    assert _is_probable_silence_hallucination("Thank you.", quiet_short_audio)
    assert not _is_probable_silence_hallucination("open the browser", quiet_short_audio)
    assert not _is_probable_silence_hallucination("Thank you.", clear_audio)

    # ── Self-echo guard ────────────────────────────────────────────────
    assert _word_tokens("Hello, World!") == ["hello", "world"]
    assert _text_similarity("the quick brown fox", "the quick brown fox") == 1.0
    assert _text_similarity("the quick brown fox", "a slow green frog") < 0.4
    assert _text_similarity("", "anything") == 0.0

    spoken = "The weather today is sunny in Bangalore."
    transcript = "the weather today is sunny"

    _set_recent("An earlier unrelated greeting.")
    assert not _is_self_echo(transcript), "no match when nothing similar was said"

    _set_recent(spoken)
    assert _is_self_echo(transcript), "a near-duplicate of what was spoken is an echo"

    _set_recent("Completely different sentence about the stock market.")
    assert not _is_self_echo(transcript), "different topic is not an echo"

    # Only one of several recent sentences needs to match (multi-sentence echo).
    _set_recent(spoken, "Another sentence played right after.")
    assert _is_self_echo(transcript)

    # Outside the window, a match is stale — a real user repeating the phrase.
    now = time.monotonic()
    audio_state.recent_spoken = [(now - 600.0, spoken)]
    assert not _is_self_echo(transcript), "a stale match is not a live echo"

    # Too-short spoken text is skipped: "Okay." must never veto a user's "Okay".
    audio_state.recent_spoken = [(now - 1.0, "Okay.")]
    assert not _is_self_echo("Okay"), "short utterances are not echo-gated"

    # Restore the shared singleton so nothing downstream sees the test data.
    audio_state.recent_spoken = []

    print("stt filters verified")


if __name__ == "__main__":
    main()
