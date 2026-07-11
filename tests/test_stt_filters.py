from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from input.stt import _is_probable_silence_hallucination


def main() -> None:
    quiet_short_audio = np.zeros(8000, dtype=np.float32)
    clear_audio = np.full(32000, 0.08, dtype=np.float32)

    assert _is_probable_silence_hallucination("Thank you.", quiet_short_audio)
    assert not _is_probable_silence_hallucination("open the browser", quiet_short_audio)
    assert not _is_probable_silence_hallucination("Thank you.", clear_audio)

    print("stt filters verified")


if __name__ == "__main__":
    main()
