"""Single-model Gemini client (no fallbacks, no parallelism).

Uses only: gemini-flash-lite-latest
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from google import genai

 
MODEL = "gemini-2.5-flash-lite"      #gemini-2.5-flash-lite  or gemini-flash-lite-latest 


def _get_client() -> genai.Client:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        sys.exit(
            "Missing API key.\n"
            "Set GEMINI_API_KEY and re-run.\n"
            "Example: export GEMINI_API_KEY='YOUR_KEY_HERE'\n"
        )
    return genai.Client(api_key=key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple Gemini client")
    parser.add_argument("prompt", nargs="?", default=None)
    parser.add_argument(
        "--stream",
        action="store_true",
        default=os.getenv("STREAM", "0") == "1",
        help="Stream tokens as they arrive",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=os.getenv("INTERACTIVE", "0") == "1",
        help="REPL mode (send many prompts in one process)",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        default=os.getenv("WARMUP", "0") == "1",
        help="Send a tiny warm-up request first",
    )
    return parser.parse_args()


def generate(client: genai.Client, prompt: str) -> None:
    start = time.perf_counter()
    resp = client.models.generate_content(model=MODEL, contents=prompt)
    text = getattr(resp, "text", "") or ""
    print(text, end="\n" if text and not text.endswith("\n") else "")
    sys.stderr.write(f"[Done {time.perf_counter() - start:0.2f}s]\n")


def stream(client: genai.Client, prompt: str) -> None:
    start = time.perf_counter()
    first_token_at: float | None = None

    sys.stderr.write(f"[Model {MODEL}]\n")
    response = client.models.generate_content_stream(model=MODEL, contents=prompt)

    for chunk in response:
        text = getattr(chunk, "text", "")
        if not text:
            continue
        if first_token_at is None:
            first_token_at = time.perf_counter()
            sys.stderr.write(f"[TTFT {first_token_at - start:0.2f}s]\n")
        print(text, end="", flush=True)

    print(flush=True)
    sys.stderr.write(f"[Done {time.perf_counter() - start:0.2f}s]\n")


def main() -> None:
    args = parse_args()
    client = _get_client()

    runner = stream if args.stream else generate

    if args.warmup:
        runner(client, "Reply with exactly one character: .")

    if args.interactive:
        if args.prompt:
            runner(client, args.prompt)
        while True:
            try:
                prompt = input("\nPrompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt:
                runner(client, prompt)
    else:
        runner(client, args.prompt or "Tell me a joke about cats.")


if __name__ == "__main__":
    main()