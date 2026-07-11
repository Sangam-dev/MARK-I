from __future__ import annotations

import json
import logging
from typing import Any

from reasoning.llm_client_mulapi import (
    ALL_MODELS,
    get_pool,
    hedged_generate,
    hedged_generate_conv,
)

logger = logging.getLogger("kancha.reasoning.llm_client")


class GeminiClient:
    """Gemini Client wrapping the multi-API key rotation and hedged generation logic."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        timeout: float = 12.0,
    ) -> None:
        self.model = model or ALL_MODELS[0]
        self.timeout = timeout
        self.max_retries = max_retries
        self.pool = None

    async def initialize(self) -> None:
        """Initialize the client by loading the key pool."""
        self.pool = get_pool()
        logger.info("GeminiClient initialized successfully.")

    async def generate(self, prompt: str, system: str = "") -> str:
        """Generate a response from the model, optionall injecting system instructions."""
        full_prompt = prompt
        if system:
            full_prompt = f"System Instruction: {system}\n\nUser Prompt: {prompt}"

        try:
            response = await hedged_generate(
                pool=self.pool,
                models=ALL_MODELS,
                prompt=full_prompt,
                hedge_width=2,
                timeout=self.timeout,
            )
            return response
        except Exception as e:
            logger.exception("Failed to generate response: %s", e)
            return "I'm having trouble thinking right now. Could you try again?"

    async def generate_with_history(
        self,
        history: list[dict],
        system: str = "",
    ) -> str:
        """
        Generate a response using proper Gemini multi-turn conversation format.

        Args:
            history: Ordered list of conversation turns.
                     Each dict must have "role" ("user" or "assistant") and "content".
                     The LAST item must have role "user" — it is the current user message.
            system:  Optional system instruction (passed via GenerateContentConfig,
                     not embedded in the prompt string).

        Returns:
            The assistant's response text.
        """
        from google.genai import types

        if not history:
            return await self.generate(prompt="", system=system)

        # Convert to Gemini native Content objects.
        # Gemini uses "model" for assistant turns (not "assistant").
        contents = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])],
                )
            )

        # Ensure conversation ends with a user turn (Gemini requirement).
        if not contents or contents[-1].role != "user":
            logger.warning(
                "generate_with_history: last turn is not 'user' — appending empty user turn"
            )
            contents.append(types.Content(role="user", parts=[types.Part(text="")]))

        config = None
        if system:
            config = types.GenerateContentConfig(system_instruction=system)

        try:
            response = await hedged_generate_conv(
                pool=self.pool,
                models=ALL_MODELS,
                contents=contents,
                config=config,
                hedge_width=2,
                timeout=self.timeout,
            )
            return response
        except Exception as e:
            logger.exception("generate_with_history failed: %s", e)
            return "I'm having trouble thinking right now. Could you try again?"

    async def generate_json(
        self, prompt: str, schema_description: str | None = None, system: str = ""
    ) -> dict[str, Any]:
        """Generate a JSON response from the model, retrying on decode errors."""
        full_prompt = prompt
        if schema_description:
            full_prompt = (
                f"{prompt}\n\n"
                f"You MUST return valid JSON matching the following schema description:\n"
                f"{schema_description}\n"
                f"Ensure the response is wrapped in standard JSON format, with NO markdown formatting, no preamble, and no code blocks (do NOT use ```json)."
            )
        else:
            full_prompt = (
                f"{prompt}\n\n"
                f"You MUST return valid JSON with no preamble, and no markdown formatting (do NOT use ```json)."
            )

        if system:
            full_prompt = f"System Instruction: {system}\n\nUser Prompt: {full_prompt}"

        for attempt in range(self.max_retries + 1):
            try:
                response = await hedged_generate(
                    pool=self.pool,
                    models=ALL_MODELS,
                    prompt=full_prompt,
                    hedge_width=2,
                    timeout=self.timeout,
                )
                cleaned = response.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    cleaned = "\n".join(lines).strip()

                return json.loads(cleaned)
            except json.JSONDecodeError as je:
                logger.warning(
                    "JSON decode failed (attempt %d/%d): %s. Raw response: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    je,
                    response,
                )
                if attempt == self.max_retries:
                    logger.exception("Failed to generate valid JSON after retries.")
                    return {}
            except Exception as e:
                logger.exception("Error during generate_json: %s", e)
                return {}
        return {}

    async def health_check(self) -> bool:
        """Return True if the key pool has available keys."""
        if not self.pool:
            return False
        return any(e.is_available for e in self.pool._entries)
