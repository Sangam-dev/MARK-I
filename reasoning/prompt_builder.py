from __future__ import annotations

import logging
from typing import Any

from core.events import IntentIdentified, MemoryRetrieved

logger = logging.getLogger("JARVIS.reasoning.prompt_builder")
from reasoning.prompt import persona

JARVIS_PERSONA = persona.strip()  # Use the persona defined in reasoning/prompt.py

class PromptBuilder:
    """Assembles prompts with durable structured facts only."""

    def __init__(self, config: Any = None, max_prompt_tokens: int = 4000) -> None:
        self.max_prompt_tokens = (
            getattr(config, "max_prompt_tokens", max_prompt_tokens)
            if config is not None
            else max_prompt_tokens
        )

    async def build(
        self,
        user_input: str,
        intent_event: IntentIdentified,
        memory_event: MemoryRetrieved | None,
        context_messages: list[dict[str, str]],
    ) -> tuple[str, list[dict[str, str]]]:
        """Build the system prompt and conversation messages list for the model."""
        structured = memory_event.structured_context if memory_event is not None else []

        facts = self._format_structured_facts(structured)

        system = f"{JARVIS_PERSONA}"
        if facts:
            system += f"\n\nStructured Facts:\n{facts}"

        messages = context_messages + [{"role": "user", "content": user_input}]

        return self._trim_to_budget(
            system,
            messages,
            context_messages,
            user_input,
            facts,
        )

    def _format_structured_facts(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return ""
        lines = []
        for fact in results:
            if "key" in fact and "value" in fact:
                lines.append(f"- {fact['key']}: {fact['value']}")
            elif "content" in fact:
                lines.append(f"- {fact['content']}")
        return "\n".join(lines)

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _trim_to_budget(
        self,
        system: str,
        messages: list[dict[str, str]],
        context_messages: list[dict[str, str]],
        user_input: str,
        facts: str,
    ) -> tuple[str, list[dict[str, str]]]:
        current_context = list(context_messages)

        while True:
            # Construct the system instruction
            system_parts = [JARVIS_PERSONA]
            if facts:
                system_parts.append(f"Structured Facts:\n{facts}")

            system_str = "\n\n".join(system_parts)
            messages_list = current_context + [{"role": "user", "content": user_input}]

            total_tokens = self._estimate_tokens(system_str) + sum(
                self._estimate_tokens(m["content"]) for m in messages_list
            )

            if total_tokens <= self.max_prompt_tokens:
                return system_str, messages_list

            if current_context:
                current_context.pop(0)
                logger.info("Token budget exceeded. Dropped oldest conversation turn.")
            else:
                return system_str, messages_list
