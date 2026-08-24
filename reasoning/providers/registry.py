"""Provider registry, hot-swap router and the swappable client facade.

The registry is data-driven: adding a provider means appending one
:class:`ProviderSpec` entry. Anything speaking the OpenAI-compatible chat
protocol needs zero new code; only a genuinely alien API would need its
own :class:`~reasoning.providers.base.LLMProvider` implementation.

The :class:`ProviderRouter` owns one live backend per lane (conversation /
task). The settings API drives :meth:`ProviderRouter.swap`, which
validates the target (build + health-check) *before* replacing anything,
so a failed swap leaves the running assistant exactly as it was.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from reasoning.llm_client import GeminiClient

from reasoning.providers.base import LLMProvider
from reasoning.providers.gemini_adapter import GeminiAdapter
from reasoning.providers.openai_compat import (
    OpenAICompatClient,
    ProviderConfig,
)

logger = logging.getLogger("kancha.reasoning.providers.registry")

# Same file the REST settings route persists to — single source of truth
# for which provider the user picked.
SETTINGS_PATH = (
    Path(__file__).resolve().parents[2] / "memory" / "data" / "settings.json"
)
_PROVIDER_KEY = "llm_provider"


@dataclass(frozen=True)
class ProviderSpec:
    """One registry entry describing how to build a backend."""

    name: str
    label: str
    kind: str  # "gemini" | "openai_compat"
    base_url: str = ""
    key_prefix: str = ""
    model_env: str = ""
    fallbacks_env: str = ""
    default_models: tuple[str, ...] = ()
    notes: str = ""


PROVIDERS: dict[str, ProviderSpec] = {
    "gemini": ProviderSpec(
        name="gemini",
        label="Gemini",
        kind="gemini",
        notes="google-genai SDK with multi-key rotation (default)",
    ),
    "groq": ProviderSpec(
        name="groq",
        label="Groq",
        kind="openai_compat",
        base_url=os.getenv("KANCHA_GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        key_prefix="GROQ_API_KEY",
        model_env="KANCHA_GROQ_LLM_MODEL",
        fallbacks_env="KANCHA_GROQ_LLM_FALLBACKS",
        default_models=("llama-3.3-70b-versatile", "openai/gpt-oss-20b"),
        notes="GROQ_API_KEY_1..9 rotation, same semantics as the Gemini pool",
    ),
    "ollama": ProviderSpec(
        name="ollama",
        label="Ollama",
        kind="openai_compat",
        base_url=os.getenv("KANCHA_OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        model_env="KANCHA_OLLAMA_LLM_MODEL",
        fallbacks_env="KANCHA_OLLAMA_LLM_FALLBACKS",
        default_models=("llama3.2",),
        notes="local daemon, no credentials",
    ),
}

DEFAULT_PROVIDER = "gemini"


def read_configured_provider(path: Path | None = None) -> str:
    """Read the persisted ``llm_provider`` choice, tolerantly.

    Missing file, missing field or garbage all fall back to Gemini — a
    corrupt settings.json must never stop boot.
    """
    provider_path = path or SETTINGS_PATH
    try:
        data = json.loads(provider_path.read_text(encoding="utf-8"))
        name = str(data.get(_PROVIDER_KEY, "")).strip().lower()
        if name in PROVIDERS:
            return name
    except Exception:  # noqa: BLE001
        pass
    return DEFAULT_PROVIDER


async def _build_client(
    spec: ProviderSpec,
    *,
    token_log=None,
    lane: str,
    lane_direction: int,
) -> LLMProvider:
    """Construct and initialize one lane client for *spec*."""
    if spec.kind == "gemini":
        client: LLMProvider = GeminiAdapter(
            GeminiClient(
                token_log=token_log,
                lane=lane,
                lane_direction=lane_direction,
            )
        )
    else:
        client = OpenAICompatClient(
            ProviderConfig(
                name=spec.name,
                base_url=spec.base_url,
                key_prefix=spec.key_prefix,
                model_env=spec.model_env,
                fallbacks_env=spec.fallbacks_env,
                default_models=list(spec.default_models),
            ),
            token_log=token_log,
            lane=lane,
            lane_direction=lane_direction,
        )
    await client.initialize()
    return client


@dataclass
class _LaneSlot:
    direction: int
    active: Optional[LLMProvider] = None
    retired: list[LLMProvider] = field(default_factory=list)


class ProviderRouter:
    """Owns the live provider per lane and performs validated swaps."""

    def __init__(self, token_log=None) -> None:
        self.token_log = token_log
        self.active_provider: str = read_configured_provider()
        self._lanes: dict[str, _LaneSlot] = {}
        self._lock = asyncio.Lock()

    def register_lane(self, name: str, direction: int) -> None:
        self._lanes[name] = _LaneSlot(direction=direction)

    async def start(self) -> None:
        """Build every lane for the configured provider.

        If the configured provider cannot serve (missing keys, daemon
        down), fall back to Gemini loudly instead of breaking boot.
        """
        spec = PROVIDERS[self.active_provider]
        try:
            for lane_name, slot in self._lanes.items():
                slot.active = await _build_client(
                    spec,
                    token_log=self.token_log,
                    lane=lane_name,
                    lane_direction=slot.direction,
                )
            first = next(iter(self._lanes.values()), None)
            if first is not None and first.active is not None:
                healthy = await first.active.health_check()
                if not healthy and self.active_provider != DEFAULT_PROVIDER:
                    raise RuntimeError(
                        f"{spec.label} has no usable credentials/endpoint"
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Configured provider '%s' unavailable at startup (%s) — "
                "falling back to '%s'",
                self.active_provider,
                exc,
                DEFAULT_PROVIDER,
            )
            await self._activate(DEFAULT_PROVIDER)

    def client_for(self, lane: str) -> LLMProvider:
        slot = self._lanes.get(lane)
        if slot is None or slot.active is None:
            raise RuntimeError(f"No active provider client for lane '{lane}'")
        return slot.active

    async def swap(self, name: str) -> tuple[bool, str]:
        """Validate-then-activate *name* across every lane.

        Builds fresh clients and health-checks them while the old ones
        keep serving; only on full success do the pointers flip. Returns
        ``(ok, message)`` — never raises, never leaves a half-swapped
        state behind.
        """
        async with self._lock:
            if name not in PROVIDERS:
                return False, f"Unknown provider '{name}'"
            if name == self.active_provider:
                return True, f"Already on {PROVIDERS[name].label}"

            spec = PROVIDERS[name]
            candidates: dict[str, LLMProvider] = {}
            try:
                for lane_name, slot in self._lanes.items():
                    candidates[lane_name] = await _build_client(
                        spec,
                        token_log=self.token_log,
                        lane=lane_name,
                        lane_direction=slot.direction,
                    )
                for lane_name, candidate in candidates.items():
                    if not await candidate.health_check():
                        raise RuntimeError(
                            f"{spec.label} failed health check on lane '{lane_name}'"
                        )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Swap to '%s' refused: %s — staying on '%s'",
                    name,
                    exc,
                    self.active_provider,
                )
                for candidate in candidates.values():
                    await self._dispose(candidate)
                return False, str(exc)

            previous = self.active_provider
            for lane_name, slot in self._lanes.items():
                old = slot.active
                slot.active = candidates[lane_name]
                if old is not None:
                    slot.retired.append(old)
            self.active_provider = name
            logger.info("LLM provider switched %s -> %s", previous, name)
            return True, f"Switched to {spec.label}"

    async def _activate(self, name: str) -> None:
        """Unconditional activation used by the startup fallback."""
        spec = PROVIDERS[name]
        for lane_name, slot in self._lanes.items():
            slot.active = await _build_client(
                spec,
                token_log=self.token_log,
                lane=lane_name,
                lane_direction=slot.direction,
            )
        self.active_provider = name

    @staticmethod
    async def _dispose(client: LLMProvider) -> None:
        """Best-effort cleanup of a retired httpx-backed client."""
        aclose = getattr(client, "aclose", None)
        if aclose is None:
            return
        try:
            await aclose()
        except Exception:  # noqa: BLE001
            pass


class SwappableClient(LLMProvider):
    """Stable per-lane facade that always delegates to the router's
    currently-active backend.

    Consumers keep one reference for the process lifetime; a provider
    swap underneath changes behaviour without touching any caller.
    In-flight requests hold their own reference to the old client and
    finish undisturbed.
    """

    def __init__(self, router: ProviderRouter, lane: str) -> None:
        self._router = router
        self._lane = lane

    @property
    def provider_name(self) -> str:
        return self._router.active_provider

    async def initialize(self) -> None:
        # Router.start() builds every lane; nothing per-client to do.
        self._router.client_for(self._lane)

    async def generate(self, prompt: str, system: str = "", hedge_width=None,
                       call_site: str = "conversational") -> str:
        return await self._router.client_for(self._lane).generate(
            prompt=prompt, system=system, hedge_width=hedge_width,
            call_site=call_site,
        )

    async def generate_with_history(self, history: list[dict], system: str = "",
                                    hedge_width=None,
                                    call_site: str = "conversational") -> str:
        return await self._router.client_for(self._lane).generate_with_history(
            history=history, system=system, hedge_width=hedge_width,
            call_site=call_site,
        )

    def generate_with_history_stream(self, history: list[dict], system: str = "",
                                     hedge_width=None,
                                     call_site: str = "conversational",
                                     max_output_tokens: int | None = None):
        return self._router.client_for(self._lane).generate_with_history_stream(
            history=history, system=system, hedge_width=hedge_width,
            call_site=call_site, max_output_tokens=max_output_tokens,
        )

    async def generate_with_history_json(self, history: list[dict], system: str = "",
                                         hedge_width=None,
                                         call_site: str = "conversational") -> dict:
        return await self._router.client_for(self._lane).generate_with_history_json(
            history=history, system=system, hedge_width=hedge_width,
            call_site=call_site,
        )

    async def generate_json(self, prompt: str, schema_description: str | None = None,
                            system: str = "", hedge_width=None,
                            call_site: str = "planner") -> dict:
        return await self._router.client_for(self._lane).generate_json(
            prompt=prompt, schema_description=schema_description,
            system=system, hedge_width=hedge_width, call_site=call_site,
        )

    async def health_check(self) -> bool:
        return await self._router.client_for(self._lane).health_check()

    def has_available_key(self) -> bool:
        return self._router.client_for(self._lane).has_available_key()

    async def prewarm(self) -> None:
        await self._router.client_for(self._lane).prewarm()


# ── Process-wide singleton (mirrors set_shared_opencode_client) ──────────────

_router: ProviderRouter | None = None


def set_shared_router(router: ProviderRouter) -> None:
    global _router
    _router = router


def get_shared_router() -> ProviderRouter | None:
    return _router
