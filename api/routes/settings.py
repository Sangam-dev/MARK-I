"""Backs the SETTINGS panel with a small JSON-backed store.

NOTE: `tts_enabled` written here takes effect on the *next* backend restart
only — TTSHandler is wired once at pipeline build time in core/pipeline.py.
Making this hot-swappable is listed as a follow-up in answers/guide.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from api.schemas import SettingsModel

router = APIRouter(prefix="/settings", tags=["settings"])

_SETTINGS_PATH = (
    Path(__file__).resolve().parents[2] / "memory" / "data" / "settings.json"
)


def _load() -> SettingsModel:
    if not _SETTINGS_PATH.exists():
        return SettingsModel()
    try:
        data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        return SettingsModel.model_validate(data)
    except Exception:
        return SettingsModel()


def _save(settings: SettingsModel) -> None:
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_PATH.write_text(settings.model_dump_json(indent=2), encoding="utf-8")


@router.get("", response_model=SettingsModel)
async def get_settings() -> SettingsModel:
    return _load()


@router.put("", response_model=SettingsModel)
async def update_settings(body: SettingsModel) -> SettingsModel:
    _save(body)
    return body
