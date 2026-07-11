from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from shutil import which
from typing import Any


@dataclass(slots=True)
class AlarmResult:
    success: bool
    message: str
    alarm_id: str | None = None


@dataclass(slots=True)
class AlarmRequest:
    kind: str
    due_at: datetime
    message: str


@dataclass(slots=True)
class ScheduledAlarm:
    alarm_id: str
    kind: str
    due_at: datetime
    message: str
    command: str
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "alarm_id": self.alarm_id,
            "kind": self.kind,
            "due_at": self.due_at.isoformat(),
            "message": self.message,
            "command": self.command,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScheduledAlarm:
        return cls(
            alarm_id=str(data["alarm_id"]),
            kind=str(data.get("kind") or "alarm"),
            due_at=datetime.fromisoformat(str(data["due_at"])),
            message=str(data.get("message") or "Alarm"),
            command=str(data.get("command") or ""),
            created_at=datetime.fromisoformat(str(data.get("created_at") or data["due_at"])),
        )


_NUMBER_WORDS: dict[str, float] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
}

_DURATION_UNITS: dict[str, float] = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "second": 1,
    "seconds": 1,
    "m": 60,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
    "h": 3600,
    "hr": 3600,
    "hrs": 3600,
    "hour": 3600,
    "hours": 3600,
    "d": 86400,
    "day": 86400,
    "days": 86400,
}

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_DURATION_RE = re.compile(
    r"\b(?:in|after|for)?\s*"
    r"(?P<value>\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty)"
    r"\s*(?P<unit>seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)\b",
    re.IGNORECASE,
)
_HALF_HOUR_RE = re.compile(r"\b(?:in|after|for)\s+half\s+(?:an?\s+)?hour\b", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})\b")
_SLASH_DATE_RE = re.compile(r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})(?:/(?P<year>\d{2,4}))?\b")
_TIME_RE = re.compile(
    r"\b(?:(?P<prefix>at|for)\s+)?"
    r"(?P<hour>[01]?\d|2[0-3])"
    r"(?::(?P<minute>[0-5]\d)(?::(?P<second>[0-5]\d))?)?"
    r"\s*(?P<ampm>a\.?m\.?|p\.?m\.?)?\b",
    re.IGNORECASE,
)

_LOCK = threading.RLock()
_ALARMS: dict[str, ScheduledAlarm] = {}
_TIMERS: dict[str, threading.Timer] = {}
_LOADED = False
_LOADED_PATH: Path | None = None
_RING_REPEATS = 10
_RING_PAUSE_SECONDS = 0.2


def set_alarm(command: str, delay_seconds: int | None = None) -> AlarmResult:
    _ensure_loaded()
    if delay_seconds is not None:
        kind = _detect_kind(command)
        due_at = datetime.now() + timedelta(seconds=delay_seconds)
        request = AlarmRequest(kind=kind, due_at=due_at, message=command)
    else:
        request = parse_alarm_request(command)

    if request is None:
        return AlarmResult(
            False,
            "Tell me when to set it, for example: set an alarm in 10 minutes or remind me at 3pm.",
        )

    alarm = ScheduledAlarm(
        alarm_id=str(uuid.uuid4()),
        kind=request.kind,
        due_at=request.due_at,
        message=request.message,
        command=f"{command} (in {delay_seconds}s)" if delay_seconds is not None else " ".join(command.strip().split()),
        created_at=datetime.now(),
    )
    with _LOCK:
        _ALARMS[alarm.alarm_id] = alarm
        _schedule_timer_unlocked(alarm)
        _persist_unlocked()

    return AlarmResult(
        True,
        f"{alarm.kind.capitalize()} set for {_format_due(alarm.due_at)}.",
        alarm.alarm_id,
    )


def list_alarms() -> AlarmResult:
    _ensure_loaded()
    with _LOCK:
        alarms = sorted(_ALARMS.values(), key=lambda item: item.due_at)

    if not alarms:
        return AlarmResult(True, "No alarms are scheduled.")

    lines = [
        f"{alarm.kind.capitalize()} at {_format_due(alarm.due_at)}"
        + (f": {alarm.message}" if alarm.message else "")
        for alarm in alarms
    ]
    return AlarmResult(True, "Scheduled alarms: " + "; ".join(lines))


def cancel_alarms() -> AlarmResult:
    _ensure_loaded()
    with _LOCK:
        count = len(_ALARMS)
        for timer in _TIMERS.values():
            timer.cancel()
        _TIMERS.clear()
        _ALARMS.clear()
        _persist_unlocked()

    if count == 0:
        return AlarmResult(True, "No alarms were scheduled.")
    if count == 1:
        return AlarmResult(True, "Cancelled 1 scheduled alarm.")
    return AlarmResult(True, f"Cancelled {count} scheduled alarms.")


def parse_alarm_request(command: str, now: datetime | None = None) -> AlarmRequest | None:
    now = now or datetime.now()
    text = " ".join(command.strip().split())
    if not text:
        return None

    kind = _detect_kind(text)
    due_at: datetime | None = None
    time_span: tuple[int, int] | None = None

    duration = _parse_relative_duration(text)
    if duration is not None:
        seconds, time_span = duration
        if seconds <= 0:
            return None
        due_at = now + timedelta(seconds=seconds)
    else:
        absolute = _parse_absolute_time(text, now)
        if absolute is not None:
            due_at, time_span = absolute

    if due_at is None or time_span is None:
        return None

    return AlarmRequest(kind=kind, due_at=due_at, message=_extract_message(text, kind, time_span))


def _parse_relative_duration(text: str) -> tuple[float, tuple[int, int]] | None:
    half_hour = _HALF_HOUR_RE.search(text)
    if half_hour:
        return 30 * 60, half_hour.span()

    match = _DURATION_RE.search(text)
    if not match:
        return None

    raw_value = match.group("value").lower()
    value = _NUMBER_WORDS.get(raw_value)
    if value is None:
        value = float(raw_value)

    unit = match.group("unit").lower()
    return value * _DURATION_UNITS[unit], match.span()


def _parse_absolute_time(text: str, now: datetime) -> tuple[datetime, tuple[int, int]] | None:
    lowered = text.lower()
    explicit_date = _extract_date(lowered, now)

    for match in _TIME_RE.finditer(text):
        ampm = match.group("ampm")
        prefix = match.group("prefix")
        has_minutes = match.group("minute") is not None
        if not ampm and not prefix and (explicit_date is None or not has_minutes):
            continue

        end = match.end()
        following = lowered[end : end + 8].strip()
        if following.startswith(("second", "sec", "minute", "min", "hour", "hr", "day")):
            continue

        hour = int(match.group("hour"))
        minute = int(match.group("minute") or "0")
        second = int(match.group("second") or "0")
        if ampm:
            hour = _apply_ampm(hour, ampm)
        if hour > 23:
            continue

        due_date = explicit_date or now.date()
        due_at = datetime.combine(due_date, datetime.min.time()).replace(hour=hour, minute=minute, second=second)
        if explicit_date is None and due_at <= now:
            due_at += timedelta(days=1)
        return due_at, match.span()

    return None


def _apply_ampm(hour: int, ampm: str) -> int:
    marker = ampm.lower().replace(".", "")
    if marker == "am":
        return 0 if hour == 12 else hour
    return hour if hour == 12 else hour + 12


def _extract_date(text: str, now: datetime):
    if "tomorrow" in text:
        return (now + timedelta(days=1)).date()
    if "today" in text:
        return now.date()

    for name, weekday in _WEEKDAYS.items():
        if re.search(rf"\b{name}\b", text):
            days_ahead = (weekday - now.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            return (now + timedelta(days=days_ahead)).date()

    iso_match = _ISO_DATE_RE.search(text)
    if iso_match:
        return datetime(
            int(iso_match.group("year")),
            int(iso_match.group("month")),
            int(iso_match.group("day")),
        ).date()

    slash_match = _SLASH_DATE_RE.search(text)
    if slash_match:
        year = slash_match.group("year")
        year_int = now.year if year is None else int(year)
        if year_int < 100:
            year_int += 2000
        return datetime(year_int, int(slash_match.group("month")), int(slash_match.group("day"))).date()

    return None


def _detect_kind(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\b(timer|countdown)\b", lowered):
        return "timer"
    if re.search(r"\b(remind|reminder|remainder)\b", lowered):
        return "reminder"
    return "alarm"


def _extract_message(text: str, kind: str, time_span: tuple[int, int]) -> str:
    before = text[: time_span[0]].strip(" ,.")
    after = text[time_span[1] :].strip(" ,.")
    candidates = [after, before]

    for candidate in candidates:
        cleaned = _clean_message_candidate(candidate)
        if cleaned:
            return cleaned

    return {"alarm": "Alarm", "timer": "Timer", "reminder": "Reminder"}[kind]


def _clean_message_candidate(candidate: str) -> str:
    cleaned = candidate.strip()
    cleaned = re.sub(
        r"^(?:please\s+)?(?:set\s+)?(?:an?\s+)?(?:alarm|timer|reminder|remainder)\s*(?:called|named|for|to)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^(?:please\s+)?(?:wake\s+me|remind\s+me)\s*(?:up\s*)?(?:to|about|for)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:to|about|for)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" ,.")[:120]


def _ensure_loaded() -> None:
    global _LOADED, _LOADED_PATH
    with _LOCK:
        path = _storage_path()
        if _LOADED and _LOADED_PATH == path:
            return
        if _LOADED_PATH is not None and _LOADED_PATH != path:
            for timer in _TIMERS.values():
                timer.cancel()
            _TIMERS.clear()
            _ALARMS.clear()
        _LOADED = True
        _LOADED_PATH = path
        if not path.exists():
            return
        try:
            raw_items = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        changed = False
        now = datetime.now()
        for item in raw_items if isinstance(raw_items, list) else []:
            try:
                alarm = ScheduledAlarm.from_dict(item)
            except (KeyError, TypeError, ValueError):
                changed = True
                continue

            if alarm.due_at <= now:
                changed = True
                _notify_alarm(alarm)
                continue

            _ALARMS[alarm.alarm_id] = alarm
            _schedule_timer_unlocked(alarm)

        if changed:
            _persist_unlocked()


def _schedule_timer_unlocked(alarm: ScheduledAlarm) -> None:
    old_timer = _TIMERS.pop(alarm.alarm_id, None)
    if old_timer is not None:
        old_timer.cancel()

    delay = max(0.0, (alarm.due_at - datetime.now()).total_seconds())
    timer = threading.Timer(delay, _fire_alarm, args=(alarm.alarm_id,))
    timer.daemon = True
    _TIMERS[alarm.alarm_id] = timer
    timer.start()


def _fire_alarm(alarm_id: str) -> None:
    with _LOCK:
        alarm = _ALARMS.pop(alarm_id, None)
        _TIMERS.pop(alarm_id, None)
        _persist_unlocked()

    if alarm is not None:
        _notify_alarm(alarm)


def _notify_alarm(alarm: ScheduledAlarm) -> None:
    print(f"\nKANCHA: {alarm.kind.capitalize()} due: {alarm.message}", flush=True)
    _ring_alarm()


def _ring_alarm(repeats: int = _RING_REPEATS) -> None:
    for _ in range(max(1, repeats)):
        if _ring_with_winsound():
            pass
        else:
            # Desktop sound helpers can block on broken audio setups; the bell gives an immediate cue.
            print("\a", end="", flush=True)
            _ring_with_system_sound()
        time.sleep(_RING_PAUSE_SECONDS)


def _ring_with_winsound() -> bool:
    try:
        import winsound
    except ImportError:
        return False

    try:
        winsound.Beep(1200, 450)
        winsound.Beep(900, 300)
        return True
    except RuntimeError:
        try:
            winsound.MessageBeep()
            return True
        except RuntimeError:
            return False


def _ring_with_system_sound() -> bool:
    for command in _system_sound_commands():
        executable = which(command[0])
        if not executable:
            continue

        args = [executable, *command[1:]]
        if any(arg.startswith("/") and not Path(arg).exists() for arg in args[1:]):
            continue

        try:
            completed = subprocess.run(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue

        if completed.returncode == 0:
            return True

    return False


def _system_sound_commands() -> tuple[tuple[str, ...], ...]:
    return (
        ("paplay", "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga"),
        ("paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"),
        ("canberra-gtk-play", "-i", "alarm-clock-elapsed"),
        ("canberra-gtk-play", "-i", "complete"),
        ("aplay", "/usr/share/sounds/alsa/Front_Center.wav"),
        ("play", "-q", "-n", "synth", "0.35", "sine", "1200"),
        ("spd-say", "Alarm"),
    )


def _persist_unlocked() -> None:
    path = _storage_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        alarms = [alarm.to_dict() for alarm in sorted(_ALARMS.values(), key=lambda item: item.due_at)]
        if not alarms:
            path.unlink(missing_ok=True)
            return
        path.write_text(json.dumps(alarms, indent=2), encoding="utf-8")
    except OSError:
        pass


def _storage_path() -> Path:
    configured = os.environ.get("KANCHHA_ALARMS_FILE")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "memory" / "alarms.json"


def _format_due(due_at: datetime) -> str:
    now = datetime.now()
    display_at = due_at
    if display_at.microsecond >= 500_000:
        display_at += timedelta(seconds=1)
    display_at = display_at.replace(microsecond=0)

    date_text: str
    if display_at.date() == now.date():
        date_text = "today"
    elif display_at.date() == (now + timedelta(days=1)).date():
        date_text = "tomorrow"
    else:
        date_text = display_at.strftime("%a, %b %d")

    delta_seconds = abs((due_at - now).total_seconds())
    show_seconds = bool(due_at.second or due_at.microsecond or delta_seconds < 60)
    time_format = "%I:%M:%S %p" if show_seconds else "%I:%M %p"
    return f"{date_text} at {display_at.strftime(time_format).lstrip('0')}"


_ensure_loaded()
