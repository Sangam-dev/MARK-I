"""Current date/time for LLM context.

Both LLM layers that reason about time — the Conversation LLM and the
Planner — get a snapshot of the wall clock in their system prompt, so
"what time is it", "remind me at 3pm" and "weather tomorrow" anchor
against real, current values instead of the model's training data.

The value is computed fresh for every turn (the system prompt is rebuilt
each turn) and always uses the machine's own local timezone, so it stays
correct across sessions and DST changes.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def _tz_name() -> str:
    """IANA timezone name where available, else the local abbreviation."""
    try:
        iana = Path("/etc/timezone").read_text().strip()
        if iana:
            return iana
    except OSError:
        pass
    return datetime.now().astimezone().tzname() or "local time"


def current_datetime_block() -> str:
    """Render the current local date/time as a system-prompt block."""
    now = datetime.now().astimezone()
    iso = now.isoformat(timespec="seconds")
    date_str = now.strftime("%A, %B %d, %Y")
    time_str = now.strftime("%I:%M:%S %p").lstrip("0")
    offset = now.strftime("%z")
    offset_str = f"UTC{offset[:3]}:{offset[3:]}" if offset else "UTC"
    return (
        "Current date and time (use this as the reference for anything "
        "involving time, dates, days of the week, or scheduling):\n"
        f"- Date: {date_str}\n"
        f"- Time: {time_str}\n"
        f"- Day of week: {now.strftime('%A')}\n"
        f"- Timezone: {_tz_name()} ({offset_str})\n"
        f"- ISO 8601: {iso}"
    )