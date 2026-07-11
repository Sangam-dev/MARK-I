from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class WeatherResult:
    success: bool
    message: str


_WEATHER_WORDS_RE = re.compile(
    r"\b(?:what(?:'s| is)?|how(?:'s| is)?|tell me|show me|check|current|the|weather|forecast|"
    r"temperature|temp|raining|rain|outside|like|in|at|for|of|today|now|please)\b",
    re.IGNORECASE,
)


def extract_weather_place(text: str) -> str:
    cleaned = " ".join(text.strip().strip(".!?").split())
    if not cleaned:
        return ""

    match = re.search(
        r"\b(?:weather|forecast|temperature|temp|raining|rain)\b.*?\b(?:in|at|for)\s+(?P<place>.+)$",
        cleaned,
        re.IGNORECASE,
    )
    if match:
        return _clean_place(match.group("place"))

    match = re.search(r"\b(?:in|at|for)\s+(?P<place>[\w .,'-]+?)\s*(?:weather|forecast|temperature|temp)\b", cleaned, re.IGNORECASE)
    if match:
        return _clean_place(match.group("place"))

    place = _WEATHER_WORDS_RE.sub(" ", cleaned)
    return _clean_place(place)


def get_weather(query: str, date: str | None = None, units: str | None = None) -> WeatherResult:
    place = extract_weather_place(query) or _clean_place(query)
    if not place or place.lower() in {"here", "my location"}:
        return WeatherResult(False, "Tell me which place to check the weather for.")

    url_place = urllib.parse.quote(place)
    url = f"https://wttr.in/{url_place}?format=j1"
    request = urllib.request.Request(url, headers={"User-Agent": "Kanchha/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return WeatherResult(False, f"I couldn't fetch weather for {place} right now: {exc}.")

    try:
        return WeatherResult(True, _format_weather(place, payload))
    except (KeyError, IndexError, TypeError, ValueError):
        return WeatherResult(False, f"I couldn't understand the weather response for {place}.")


def _format_weather(place: str, payload: dict[str, Any]) -> str:
    current = payload["current_condition"][0]
    description = current["weatherDesc"][0]["value"]
    temp_c = current["temp_C"]
    temp_f = current["temp_F"]
    feels_c = current["FeelsLikeC"]
    humidity = current["humidity"]
    wind_kmph = current["windspeedKmph"]
    return (
        f"Weather in {place}: {description}, {temp_c}C/{temp_f}F, "
        f"feels like {feels_c}C, humidity {humidity}%, wind {wind_kmph} km/h."
    )


def _clean_place(place: str) -> str:
    cleaned = " ".join(place.strip(" ,.?").split())
    cleaned = re.sub(r"\b(?:right now|now|today|please)\b", "", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.strip(" ,.?").split())
