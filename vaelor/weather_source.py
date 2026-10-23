"""Weather for a named place, read from Open-Meteo through the guarded fetch (VD-222 item 4).

The 1.5.5 live check asked AI Chat "temperature in Osaka today" with Web search
on: the keyless engines answered with Holly Hill, Florida and a bank called
Current. A weather reading is not a search problem. Open-Meteo answers it
directly and without a key: its geocoder turns the place words into candidate
places, and its forecast gives the current conditions and the next days in the
place's own time zone.

What leaves the appliance: only the place words (to the geocoder) and the
chosen place's coordinates and time zone (to the forecast) - never the question.
Both requests go through AI Chat's own guarded fetch (`research.fetch`, the
isolated research broker that re-proves every hop public HTTPS), with
:data:`OPEN_METEO_HOSTS` as the only domains this caller may open.

Choosing the place (research-asks-when-unsure): the geocoder returns up to ten
candidates and ignores a qualifier written into the name ("Osaka Japan" finds
nothing). So the place words are split into a name and a qualifier, the
qualifier is matched against each candidate's own country and regions, and the
most populous match wins - unless a second, different place is nearly as big
and nothing in the question decides, in which case the reply asks which one.
No place is named in this file; the candidates are the geocoder's.

The reply is deterministic, not the model's (Claude's choice, Adopted - see
:func:`weather_reply`).
"""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlencode

#: VD-222 item 4: the two Open-Meteo hosts AI Chat's weather answer may open,
#: and nothing else. Added deliberately as a new outside destination (owner,
#: VD-222 item 4: "a new outside destination, named in the Web search note").
#: They are this caller's allowlist only - handed to `research.fetch` as its
#: `allowed_domains` per request - so no other caller, agent or search gains
#: them. geocoding-api turns place words into candidate places; api serves the
#: forecast for the chosen coordinates.
OPEN_METEO_HOSTS = ("geocoding-api.open-meteo.com", "api.open-meteo.com")
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
#: The provider as the answer and its citation name it.
OPEN_METEO = "Open-Meteo"
#: Candidates asked of the geocoder: its own maximum page is 100; ten covers
#: every same-named town the live measurement saw ("Springfield", "Paris").
GEOCODE_CANDIDATES = 10
#: A runner-up place at least a third the leader's population is a real
#: contender. Measured 2026-10-19 from the Z2: Springfield, Missouri (170,188)
#: against Massachusetts (154,341) asks; Paris, France against Paris, Texas
#: (2.1 M / 25 k) and London against London, Ontario (9.0 M / 422 k) do not.
AMBIGUITY_RATIO = 3
#: Places offered when the reply asks which one was meant.
MAX_CHOICES = 4
#: Days of forecast read: today and tomorrow, and the rest of a week asked of.
FORECAST_DAYS = 7
#: Seconds one Open-Meteo request may take inside the tool's own limit.
REQUEST_SECONDS = 15.0
#: Evening, for "tonight": these local hours of today.
TONIGHT_FROM_HOUR = 18
#: Before this local hour, "tonight" is the night already under way.
NIGHT_ENDS_HOUR = 6
#: What today's chance covers: the hours still to come (review L8).
_LEFT_TODAY = " in the hours left today"
#: The widest UTC offset a place has (+14:00) and the narrowest (-12:00), with room.
MAX_UTC_OFFSET_SECONDS = 15 * 3600

CURRENT_FIELDS = (
    "temperature_2m", "apparent_temperature", "relative_humidity_2m", "precipitation",
    "rain", "showers", "snowfall", "weather_code", "wind_speed_10m",
)
HOURLY_FIELDS = ("temperature_2m", "precipitation_probability", "weather_code")
DAILY_FIELDS = (
    "weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
    "precipitation_probability_max",
)

#: What the weather answer did, as its metadata and audit say it.
WEATHER_ANSWERED = "weather"
WEATHER_ASKED = "weather_ask"
#: The question back when no town was named: "which place?" (or which town in a country).
WEATHER_WHERE = "weather_where"
WEATHER_NOT_FOUND = "weather_not_found"
WEATHER_FAILED = "weather_failed"

#: WMO weather interpretation codes, as Open-Meteo documents them.
WMO_CODES: Mapping[int, str] = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "light freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "light snow showers", 86: "snow showers",
    95: "thunderstorm", 96: "thunderstorm with light hail", 97: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}
#: Codes that mean water is falling, and those that mean snow.
RAIN_CODES = frozenset({51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 97, 99})
SNOW_CODES = frozenset({71, 73, 75, 77, 85, 86})

_WHEN_TOMORROW = "tomorrow"
_WHEN_TONIGHT = "tonight"
_WHEN_WEEK = "week"
_WHEN_NOW = "now"


class WeatherFetchError(RuntimeError):
    """An Open-Meteo request could not be read through the guarded fetch."""


Fetch = Callable[[str], Mapping[str, Any]]

#: What a read may fail with: the fetch itself, or a response whose values do
#: not fit (a non-finite number, a missing field). Each is said as a failed
#: read and the guarded search answers instead (review L7) - never raised
#: into the chat turn.
_READ_FAILURES = (WeatherFetchError, ValueError, TypeError, ArithmeticError, LookupError, AttributeError)


# ---------------------------------------------------------------------------
# Words and places.
# ---------------------------------------------------------------------------


def plain(text: Any) -> str:
    """Lower case, accents dropped, words only: "Île-de-France" -> "ile de france"."""
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    bare = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^\w]+", " ", bare.casefold()).split())


#: Words a weather question uses around a place that never name one (review
#: HIGH 1): times, "local", "outside", report words, polite filler, and the
#: words of a short reply ("the one in Ohio", "thanks!"). The geocoder answers
#: "today" with Todaya and "current" with Current, Bahamas, so these are
#: removed before anything is sent, and a place made only of them is no place.
NOT_PLACE_WORDS = frozenset({
    "today", "todays", "tonight", "tonights", "tomorrow", "tomorrows", "now", "right", "currently",
    "current", "local", "outside", "forecast", "forecasts", "report", "update", "updates", "alert",
    "alerts", "warning", "warnings", "radar", "near", "nearby", "me", "please", "here", "weather",
    "temperature", "temp", "check", "show", "tell", "give", "for", "in", "at", "next", "week",
    "weekend", "this", "morning", "afternoon", "evening", "home", "work", "office", "school", "sea",
    "area", "thanks", "thank", "you", "ok", "okay", "yes", "no", "great", "cool", "the", "one", "from",
    "about", "and", "what", "how",
})
#: The time a removed word asks for, when the question did not already say one.
TIME_WORDS = {
    "tonight": "tonight", "tonights": "tonight", "tomorrow": "tomorrow", "tomorrows": "tomorrow",
    "week": "week", "weekend": "week",
}
#: Hardware nouns that are never a place, beside the ones the appliance
#: vocabulary (`chat_appliance_scope.is_appliance_question`) already knows
#: alone ("cpu", "gpu", "nvme", "disk", "box", "machine", "appliance").
COMPONENT_WORDS = frozenset({"npu", "server", "servers", "chassis", "ssd", "hdd", "rack", "psu", "motherboard"})


def place_words(place: Any) -> Dict[str, Any]:
    """The place words with every never-a-place word taken out.

    Returns ``{"place": what is left (commas kept), "when": a time word that
    was taken out, or ""}``.
    """
    kept: List[str] = []
    when = ""
    for token in re.findall(r"[^\s,]+|,", str(place or "")):
        word = plain(re.sub(r"'s$", "", token, flags=re.IGNORECASE))
        if token == ",":
            kept.append(",")
        elif word in NOT_PLACE_WORDS:
            when = when or TIME_WORDS.get(word, "")
        elif word:
            kept.append(token)
    text = " ".join(kept).replace(" ,", ",").strip(" ,")
    return {"place": re.sub(r",(?:\s*,)+", ",", text), "when": when}


def when_of(word: Any) -> str:
    """The forecast a time word asks for: tomorrow, tonight, the week, or now."""
    text = plain(word)
    if text == "tomorrow":
        return _WHEN_TOMORROW
    if text in {"tonight", "this evening"}:
        return _WHEN_TONIGHT
    if text.split()[-1:] in (["week"], ["weekend"]):
        return _WHEN_WEEK
    return _WHEN_NOW


def name_attempts(place: str) -> List[Tuple[str, str, bool]]:
    """The (name, qualifier, explicit) readings of the place words, most literal first.

    "Paris, Texas" is one reading, and its qualifier is explicit. "new york" is
    first asked whole, since a place name may be several words; "osaka japan"
    then as "osaka" qualified by "japan" (the geocoder finds nothing for the
    whole). At most three readings, so at most three geocoder requests.
    """
    if "," in place:
        name, _, qualifier = place.partition(",")
        return [(" ".join(name.split()), " ".join(qualifier.replace(",", " ").split()), True)]
    words = place.split()
    return [(" ".join(words[:cut]), " ".join(words[cut:]), False) for cut in range(len(words), 0, -1)]


def _region_words(candidate: Mapping[str, Any]) -> List[str]:
    return [plain(candidate.get(key)) for key in (
        "country", "country_code", "admin1", "admin2", "admin3", "admin4") if candidate.get(key)]


def matches_qualifier(candidate: Mapping[str, Any], qualifier: str) -> bool:
    """Whether every word of the qualifier names the candidate's country or a region of it."""
    wanted = plain(qualifier).split()
    if not wanted:
        return True
    regions = set(" ".join(_region_words(candidate)).split())
    return all(word in regions for word in wanted)


def label(candidate: Mapping[str, Any]) -> str:
    """"Osaka, Japan", "Paris, Texas, United States": the place, its region, its country.

    The country is always named - the code when the geocoder gives no name.
    """
    parts: List[str] = []
    country = candidate.get("country") or candidate.get("country_code")
    for value in (candidate.get("name"), candidate.get("admin1"), country):
        value = " ".join(str(value or "").split())
        if value and value not in parts:
            parts.append(value)
    return ", ".join(parts)


def _population(candidate: Mapping[str, Any]) -> int:
    value = _number(candidate.get("population"))
    return int(min(value, 1e10)) if value is not None and value > 0 else 0


def _feature(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("feature_code") or "").upper()


def is_region(candidate: Mapping[str, Any]) -> bool:
    """A country or a first-level region ("Georgia", "Jordan"): too big for one reading."""
    return _feature(candidate).startswith(("PCL", "ADM1", "ADM2"))


def is_town(candidate: Mapping[str, Any]) -> bool:
    """A populated place (GeoNames PPL*), not an airport, a mountain or a dam."""
    return _feature(candidate).startswith("PPL") or not _feature(candidate)


def bears_name(candidate: Mapping[str, Any], name: str) -> bool:
    """Whether a candidate's own name is the name asked, or that name's longer form.

    "Frankfurt am Main" and "Frankfurt (Oder)" bear "frankfurt"; "New York"
    bears "new york city" (its own name, a whole word, opens the phrase). A
    near spelling ("Todaya" for "today", "Current" for "current weather") does
    not: the geocoder returns those, and answering with them is answering a
    question nobody asked (VD-222 item 4 review, HIGH 1).
    """
    own, asked = plain(candidate.get("name")), plain(name)
    if not own or not asked:
        return False
    if own == asked or own.startswith(asked + " "):
        return True
    return len(own.split()) >= 2 and asked.startswith(own + " ")


def _distinct(candidates: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """One candidate per (region, country): the geocoder lists some towns twice."""
    seen = set()
    kept = []
    for candidate in candidates:
        key = (plain(candidate.get("admin1")), plain(candidate.get("country")))
        if key not in seen:
            seen.add(key)
            kept.append(candidate)
    return kept


def choose_place(results: Sequence[Mapping[str, Any]], name: str, qualifier: str) -> Dict[str, Any]:
    """Pick the place meant, or what to ask.

    Returns ``{"place": candidate}``; ``{"choices": [...]}`` (ask which one),
    with ``"unmatched": True`` when the qualifier named none of them;
    ``{"region": candidate}`` when the name is a country or a large region
    (ask for a town in it); or ``{}`` (no candidate bears the name).
    Exact names and their longer forms are ranked together by population,
    so "Frankfurt" is Frankfurt am Main, not the hamlet of that exact name.
    """
    named = [item for item in results if isinstance(item, Mapping) and _coordinates(item)
             and bears_name(item, name)]
    fitting = [item for item in named if matches_qualifier(item, qualifier)]
    if not fitting:
        towns = [item for item in named if is_town(item)]
        # A qualifier nothing matches is not overridden by a guess.
        return {"choices": _distinct(sorted(towns, key=_population, reverse=True))[:MAX_CHOICES],
                "unmatched": True} if (qualifier and towns) else {}
    ranked = sorted(fitting, key=_population, reverse=True)
    if is_region(ranked[0]):
        return {"region": ranked[0]}
    ranked = _distinct([item for item in ranked if is_town(item)])
    if not ranked:
        return {}
    if len(ranked) == 1:
        return {"place": ranked[0]}
    leader, runner_up = _population(ranked[0]), _population(ranked[1])
    if leader and runner_up * AMBIGUITY_RATIO < leader:
        return {"place": ranked[0]}
    return {"choices": ranked[:MAX_CHOICES]}


def _coordinates(candidate: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    try:
        latitude = float(candidate.get("latitude"))
        longitude = float(candidate.get("longitude"))
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(latitude) and math.isfinite(longitude)):
        return None
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        return None
    return latitude, longitude


# ---------------------------------------------------------------------------
# The two requests.
# ---------------------------------------------------------------------------


def geocode_url(name: str) -> str:
    return "{}?{}".format(GEOCODE_URL, urlencode({
        "name": name[:100], "count": GEOCODE_CANDIDATES, "language": "en", "format": "json"}))


def _time_zone(candidate: Mapping[str, Any]) -> str:
    zone = str(candidate.get("timezone") or "")
    return zone if re.fullmatch(r"[A-Za-z_]+(?:/[A-Za-z0-9_+-]+){0,2}", zone) else "auto"


def forecast_url(candidate: Mapping[str, Any]) -> str:
    latitude, longitude = _coordinates(candidate) or (0.0, 0.0)
    return "{}?{}".format(FORECAST_URL, urlencode({
        "latitude": "{:.5f}".format(latitude), "longitude": "{:.5f}".format(longitude),
        "current": ",".join(CURRENT_FIELDS), "hourly": ",".join(HOURLY_FIELDS),
        "daily": ",".join(DAILY_FIELDS), "timezone": _time_zone(candidate),
        "forecast_days": FORECAST_DAYS,
    }))


def guarded_fetch(tools: Any, *, actor: str, administrator: bool) -> Fetch:
    """One Open-Meteo read through AI Chat's guarded fetch, allowlisted to Open-Meteo."""

    def fetch(url: str) -> Mapping[str, Any]:
        # Imported here: the tool registry imports half the control plane.
        from .assistant_tools import AssistantToolError
        try:
            ran = tools.run(
                "research.fetch", {"url": url}, actor=actor, administrator=administrator,
                granted_scopes=("research:read",),
                web_access={"enabled": True, "allowed_domains": list(OPEN_METEO_HOSTS)},
                timeout_seconds=REQUEST_SECONDS,
            )
        except AssistantToolError as error:
            raise WeatherFetchError(" ".join(str(error).split())[:300]) from error
        evidence = (ran.get("result") or {}).get("evidence") if isinstance(ran, Mapping) else None
        if not isinstance(evidence, Mapping):
            raise WeatherFetchError("Open-Meteo returned no readable evidence.")
        return evidence

    return fetch


def _document(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = evidence.get("metadata") if isinstance(evidence.get("metadata"), Mapping) else {}
    document = metadata.get("document")
    if metadata.get("format") != "json" or not isinstance(document, Mapping):
        raise WeatherFetchError("Open-Meteo did not answer with JSON.")
    if document.get("error"):
        raise WeatherFetchError("Open-Meteo refused the request: {}".format(
            " ".join(str(document.get("reason") or "reason not stated").split())[:200]))
    return document


def _read_at(evidence: Mapping[str, Any]) -> datetime:
    source = evidence.get("source") if isinstance(evidence.get("source"), Mapping) else {}
    try:
        read = datetime.fromisoformat(str(source.get("fetched_at") or ""))
    except ValueError:
        read = datetime.now(timezone.utc)
    return read if read.tzinfo else read.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# The answer.
# ---------------------------------------------------------------------------


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def celsius(value: Any) -> str:
    """"19.8 °C (67.6 °F)": metric first, Fahrenheit beside it."""
    number = _number(value)
    if number is None:
        return "unknown"
    return "{:.1f} °C ({:.1f} °F)".format(number, number * 9 / 5 + 32)


def _wind(value: Any) -> str:
    number = _number(value)
    return "unknown" if number is None else "{:.0f} km/h ({:.0f} mph)".format(number, number / 1.609344)


def _sky(code: Any) -> str:
    number = _number(code)
    return WMO_CODES.get(int(number), "conditions code {}".format(int(number))) if number is not None else ""


def _series(document: Mapping[str, Any], block: str) -> Mapping[str, Any]:
    value = document.get(block)
    return value if isinstance(value, Mapping) else {}


def _at(values: Any, index: int) -> Any:
    return values[index] if isinstance(values, list) and 0 <= index < len(values) else None


def _day_line(daily: Mapping[str, Any], index: int, title: str, chance: Optional[float] = None,
              chance_span: str = "") -> str:
    """One day: its high and low, the worst weather expected, and the chance of precipitation.

    The daily code is the day's worst (review L8), so it is said as that, not
    as the sky. ``chance`` overrides the whole day's chance - for today, the
    hours that are left (``chance_span`` says so).
    """
    if _at(daily.get("time"), index) is None:
        return ""
    sky = _sky(_at(daily.get("weather_code"), index))
    day_chance = _number(_at(daily.get("precipitation_probability_max"), index))
    total = _number(_at(daily.get("precipitation_sum"), index))
    wet = ""
    if chance is not None or day_chance is not None:
        wet = ", {:.0f}% chance of precipitation{}".format(
            chance if chance is not None else day_chance, chance_span if chance is not None else "")
        if total:
            wet += " (about {:.1f} mm over the day)".format(total)
    return "{}: high {}, low {}{}{}.".format(
        title, celsius(_at(daily.get("temperature_2m_max"), index)),
        celsius(_at(daily.get("temperature_2m_min"), index)),
        ", at worst " + sky if sky else "", wet)


def _local_hour(document: Mapping[str, Any]) -> Tuple[str, int]:
    """The place's own date and hour now, from the current reading's time."""
    stamp = str(_series(document, "current").get("time") or "")
    return stamp[:10], int(stamp[11:13]) if stamp[11:13].isdigit() else 0


def _hours(document: Mapping[str, Any], first: int, last: int) -> List[int]:
    """Hourly rows of the place's today from hour ``first`` up to ``last`` (exclusive)."""
    today, _hour = _local_hour(document)
    return [
        index for index, stamp in enumerate(_series(document, "hourly").get("time") or [])
        if str(stamp)[:10] == today and str(stamp)[11:13].isdigit() and first <= int(str(stamp)[11:13]) < last
    ]


def _chance_over(document: Mapping[str, Any], rows: Sequence[int]) -> Optional[float]:
    hourly = _series(document, "hourly")
    chances = [value for value in (_number(_at(hourly.get("precipitation_probability"), i)) for i in rows)
               if value is not None]
    return max(chances) if chances else None


def _now_lines(current: Mapping[str, Any], where: str, ask: str) -> List[str]:
    code = _number(current.get("weather_code"))
    falling_rain = bool(_number(current.get("rain")) or _number(current.get("showers"))) or (
        code is not None and int(code) in RAIN_CODES)
    falling_snow = bool(_number(current.get("snowfall"))) or (code is not None and int(code) in SNOW_CODES)
    lead = ""
    if ask == "raining":
        lead = "Yes, it is raining in {} now. ".format(where) if falling_rain else (
            "No, it is not raining in {} now. ".format(where))
    elif ask == "snowing":
        lead = "Yes, it is snowing in {} now. ".format(where) if falling_snow else (
            "No, it is not snowing in {} now. ".format(where))
    sky = _sky(current.get("weather_code"))
    humidity = _number(current.get("relative_humidity_2m"))
    return ["{}{}: {}{}, feels like {}{}, wind {}.".format(
        lead, "Now" if lead else "Now in " + where, celsius(current.get("temperature_2m")), ", " + sky if sky else "",
        celsius(current.get("apparent_temperature")),
        ", humidity {:.0f}%".format(humidity) if humidity is not None else "",
        _wind(current.get("wind_speed_10m")))]


def _tonight(document: Mapping[str, Any]) -> Tuple[str, Optional[float]]:
    """Tonight's line, and its highest chance of precipitation.

    Evening is from 18:00 to midnight; after midnight and before dawn,
    "tonight" is the rest of this night, up to 06:00 (review L10).
    """
    hourly = _series(document, "hourly")
    _today, now_hour = _local_hour(document)
    first, last = (now_hour, NIGHT_ENDS_HOUR) if now_hour < NIGHT_ENDS_HOUR else (
        max(TONIGHT_FROM_HOUR, now_hour), 24)
    rows = _hours(document, first, last)
    temperatures = [value for value in (_number(_at(hourly.get("temperature_2m"), i)) for i in rows)
                    if value is not None]
    if not temperatures:
        return "", None
    codes = [int(value) for value in (_number(_at(hourly.get("weather_code"), i)) for i in rows)
             if value is not None]
    skies = [WMO_CODES.get(code, "") for code in sorted(set(codes), key=codes.index)]
    chance = _chance_over(document, rows)
    return "Tonight ({:02d}:00 to {:02d}:00): {} to {}{}{}.".format(
        first, last % 24, celsius(min(temperatures)), celsius(max(temperatures)),
        ", " + " then ".join(sky for sky in skies[:3] if sky) if any(skies) else "",
        ", {}{:.0f}% chance of precipitation".format("up to " if chance else "", chance)
        if chance is not None else ""), chance


def _forecast_lead(chance: Optional[float], will: str, period: str) -> str:
    """"Rain is likely tomorrow." - only for "will it rain/snow ...", from the forecast's chance."""
    if will not in {"rain", "snow"} or chance is None:
        return ""
    likely = "likely" if chance >= 60 else "unlikely" if chance <= 20 else "possible"
    return "{} is {} {} ({:.0f}% chance of precipitation). ".format(will.capitalize(), likely, period, chance)


def _day_name(stamp: Any) -> str:
    """"Wed 21 Oct" from Open-Meteo's "2026-10-21"; "" when it is not a date."""
    try:
        return datetime.strptime(str(stamp or ""), "%Y-%m-%d").strftime("%a %d %b").replace(" 0", " ")
    except ValueError:
        return ""


def _tomorrow_date(daily: Mapping[str, Any], today: str, read_at: datetime) -> str:
    """Whose tomorrow (review M4): the place's, dated, and said when this
    appliance's own date is not the place's today."""
    day = _day_name(_at(daily.get("time"), 1)) or "the next day"
    here = read_at.astimezone().strftime("%Y-%m-%d")
    if today and today != here:
        return "{}, the day after today there, which is {}".format(day, _day_name(today) or today)
    return "{}, the place's own date".format(day)


def _clock(moment: datetime, offset_seconds: int, zone: str) -> str:
    local = moment.astimezone(timezone(timedelta(seconds=offset_seconds)))
    return "{} {} on {} ({} UTC)".format(
        local.strftime("%H:%M"), zone or "local time", local.strftime("%a %d %b %Y").replace(" 0", " "),
        moment.astimezone(timezone.utc).strftime("%H:%M"))


def weather_reply(document: Mapping[str, Any], place: Mapping[str, Any], when: str, ask: str,
                  read_at: datetime) -> Tuple[str, str]:
    """The answer, and the title its citation carries.

    Deterministic, not the model's (Claude's choice, Adopted): every question
    that reaches here is one of the tight live shapes - a reading of a named
    place - which the source answers whole, so a model adds nothing but a
    chance to misquote the number this whole row exists to get right. It also
    answers when AI Chat's model is asleep, busy or unreachable, and the place
    never reaches a hosted model. The cost: a fixed English form.
    """
    where = label(place)
    current = _series(document, "current")
    daily = _series(document, "daily")
    lines: List[str] = []
    def chance(index: int) -> Optional[float]:
        return _number(_at(daily.get("precipitation_probability_max"), index))

    today, now_hour = _local_hour(document)
    left_today = _chance_over(document, _hours(document, now_hour, 24))
    if when == _WHEN_TOMORROW:
        lines.append(_forecast_lead(chance(1), ask, "tomorrow"))
        lines.append(_day_line(daily, 1, "Tomorrow in {} ({})".format(where, _tomorrow_date(daily, today, read_at))))
    elif when == _WHEN_TONIGHT:
        tonight, tonight_chance = _tonight(document)
        lines.append(_forecast_lead(tonight_chance, ask, "tonight"))
        lines.extend(_now_lines(current, where, ask))
        lines.append(tonight or _day_line(daily, 0, "Today", left_today, _LEFT_TODAY))
    else:
        lines.append(_forecast_lead(left_today if left_today is not None else chance(0), ask, "today"))
        lines.extend(_now_lines(current, where, ask))
        lines.append(_day_line(daily, 0, "Today", left_today, _LEFT_TODAY))
        if when == _WHEN_WEEK:
            for index in range(1, FORECAST_DAYS):
                title = _day_name(_at(daily.get("time"), index))
                if title:
                    lines.append(_day_line(daily, index, title))
    offset = int(_number(document.get("utc_offset_seconds")) or 0)
    offset = offset if abs(offset) <= MAX_UTC_OFFSET_SECONDS else 0
    zone = " ".join(str(document.get("timezone_abbreviation") or "").split())[:12]
    read = _clock(read_at, offset, zone)
    title = "{} forecast for {}, read at {}".format(OPEN_METEO, where, read)
    observed = str(current.get("time") or "")
    # [S1] is the marker AI Chat links to the answer's one citation (`read_weather`).
    lines.append("Source [S1]: {}{}.".format(
        title, "; conditions as of {} local time".format(observed[11:16]) if observed[11:16] else ""))
    # Each part is stripped before joining: a lead ends in a space (review L10).
    return " ".join(line.strip() for line in lines[:-1] if line.strip()) + "\n\n" + lines[-1], title


def ask_where(region: str = "", when: str = _WHEN_NOW) -> str:
    """The question when no town was named, or only a country or a large region."""
    day = {_WHEN_TOMORROW: "tomorrow's ", _WHEN_TONIGHT: "tonight's "}.get(when, "the ")
    if region:
        return ("{} is a country or a large region, and one weather reading cannot cover it. "
                "Which town or city in {} do you mean?").format(region, region)
    return "Which place do you want {}weather for? Name a town or city.".format(day)


def ask_which(name: str, choices: Sequence[Mapping[str, Any]]) -> str:
    """The question when two different places fit and nothing decides."""
    names = [label(choice) for choice in choices]
    # The geocoder's own spelling of the name, not the owner's typing.
    name = str(choices[0].get("name") or name) if choices else name
    listed = "; ".join(names[:-1]) + "; or " + names[-1] if len(names) > 1 else names[0]
    example = str(choices[0].get("admin1") or choices[0].get("country") or "") if choices else ""
    return (
        "{} found more than one place called {}, and nothing in the question says which: {}. "
        "Which one do you mean? Reply with its region or country{}, or ask again as "
        "\"weather in {}, <region>\"."
    ).format(OPEN_METEO, name, listed, ' (for example "{}")'.format(example) if example else "", name)


def read_weather(fetch: Fetch, place: str, when: str = _WHEN_NOW, ask: str = "",
                 *, chosen: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Geocode ``place`` (or use the ``chosen`` candidate), then read its weather.

    Returns ``{"outcome", "place", "requests", ...}``; ``answer`` and
    ``citation`` for an answer, ``answer`` and ``choices`` for a question back,
    ``reason`` for a failure. Never raises for an Open-Meteo failure.
    """
    requests: List[str] = []
    outcome: Dict[str, Any] = {"outcome": WEATHER_NOT_FOUND, "place": place, "requests": requests,
                               "when": when, "ask": ask}
    try:
        picked = dict(chosen) if chosen and _coordinates(chosen) else None
        if picked is None:
            for name, qualifier, explicit in name_attempts(" ".join(str(place or "").split())[:100]):
                url = geocode_url(name)
                requests.append(url)
                results = _document(fetch(url)).get("results")
                verdict = choose_place(results if isinstance(results, list) else [], name, qualifier)
                if verdict.get("place"):
                    picked = dict(verdict["place"])
                    break
                if verdict.get("region"):
                    return {**outcome, "outcome": WEATHER_WHERE, "place": str(verdict["region"].get("name") or name),
                            "answer": ask_where(str(verdict["region"].get("name") or name))}
                # Asked about only when the owner wrote the qualifier after a
                # comma: a space-joined word nothing matches ("osaka ignore",
                # "Paris Hilton") is not trusted as one, so the shorter reading
                # is tried and, failing it, the search answers instead.
                if verdict.get("choices") and (explicit or not verdict.get("unmatched")):
                    return _asked(outcome, name, verdict["choices"])
        if picked is None:
            return outcome
        url = forecast_url(picked)
        requests.append(url)
        evidence = fetch(url)
        answer, title = weather_reply(_document(evidence), picked, when, ask, _read_at(evidence))
    except _READ_FAILURES as error:
        return {**outcome, "outcome": WEATHER_FAILED, "reason": " ".join(str(error).split())[:300]
                or type(error).__name__}
    return {
        **outcome, "outcome": WEATHER_ANSWERED, "place": label(picked), "answer": answer,
        "citation": {
            "id": "web-1", "collection_id": "web", "collection": "Web", "document_id": url,
            "document": title, "chunk": 1, "excerpt": answer.split("\n\n", 1)[0][:400], "url": url,
        },
    }


def _asked(outcome: Dict[str, Any], name: str, candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    choices = [_choice(item) for item in candidates]
    # ``place`` is the name asked about, as the geocoder spells it: the stored
    # answer keeps it, and the owner's reply is read against it.
    spelled = str(choices[0].get("name") or name) if choices else name
    return {**outcome, "outcome": WEATHER_ASKED, "place": spelled, "choices": choices,
            "answer": ask_which(name, choices)}


def read_weather_reply(fetch: Fetch, name: str, reply: str, when: str = _WHEN_NOW,
                       ask: str = "") -> Dict[str, Any]:
    """The owner's reply to "which one?": the place it names, read; or not found.

    The question back is not stored with its choices (a stored answer keeps
    only its search summary), so the name is geocoded again and the reply is
    read against the same candidates, ranked the same way.
    """
    url = geocode_url(name)
    outcome: Dict[str, Any] = {"outcome": WEATHER_NOT_FOUND, "place": name, "requests": [url],
                               "when": when, "ask": ask}
    try:
        results = _document(fetch(url)).get("results")
    except WeatherFetchError as error:
        return {**outcome, "outcome": WEATHER_FAILED, "reason": str(error)}
    usable = [item for item in (results if isinstance(results, list) else [])
              if isinstance(item, Mapping) and _coordinates(item)]
    named = [item for item in usable if bears_name(item, name) and is_town(item)]
    choices = [_choice(item) for item in _distinct(sorted(named, key=_population, reverse=True))]
    picked = pick_from_reply(reply, choices)
    if picked is None:
        return outcome
    read = read_weather(fetch, label(picked), when, ask, chosen=picked)
    return {**read, "requests": [url, *read["requests"]]}


def _choice(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """A candidate as the question back stores it, so a reply can pick it."""
    latitude, longitude = _coordinates(candidate) or (0.0, 0.0)
    return {
        "name": str(candidate.get("name") or "")[:100], "admin1": str(candidate.get("admin1") or "")[:100],
        "country": str(candidate.get("country") or "")[:100],
        "country_code": str(candidate.get("country_code") or "")[:4],
        "latitude": latitude, "longitude": longitude, "timezone": _time_zone(candidate),
    }


def pick_from_reply(message: Any, choices: Any) -> Optional[Dict[str, Any]]:
    """The one offered place a short reply names ("Illinois", "the one in Ohio"), or None.

    Only a short reply, and only when exactly one choice's region or country is
    named in it - "United States" names them all, and is asked again by the
    caller's ordinary path.
    """
    words = plain(message).split()
    if not words or len(words) > 6 or not isinstance(choices, list):
        return None
    said = " {} ".format(" ".join(words))
    named = [
        choice for choice in choices if isinstance(choice, Mapping) and _coordinates(choice) and any(
            region and " {} ".format(region) in said
            for region in (plain(choice.get("admin1")), plain(choice.get("country"))))
    ]
    return dict(named[0]) if len(named) == 1 else None
