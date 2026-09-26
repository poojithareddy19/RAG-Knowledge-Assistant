"""Which questions go to an MCP tool rather than to generated SQL.

Most questions are answered by the SQL generator, because most questions have
a shape nobody wrote down in advance. A few do not. "What are the nearest ARGO
floats to this location" is one point in and floats ranked by distance out,
every time, and the MCP server answers it with one fixed, parameterised
statement. Sending it through the model instead would spend thirty seconds of
generation to reproduce a query that already exists, and would add the chance
of getting the distance or the ranking wrong.

So this runs before the router, and only matches what it can match exactly.
It is written to decline rather than to guess: a question carrying a date or a
measured quantity alongside the location is left to the SQL path, which can
honour the extra condition, and a tool that returns the right floats for the
wrong year is a wrong answer delivered confidently.

Detection is rules, not a model call, for the same reason the router tries its
rules first: a question that matches is recognisable from its words, and one
that does not costs nothing here.
"""

from __future__ import annotations

import re

# Asked for the nearest floats but gave no position. Returned instead of a tool
# name so the caller can ask for one rather than guess where "here" is.
NEEDS_LOCATION = "needs_location"

_NUMBER = r"[-+]?\d{1,3}(?:\.\d+)?"

# 10.5N 65.2E, 10.5°N, 65.2°E, 12 S 75 E
_HEMISPHERE = re.compile(
    rf"(?<![\w.])({_NUMBER})\s*°?\s*([NS])\b[\s,;/]*({_NUMBER})\s*°?\s*([EW])\b",
    re.I,
)

# lat 10.5 lon 65.2, latitude: -12, longitude = 75.5
_NAMED = re.compile(
    rf"\blat(?:itude)?\s*[:=]?\s*({_NUMBER})(?![\w.])"
    rf".{{0,40}}?\blon(?:g|gitude)?\s*[:=]?\s*({_NUMBER})(?![\w.])",
    re.I,
)

# (10.5, 65.2) or 10.5, 65.2. Only consulted once the question has already
# said it wants the nearest floats, because a bare pair of numbers means
# nothing on its own.
_PAIR = re.compile(rf"(?<![\w.])({_NUMBER})\s*,\s*({_NUMBER})(?![\w.])")

_NEAREST = re.compile(r"\b(nearest|closest|near(?:by)?|around|proximity)\b", re.I)
_FLOATS = re.compile(r"\bfloats?\b", re.I)

# "near" and "around" are ordinary words; they only mean "rank by distance"
# when the question also points at a place.
_POINTED = re.compile(
    r"\b(nearest|closest)\b|\bthis location\b|\bmy location\b|\bhere\b"
    r"|\bto\s+(?:the\s+)?(?:point|position|coordinates?)\b",
    re.I,
)

# Anything that narrows the question beyond "which floats, how far". A year, a
# month, a measured quantity or a statistic means the tool's fixed shape would
# drop part of what was asked.
_EXTRA_CONDITION = re.compile(
    r"\b(19|20)\d{2}\b"
    r"|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b"
    r"|\b(temperature|salinity|oxygen|chlorophyll|nitrate|ph|pressure|depth"
    r"|average|mean|median|count|how many|between|since|last|during)\b",
    re.I,
)

_PROFILE_OF_FLOAT = re.compile(
    r"^\s*(?:please\s+)?(?:show|plot|display|draw|get|give\s+me|fetch)\b"
    r".{0,40}?\bprofiles?\b.{0,20}?\bfloat\s*(?:#|no\.?|id)?\s*(\d{5,8})\b",
    re.I,
)

_CYCLE = re.compile(r"\bcycle\s*(?:no\.?|number|#)?\s*(\d{1,4})\b", re.I)


def coordinates(question: str) -> tuple[float, float] | None:
    """A (latitude, longitude) named in the question, or None."""

    text = question or ""

    match = _HEMISPHERE.search(text)

    if match:
        lat = float(match.group(1)) * (-1 if match.group(2).upper() == "S" else 1)
        lon = float(match.group(3)) * (-1 if match.group(4).upper() == "W" else 1)
        return _checked(lat, lon)

    match = _NAMED.search(text)

    if match:
        return _checked(float(match.group(1)), float(match.group(2)))

    return None


def _checked(lat: float, lon: float) -> tuple[float, float] | None:
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return lat, lon

    return None


def match_tool(question: str) -> tuple[str, dict] | None:
    """The MCP tool that answers this question exactly, with its arguments.

    Returns ``(NEEDS_LOCATION, {})`` for a nearest-floats question with no
    position, and None for everything the SQL path should handle.
    """

    text = question or ""

    profile = _PROFILE_OF_FLOAT.search(text)

    if profile:
        arguments = {"float_id": profile.group(1)}
        cycle = _CYCLE.search(text)

        if cycle:
            arguments["cycle"] = int(cycle.group(1))

        return "get_profile", arguments

    if not (_NEAREST.search(text) and _FLOATS.search(text) and _POINTED.search(text)):
        return None

    if _EXTRA_CONDITION.search(text):
        return None

    point = coordinates(text)

    if point is None:
        pair = _PAIR.search(text)

        if pair:
            point = _checked(float(pair.group(1)), float(pair.group(2)))

    if point is None:
        return NEEDS_LOCATION, {}

    return "nearest_floats", {"latitude": point[0], "longitude": point[1]}


# Words that make a question depend on where the user is. A question with none
# of them is sent as typed even when a location is set, so a position set for
# one question does not quietly narrow the next.
_REFERS_TO_A_PLACE = re.compile(
    r"\b(this location|my location|this point|here|nearest|closest|nearby)\b",
    re.I,
)


def with_location(question: str, location: tuple[float, float] | None) -> str:
    """The question with a position appended, when it asks about one and lacks it.

    ``location`` comes from the page's location panel and is None when that is
    switched off. A question that already names a position keeps its own: the
    one typed is more specific than the one set in a panel.
    """
    if location is None or coordinates(question) is not None:
        return question

    if not _REFERS_TO_A_PLACE.search(question or ""):
        return question

    latitude, longitude = location

    return f"{question} (lat {latitude:.4f} lon {longitude:.4f})"
