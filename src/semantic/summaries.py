"""Describe, in English, what the measurements tables actually contain.

The schema catalog tells the model which columns exist. It cannot tell it that
float 1901393 is real, worked the Arabian Sea, and stopped reporting in 2021.
Without that, "what did 1901393 measure off Goa" is answered by a model
guessing whether such a float exists.

One summary per float and one per region, each written as a sentence rather
than a row, because a sentence is what an embedding model reads well.

Writing the text is kept separate from reading the database so the wording,
which is the part that decides whether retrieval works, can be tested without
any infrastructure.
"""

from __future__ import annotations

FLOAT_ROWS = """
SELECT
    f.float_id,
    coalesce(f.platform, '') AS platform,
    coalesce(f.project, '')  AS project,
    count(DISTINCT p.profile_id) AS profiles,
    min(p.obs_time)::date AS first_obs,
    max(p.obs_time)::date AS last_obs,
    array_agg(DISTINCT p.region)
        FILTER (WHERE p.region IS NOT NULL) AS regions,
    round(min(m.pressure_dbar)::numeric, 1) AS min_pressure,
    round(max(m.pressure_dbar)::numeric, 1) AS max_pressure,
    round(
        avg(m.temperature_c) FILTER (WHERE m.pressure_dbar < 10)::numeric, 2
    ) AS mean_surface_temp,
    round(
        avg(m.salinity_psu) FILTER (WHERE m.pressure_dbar < 10)::numeric, 2
    ) AS mean_surface_salinity,
    count(m.oxygen_umol_kg) AS oxygen_n,
    count(m.chlorophyll_mg_m3) AS chlorophyll_n,
    count(m.nitrate_umol_kg) AS nitrate_n,
    count(m.ph_total) AS ph_n,
    count(m.backscatter_700) AS backscatter_n
FROM floats f
JOIN profiles p
    ON p.float_id = f.float_id
LEFT JOIN measurements m
    ON m.profile_id = p.profile_id
   AND m.qc_flag = 1
GROUP BY f.float_id, f.platform, f.project
"""

REGION_ROWS = """
SELECT
    p.region,
    count(DISTINCT p.float_id) AS floats,
    count(DISTINCT p.profile_id) AS profiles,
    min(p.obs_time)::date AS first_obs,
    max(p.obs_time)::date AS last_obs,
    round(
        avg(m.temperature_c) FILTER (WHERE m.pressure_dbar < 10)::numeric, 2
    ) AS mean_surface_temp,
    round(
        avg(m.salinity_psu) FILTER (WHERE m.pressure_dbar < 10)::numeric, 2
    ) AS mean_surface_salinity,
    count(m.oxygen_umol_kg) AS oxygen_n,
    count(m.chlorophyll_mg_m3) AS chlorophyll_n,
    count(m.nitrate_umol_kg) AS nitrate_n,
    count(m.ph_total) AS ph_n,
    count(m.backscatter_700) AS backscatter_n
FROM profiles p
LEFT JOIN measurements m
    ON m.profile_id = p.profile_id
   AND m.qc_flag = 1
WHERE p.region IS NOT NULL
GROUP BY p.region
"""


def summarise_float(row: dict) -> str:
    """One float, described the way somebody would ask about it."""
    float_id = row["float_id"]

    descriptors = []

    platform = _named(row.get("platform"))
    project = _named(row.get("project"))

    if platform:
        descriptors.append(f"{_article(platform)} {platform} platform")

    if project:
        descriptors.append(f"part of project {project}")

    opening = (
        f"ARGO float {float_id} is {' and '.join(descriptors)}."
        if descriptors
        else f"ARGO float {float_id}."
    )

    sentences = [opening, _coverage(row)]

    regions = _regions(row.get("regions"))

    if regions:
        sentences.append(f"It reported in the {regions}.")

    depth = _depth(row)

    if depth:
        sentences.append(depth)

    surface = _surface(row)

    if surface:
        sentences.append(surface)

    sensors = _sensors(row)

    if sensors:
        sentences.append(
            f"It is a BGC float and also measures {sensors}."
        )

    return " ".join(sentences)


def summarise_region(row: dict) -> str:
    """One region, described with the numbers a reader would ask to compare."""
    sentences = [
        f"The {row['region']} holds "
        f"{_count(row.get('floats'), 'float')} and "
        f"{_count(row.get('profiles'), 'profile')} in this database."
    ]

    span = _span(row)

    if span:
        sentences.append(f"Coverage runs {span}.")

    surface = _surface(row)

    if surface:
        sentences.append(surface)

    sensors = _sensors(row)

    if sensors:
        sentences.append(f"Some of its floats also measure {sensors}.")

    return " ".join(sentences)


def _coverage(row: dict) -> str:
    profiles = _count(row.get("profiles"), "profile")
    span = _span(row)

    if span:
        return f"It recorded {profiles} {span}."

    return f"It recorded {profiles}."


def _span(row: dict) -> str:
    first = row.get("first_obs")
    last = row.get("last_obs")

    if not first or not last:
        return ""

    if first == last:
        return f"on {first}"

    return f"from {first} to {last}"


def _regions(regions) -> str:
    names = sorted({str(r) for r in (regions or []) if r})

    if not names:
        return ""

    if len(names) == 1:
        return names[0]

    return f"{', '.join(names[:-1])} and {names[-1]}"


def _depth(row: dict) -> str:
    shallowest = row.get("min_pressure")
    deepest = row.get("max_pressure")

    if shallowest is None or deepest is None:
        return ""

    return (
        f"Its measurements span {shallowest} to {deepest} decibars "
        "of pressure, roughly that depth in metres."
    )


def _surface(row: dict) -> str:
    temperature = row.get("mean_surface_temp")
    salinity = row.get("mean_surface_salinity")

    stated = []

    if temperature is not None:
        stated.append(
            f"mean surface temperature of {temperature} degrees Celsius"
        )

    if salinity is not None:
        stated.append(f"mean surface salinity of {salinity} PSU")

    if not stated:
        return ""

    return f"Good-quality readings give a {' and a '.join(stated)}."


# The biogeochemical sensors, named the way somebody would ask for them.
_SENSORS = (
    ("oxygen_n", "dissolved oxygen"),
    ("chlorophyll_n", "chlorophyll"),
    ("nitrate_n", "nitrate"),
    ("ph_n", "pH"),
    ("backscatter_n", "particle backscatter"),
)


def _sensors(row: dict) -> str:
    """The BGC parameters this subject actually carries readings for.

    Most of the fleet is core-only, so saying which floats have a sensor is
    the difference between a BGC question finding data and averaging nothing.
    """
    carried = [
        label for key, label in _SENSORS if int(row.get(key) or 0) > 0
    ]

    if not carried:
        return ""

    if len(carried) == 1:
        return carried[0]

    return f"{', '.join(carried[:-1])} and {carried[-1]}"


def _article(word: str) -> str:
    """Platform names are read as words, so the first letter decides."""
    return "an" if word[:1].upper() in "AEIOU" else "a"


def _named(value) -> str:
    """A metadata string worth putting in a sentence, or empty.

    ARGO records unknown platforms and projects as the literal word, and the
    CSV loader writes it too. Repeating it in a summary would only teach the
    retriever that every float is "unknown".
    """
    text = (value or "").strip()

    return "" if text.lower() in ("", "unknown", "n/a", "none") else text


def _count(value, noun: str) -> str:
    number = int(value or 0)

    return f"{number:,} {noun}{'' if number == 1 else 's'}"


def collect() -> list[tuple[str, str, str]]:
    """Read the database and return ``(kind, id, text)`` for every subject."""
    from src.utils.db import fetch_all

    out = []

    columns, rows = fetch_all(FLOAT_ROWS, readonly=True, timeout_ms=120_000)

    for row in _dicts(columns, rows):
        out.append(
            ("float", str(row["float_id"]), summarise_float(row))
        )

    columns, rows = fetch_all(REGION_ROWS, readonly=True, timeout_ms=120_000)

    for row in _dicts(columns, rows):
        out.append(
            ("region", str(row["region"]), summarise_region(row))
        )

    return out


def _dicts(columns, rows) -> list[dict]:
    return [dict(zip(columns, row, strict=True)) for row in rows]
