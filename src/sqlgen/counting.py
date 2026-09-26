"""A question that counts profiles, answered by counting depth levels.

A profile is one row in profiles and hundreds of rows in measurements. Asked
"number of profiles per region", "per year" and "per year with a running
total", the model joined measurements, filtered to good surface readings, and
counted with COUNT(*). That counts levels, not casts, and the filter drops any
profile without a good surface reading. All three were wrong in the benchmark,
and asked for the Arabian Sea in 2022 live, it did the same and was right only
because the answer was zero. The schema's QC convention, read as a rule for
every query, is what pulls the join in.

Like the period check, this reports and never rewrites; the pipeline hands the
reason to its one repair attempt. Two cases:

- the question counts profiles and names no measured quantity: nothing in
  measurements is needed, so the join itself is the mistake
- it counts profiles and does name one ("profiles with oxygen readings",
  "profiles deeper than 1000 decibars"): the join is right, but the count
  must be COUNT(DISTINCT profile_id) or it counts levels

It stays quiet when the question is not counting profiles, or the query does
not touch measurements.
"""

from __future__ import annotations

import re

_COUNTS_PROFILES = re.compile(
    r"\b(how many|number of|count of|count the|total)\b[^?]*\bprofiles?\b"
    r"|\bprofiles? (per|by|each|in each)\b",
    re.I,
)

# Anything that is only answerable from measurements.
_MEASURED = re.compile(
    r"\b(temperature|temp|salinity|pressure|depths?|deep|deeper|shallow|dbar|"
    r"decibars?|oxygen|chlorophyll|nitrate|ph|backscatter|bgc|biogeochemi\w*|"
    r"qc|quality|flags?|measurements?|levels?|readings?|sensors?|surface)\b",
    re.I,
)

_MEASUREMENTS = re.compile(r"\bmeasurements\b", re.I)
_DISTINCT_PROFILE = re.compile(r"count\s*\(\s*distinct\s+[\w.]*profile_id\s*\)", re.I)


def profiles_overcounted(question: str, sql: str) -> str | None:
    """Why this query counts levels where the question counts profiles, or None."""
    if not question or not sql:
        return None

    if not _COUNTS_PROFILES.search(question) or not _MEASUREMENTS.search(sql):
        return None

    if not _MEASURED.search(question):
        # The fix has to name the filters as well as the join. Told only to
        # drop the join, the repair kept "pressure_dbar < 10" and moved it onto
        # profiles, where the column does not exist, and the query failed.
        return (
            "the question counts profiles and asks about nothing measured, but "
            "the query joins measurements, so COUNT(*) counts depth levels "
            "rather than profiles and any filter on measurements drops "
            "profiles. Count from profiles alone, with no join to measurements: "
            "remove the join and every condition on measurement columns "
            "(pressure_dbar, qc_flag, temperature_c, salinity_psu and the other "
            "measured values). profiles has only profile_id, float_id, "
            "cycle_number, obs_time, latitude, longitude, region and data_mode"
        )

    if not _DISTINCT_PROFILE.search(sql):
        return (
            "the question counts profiles, and the query joins measurements, "
            "where one profile is many rows, so a plain COUNT counts depth "
            "levels. Count COUNT(DISTINCT p.profile_id) instead"
        )

    return None


# "how many measurements", "number of BGC measurements", "measurements per".
_COUNTS_MEASUREMENTS = re.compile(
    r"\b(how many|number of|count of|count the|total)\s+(?:[\w-]+\s+){0,2}measurements\b"
    r"|\bmeasurements\s+(per|by|each|in each)\b",
    re.I,
)


def measurements_undercounted(question: str, sql: str) -> str | None:
    """Why this query counts something other than measurements, or None.

    The reverse of the profile case. Asked "how many measurements belong to
    floats in the Southern Indian Ocean?", the model counted rows of profiles
    joined to floats and never read measurements at all.
    """
    if not question or not sql:
        return None

    if not _COUNTS_MEASUREMENTS.search(question) or _MEASUREMENTS.search(sql):
        return None

    return (
        "the question counts measurements, but the query never reads the "
        "measurements table, so it counts profiles or floats instead. Count "
        "rows of measurements, joining profiles on profile_id for region or "
        "time and floats on float_id for float details"
    )


def count_problem(question: str, sql: str) -> str | None:
    """The first way this query counts the wrong thing, or None."""
    return profiles_overcounted(question, sql) or measurements_undercounted(question, sql)
