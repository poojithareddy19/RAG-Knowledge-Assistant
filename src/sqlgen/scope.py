"""Refuse questions asking for a quantity this database does not hold.

The drifting buoys widened the hallucination surface rather than narrowing it.
Before they were loaded, "what is the current speed at 1000 decibars" was
refused because nothing in the schema resembled a current. Afterwards the
buoys carry `eastward_velocity_m_s` and `northward_velocity_m_s`, something
called current exists, and the model answers. It is still wrong: a drifter
rides the surface and measures nothing below it.

Prompt wording did not close this. The catalog says "this is the only current
speed in the database, and it is at the surface only" in the sentence directly
under the column, and says of `pressure_dbar` that it is "not the depth of the
seabed, which this database does not record". Both were read and both were
overridden, the second by aliasing `max(pressure_dbar)` to `seafloor_depth`.

So the rule moves out of the prose the model may ignore and into code that
runs before the model is called at all. Two kinds of question are refused:

- a quantity no column holds: wind, rainfall, the depth of the seabed, waves,
  marine life, pollution
- a quantity held only at the surface, asked for at depth: the buoy current
  velocities, which exist for the sea surface and for nothing deeper

Both are read off the schema rather than off the evaluation set. The catalog
holds pressure, temperature, salinity, oxygen, chlorophyll, nitrate, pH,
backscatter, position, time, and a surface current. Everything refused here is
absent from that list, and the depth rule is the one column whose coverage is
narrower than the question a user would ask of it.

The honest limit: this is a lexical gate, so a paraphrase it does not carry
passes straight through to the model, exactly as before. It can only add a
refusal, never remove one, so a question it says nothing about is generated
and validated on the path it always used.
"""

from __future__ import annotations

import re

# Quantities with no column anywhere in the schema. Each is something a user
# plausibly asks an ocean assistant, which is why the model reaches for the
# nearest column instead of declining.
_UNHELD = (
    (
        re.compile(r"\bwind\b|\bwinds\b|\bgust", re.I),
        "wind is not measured: this database holds in-situ ocean "
        "observations, not atmospheric ones",
    ),
    (
        re.compile(r"\brain\b|\brainfall\b|\bprecipitation\b", re.I),
        "rainfall is not measured: this database holds in-situ ocean "
        "observations, not atmospheric ones",
    ),
    (
        re.compile(
            r"\bseafloor\b|\bsea floor\b|\bseabed\b|\bsea bed\b"
            r"|\bbathymetr"
            # The seabed asked for without being named: the depth belongs to
            # the ocean rather than to the float, which is the distinction
            # the model collapses when it aliases max(pressure_dbar).
            r"|\b(ocean|sea|water)\s+(floor|bed|depth)\b"
            r"|\bdepth\s+of\s+the\s+(ocean|sea|water)\b"
            r"|\bhow\s+deep\s+is\s+the\s+(ocean|sea|water)\b",
            re.I,
        ),
        "the depth of the seabed is not recorded: pressure_dbar is how deep "
        "the float descended, not how deep the ocean is beneath it",
    ),
    (
        re.compile(
            r"\bwave height\b|\bwave heights\b|\bswell\b"
            r"|\bsignificant wave\b|\btide\b|\btidal\b",
            re.I,
        ),
        "waves and tides are not measured: a float reports a profile, not a "
        "sea state",
    ),
    (
        re.compile(
            r"\bwhale|\bdolphin|\bfish\b|\bplankton\b|\bcoral\b|\bturtle",
            re.I,
        ),
        "marine life is not observed: these are physical and biogeochemical "
        "sensors, not biological surveys",
    ),
    (
        re.compile(r"\bplastic\b|\bmicroplastic|\bpollution\b|\bdebris\b", re.I),
        "pollution is not measured: no sensor in this database detects it",
    ),
)

# "current" is refused only in its ocean sense. On its own it is one of the
# commonest words in English and belongs to "the current year" and "currently
# reporting", neither of which is a question about water movement, so the
# plural or an adjacent speed word is required before it counts.
_CURRENT = re.compile(
    r"\bcurrents\b"
    r"|\bcurrent\s+(speed|velocity|direction|magnitude)\b"
    r"|\b(speed|velocity|direction)\s+of\s+(the\s+)?current\b",
    re.I,
)

# A depth named anywhere below the surface. A bare number with a pressure or
# distance unit counts, as do the words used instead of one.
_BELOW_SURFACE = re.compile(
    r"\b\d+(\.\d+)?\s*(dbar|decibars?|db|metres?|meters?|m)\b"
    r"|\bat depth\b|\bdeep\b|\bdeeper\b|\bdepths?\b"
    r"|\bsubsurface\b|\bmid-?water\b|\bthermocline\b|\bsea ?bed\b",
    re.I,
)

# Held, but for the sea surface and nothing below it.
_SURFACE_ONLY = (
    (
        _CURRENT,
        "the only current in this database is the surface current the "
        "drifting buoys derive from their own movement, so there is no "
        "current at depth to report",
    ),
)


def out_of_scope(question: str) -> str | None:
    """Return why the question cannot be answered, or None to carry on.

    A returned string is shown to the user as the reason for the refusal, so
    it names what is missing rather than saying the question failed.
    """

    if not question or not question.strip():
        return None

    for pattern, reason in _UNHELD:
        if pattern.search(question):
            return reason

    # A surface-only quantity is in scope until the question asks for it
    # somewhere the sensor never was. "Average surface current speed" is a
    # real query against real columns; the same question at 1000 decibars is
    # not, and the difference is the depth rather than the quantity.
    for pattern, reason in _SURFACE_ONLY:
        if pattern.search(question) and _BELOW_SURFACE.search(question):
            return reason

    return None


# ---------------------------------------------------------------------------
# When a failure is the answer
# ---------------------------------------------------------------------------

_MISSING_COLUMN = re.compile(
    r'column\s+"?(?:([a-z_][a-z0-9_]*)\.)?([a-z_][a-z0-9_]*)"?\s+does not exist',
    re.I,
)

_TABLE_REF = re.compile(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", re.I)


def error_is_the_answer(sql, error, column_catalog, platforms):
    """Whether a failed query failed because the data is not there.

    A repair attempt is given the error and told to fix it, which is right when
    the error is a mistake and catastrophic when it is a fact. Asked how deep
    each drifting buoy dived, the model wrote ``MIN(o.pressure_dbar)`` on
    ``drifter_observations``, which failed because a buoy has no depth. The
    repair then substituted sea surface temperature and kept the aliases:

        MIN(o.sst_c) AS min_pressure,
        MAX(o.sst_c) AS max_pressure

    Three rows of temperatures labelled as pressures, returned as a confident
    answer. The first query was right to fail.

    So: if the missing column exists nowhere on the platform the query reads,
    the schema has answered the question and there is nothing to repair. If it
    exists on that platform and the query reached it through the wrong alias,
    that is a mistake and the repair is worth a try. `p.pressure_dbar` on the
    Argo tables is the second kind; `o.pressure_dbar` on the buoys is the first.

    Returns a reason to refuse, or None to let the repair proceed.
    """

    match = _MISSING_COLUMN.search(str(error))

    if not match or not column_catalog:
        return None

    column = match.group(2)

    tables = {t.lower() for t in _TABLE_REF.findall(sql or "")}

    # Which platforms this query actually reads. A query touching neither is
    # not something this rule can judge.
    families = {
        family
        for family, members in platforms.items()
        if tables & {m.lower() for m in members}
    }

    if not families:
        return None

    reachable = {
        col.lower()
        for family in families
        for table in platforms[family]
        for col in column_catalog.get(table, ())
    }

    if column.lower() in reachable:
        return None

    return (
        f"there is no {column} on "
        + " or ".join(sorted(families))
        + " data, so that quantity is not recorded for the platform this "
        "question is about"
    )
