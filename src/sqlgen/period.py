"""A question about one year, answered with a query that never stops.

The SQL prompt says it in as many words: "in 2020" means obs_time >= '2020-01-01'
AND obs_time < '2021-01-01', never an open-ended >= alone. The model broke it
anyway. Asked the follow-up "and in 2022?" about the Arabian Sea, it wrote
obs_time >= '2022-01-01' with no upper bound and answered 63, the profiles from
2023 onwards, when 2022 itself has none. A wrong number with a plausible query
under it is the failure this project exists to prevent, and a rule the model
reads and overrides has to be code, the same lesson as the scope gate.

So this is a check, not a rewrite. It reports the problem and the pipeline
hands that to the one repair attempt it already makes; the query is never
edited behind the model's back. It is deliberately narrow: it speaks only when
the question names exactly one year with no word that opens the range ("since
2022", "from 2020 to 2023"), and the query has a lower bound on that year and
no upper bound on any date. Anything it is unsure of, it leaves alone.

The second check is the other end of the same rule. "Between 2010 and 2015"
names two years, and a named year is a whole year, as a named month is a whole
month in the schema catalog, so the range runs to the end of 2015. The model
wrote obs_time < '2015-01-01', which drops 2015 entirely, and missed that gold
question in every recorded run. The reference answer has always been
< '2016-01-01'. The question is left as it is: changing it to fit the model
would be tuning the benchmark.
"""

from __future__ import annotations

import re

_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Words that make a year one end of a range rather than the whole period.
_RANGE = re.compile(
    r"\b(since|after|before|from|until|till|between|through|onwards?|"
    r"prior|later|earlier|up to|to date|so far)\b"
    r"|\d{4}\s*(?:-|to|and)\s*\d{4}",
    re.I,
)

# An upper bound on any date: < or <= followed by a quoted date, or BETWEEN.
_UPPER = re.compile(r"<\s*=?\s*(?:date\s*|timestamp\s*)?'\d{4}-|\bbetween\b", re.I)


def open_ended_year(question: str, sql: str) -> str | None:
    """Why this query answers more than the one year asked about, or None."""
    if not question or not sql:
        return None

    years = set(_YEAR.findall(question))

    if len(years) != 1 or _RANGE.search(question):
        return None

    year = int(years.pop())

    lower = re.compile(
        rf">\s*=?\s*(?:date\s*|timestamp\s*)?'{year}-", re.I
    ).search(sql)

    if not lower or _UPPER.search(sql):
        return None

    return (
        f"the question asks about {year} only, but the query has a lower bound "
        f"on {year} and no upper bound, so it also counts every year after it. "
        f"Bound it at both ends: obs_time >= '{year}-01-01' AND "
        f"obs_time < '{year + 1}-01-01'"
    )


# "between 2010 and 2015", "from 2010 to 2015", "2010-2015": two named years.
_YEAR_RANGE = re.compile(
    r"\b(?:between|from)\s+((?:19|20)\d{2})\s+(?:and|to|through|until|till)\s+"
    r"((?:19|20)\d{2})\b"
    r"|\b((?:19|20)\d{2})\s*-\s*((?:19|20)\d{2})\b",
    re.I,
)


def range_end_excluded(question: str, sql: str) -> str | None:
    """Why this query stops at the start of the last year named, or None.

    Speaks only when the query bounds the range at the first instant of the
    end year: obs_time < 'END-01-01', or <= it, or BETWEEN ... AND 'END-01-01'.
    Every other form, including the correct < 'END+1-01-01', is left alone.
    """
    if not question or not sql:
        return None

    match = _YEAR_RANGE.search(question)

    if not match:
        return None

    start, end = [int(y) for y in match.groups() if y]

    if end <= start:
        return None

    cut = rf"'{end}-01-01(?:[ t]00:00(?::00)?)?'"
    excluded = re.compile(
        rf"<\s*=?\s*(?:date\s*|timestamp\s*)?{cut}|\bbetween\s+'[^']*'\s+and\s+{cut}",
        re.I,
    )

    if not excluded.search(sql):
        return None

    return (
        f"the question asks for {start} to {end}, and a named year is the whole "
        f"year, so {end} is included, but the query stops at the start of {end} "
        f"and leaves it out. Bound it as obs_time >= '{start}-01-01' AND "
        f"obs_time < '{end + 1}-01-01'"
    )


def period_problem(question: str, sql: str) -> str | None:
    """The first period mismatch between the question and the query, or None."""
    return open_ended_year(question, sql) or range_end_excluded(question, sql)
