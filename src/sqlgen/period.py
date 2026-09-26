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


class OpenEndedPeriod(ValueError):
    """A query that runs past the one year the question asked about.

    Not an SQLRejected on purpose: a validator refusal is never repaired, and
    this is exactly the kind of mistake a repair given the reason can fix.
    """
