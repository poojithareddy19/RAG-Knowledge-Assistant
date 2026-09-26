"""A filter the question never asked for, copied from the retrieved notes.

The SQL prompt carries the float and region summaries that retrieval found, as
background, under a rule that says in capitals not to turn them into WHERE
filters. The model does it anyway. Asked for profiles per year with a running
total, it added region = 'Bay of Bengal', platform = 'PROVOR_MT', project =
'Argo INDIA' and a 2003 date range, every one of them lifted from a summary.
Asked how many drifting buoys there were, it added buoy_type = 'SVPB'. The
query runs, returns a number, and answers a narrower question than the one
asked.

Like the other checks, this reports and never rewrites. A filter value is
suspect when it appears in the retrieved context and nowhere in the question:
a region, a platform, a project, a buoy type, or a date whose year the
question does not name. Values the question itself contains are always
allowed, and so are short codes such as data_mode 'D', which a question says
in words ("delayed mode") rather than letters. A question about a relative
period ("the last six months") is left alone for dates, because its bounds
come from the archive's coverage, not from the question's own words.
"""

from __future__ import annotations

import re

# A quoted literal compared in a condition: = 'x', <> 'x', LIKE 'x', >= 'x' ...
_COMPARED = re.compile(r"(?:=|<>|!=|<=?|>=?|\blike|\bilike)\s*'([^']+)'", re.I)
_IN_LIST = re.compile(r"\bin\s*\(([^)]*)\)", re.I)
_BETWEEN = re.compile(r"\bbetween\s*'([^']+)'\s*and\s*'([^']+)'", re.I)
_QUOTED = re.compile(r"'([^']+)'")

_DATE = re.compile(r"^((?:19|20)\d{2})-\d{2}-\d{2}")
_YEARS = re.compile(r"\b((?:19|20)\d{2})\b")
_RELATIVE = re.compile(
    r"\b(last|past|recent|recently|latest|since|ago|this year|so far|to date|current)\b",
    re.I,
)

# Shorter than this is a code the question states in words, not a copied value.
_MIN_LENGTH = 3


def _literals(sql: str) -> list[str]:
    found = [m.group(1) for m in _COMPARED.finditer(sql)]

    for m in _BETWEEN.finditer(sql):
        found.extend(m.groups())

    for m in _IN_LIST.finditer(sql):
        found.extend(_QUOTED.findall(m.group(1)))

    return found


def copied_filter(question: str, sql: str, context: str) -> str | None:
    """The first filter value copied from the context rather than the question."""
    if not question or not sql or not context:
        return None

    asked = question.lower()
    notes = context.lower()
    asked_years = {int(y) for y in _YEARS.findall(question)}
    relative = bool(_RELATIVE.search(question))

    for value in _literals(sql):
        text = value.strip()
        date = _DATE.match(text)

        if date:
            year = int(date.group(1))

            # A year the question names, or the bound just after it.
            if relative or year in asked_years or (year - 1) in asked_years:
                continue

            if date.group(1) in notes:
                return _reason(f"the date '{text}' (the year {year})")

            continue

        low = text.lower().strip("%")

        if len(low) < _MIN_LENGTH or low in asked:
            continue

        if low in notes:
            return _reason(f"'{text}'")

    return None


def _reason(what: str) -> str:
    return (
        f"the query filters on {what}, a value taken from the notes about what "
        "is in the database, which the question never mentions. The notes are "
        "background and never become filters: remove every condition the "
        "question did not ask for, and answer exactly the question as asked"
    )


class CopiedFilter(ValueError):
    """A filter lifted from the retrieved context. Repairable, like the other
    checks, so deliberately not an SQLRejected."""
