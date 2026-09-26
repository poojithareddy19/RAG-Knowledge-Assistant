"""A region name compared with a column that does not hold regions.

Asked how many measurements belong to floats in the Southern Indian Ocean, the
model wrote f.platform = 'Southern Indian Ocean': the right value on the wrong
column. floats.platform holds instrument types such as APEX and PROVOR_MT, so
the query ran, matched nothing, and answered zero in every one of three runs.

The rule is exact, so the check can be too. The schema catalog says region is
"exactly one of" three names, and the only columns that hold them are
profiles.region and drifter_observations.region. A comparison with one of
those names against any other column is a mistake, whatever the question.

It reports and never rewrites; the reason goes to the repair with every other
check's through checks.QueryMismatch.
"""

from __future__ import annotations

import re

# The three values of profiles.region and drifter_observations.region, as the
# schema catalog lists them. load_regions.py loads exactly these polygons.
REGION_NAMES = ("Arabian Sea", "Bay of Bengal", "Southern Indian Ocean")

_LOWER = {name.lower(): name for name in REGION_NAMES}

# column = 'x', column <> 'x', column LIKE 'x', column ILIKE 'x'
_COMPARED = re.compile(
    r'([\w."]+)\s*(?:=|<>|!=|\blike\b|\bilike\b)\s*\'([^\']+)\'',
    re.I,
)
# column IN ('x', 'y')
_IN_LIST = re.compile(r'([\w."]+)\s+in\s*\(([^)]*)\)', re.I)
_QUOTED = re.compile(r"'([^']+)'")


def _comparisons(sql: str) -> list[tuple[str, str]]:
    found = _COMPARED.findall(sql)

    for column, values in _IN_LIST.findall(sql):
        found.extend((column, value) for value in _QUOTED.findall(values))

    return found


def region_misplaced(question: str, sql: str) -> str | None:
    """Why this query compares a region name with a non-region column, or None."""
    if not sql:
        return None

    for column, value in _comparisons(sql):
        name = _LOWER.get(value.strip().strip("%").strip().lower())

        if name is None:
            continue

        if column.split(".")[-1].strip('"').lower() == "region":
            continue

        return (
            f"'{value}' is a region, but the query compares it with {column}, "
            "which holds something else, so it matches nothing. Region names "
            "are only in the region column: profiles.region for floats, "
            "drifter_observations.region for buoys. Filter on "
            f"p.region = '{name}', joining profiles if the query does not "
            "already read it"
        )

    return None
