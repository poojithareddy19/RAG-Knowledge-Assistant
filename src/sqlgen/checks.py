"""Does the query answer the question that was asked?

The validator decides whether a query is safe. These checks decide whether a
safe query answers the question: the right period, the right thing counted, no
filter the question never asked for. Each one exists because the model got
exactly that wrong on real questions, despite a prompt rule saying not to.

They all run, and every problem found goes to the repair in one message. They
used to raise one at a time, and a query with two problems ("measurements in
the Southern Indian Ocean" copied a project filter and counted profiles) had
its first problem repaired, then failed the second check with no repair left.

Raised before the database is touched, and not as an SQLRejected, which is
never repaired: every one of these is a mistake a repair given the reason can
fix.
"""

from __future__ import annotations

from src.sqlgen.context_filters import copied_filter
from src.sqlgen.counting import count_problem
from src.sqlgen.period import period_problem


def mismatches(question: str, sql: str, context: str = "") -> list[str]:
    """Every way this query fails to answer the question as asked."""
    found = [
        period_problem(question, sql),
        count_problem(question, sql),
        copied_filter(question, sql, context),
    ]

    return [reason for reason in found if reason]


class QueryMismatch(ValueError):
    """A safe query that answers a different question, with every reason."""

    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__(
            "the query does not answer the question as asked:\n- "
            + "\n- ".join(self.reasons)
        )


def check(question: str, sql: str, context: str = "") -> None:
    """Raise QueryMismatch if the query does not answer the question."""
    reasons = mismatches(question, sql, context)

    if reasons:
        raise QueryMismatch(reasons)
