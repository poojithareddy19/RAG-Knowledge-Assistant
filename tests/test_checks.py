"""All the answer checks together.

What is pinned down: every problem in a query is reported at once, so the one
repair attempt can fix them all. They used to raise one at a time, and the
real "measurements in the Southern Indian Ocean" query, which both copied a
filter and counted the wrong table, had one problem repaired and was then
refused for the other.
"""

import pytest

from src.sqlgen.checks import QueryMismatch, check, mismatches

QUESTION = "How many measurements belong to floats in the Southern Indian Ocean?"
CONTEXT = "- ARGO float 1900042 is an APEX platform and part of project US ARGO PROJECT."
# The model's SQL in the benchmark run, verbatim.
SQL = (
    "SELECT COUNT(*) FROM profiles p JOIN floats f ON f.float_id = p.float_id "
    "WHERE f.project = 'US ARGO PROJECT' AND p.region = 'Southern Indian Ocean'"
)


def test_every_problem_in_the_real_query_is_found():
    reasons = mismatches(QUESTION, SQL, CONTEXT)

    assert len(reasons) == 2
    assert any("never reads the measurements table" in r for r in reasons)
    assert any("US ARGO PROJECT" in r for r in reasons)


def test_one_exception_carries_them_all():
    with pytest.raises(QueryMismatch) as raised:
        check(QUESTION, SQL, CONTEXT)

    assert len(raised.value.reasons) == 2
    message = str(raised.value)
    assert "never reads the measurements table" in message
    assert "US ARGO PROJECT" in message


def test_a_query_that_answers_the_question_passes_quietly():
    sql = (
        "SELECT count(*) FROM measurements m JOIN profiles p ON p.profile_id = m.profile_id "
        "WHERE p.region = 'Southern Indian Ocean'"
    )

    assert mismatches(QUESTION, sql, CONTEXT) == []
    check(QUESTION, sql, CONTEXT)


def test_it_is_repairable_not_a_refusal():
    """An SQLRejected is never repaired; this has to reach the repair."""
    from src.sqlgen.validator import SQLRejected

    assert not issubclass(QueryMismatch, SQLRejected)
