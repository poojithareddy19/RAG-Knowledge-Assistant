"""The one-year period check.

It exists because of one real answer: "and in 2022?" about the Arabian Sea was
answered 63 from obs_time >= '2022-01-01', when 2022 has no profiles there at
all. What is pinned down is that it catches that shape and stays quiet about
everything it cannot be sure of.
"""

import pytest

from src.sqlgen.period import open_ended_year

REAL_QUESTION = "How many profiles are in the Arabian Sea and in 2022?"
REAL_SQL = (
    "SELECT COUNT(*) FROM profiles WHERE region = 'Arabian Sea' "
    "AND obs_time >= '2022-01-01' LIMIT 500"
)


def test_the_real_follow_up_is_caught():
    reason = open_ended_year(REAL_QUESTION, REAL_SQL)

    assert reason is not None
    assert "2022" in reason
    assert "obs_time < '2023-01-01'" in reason


@pytest.mark.parametrize(
    "sql",
    [
        # Bounded at both ends, the form the prompt asks for.
        "SELECT count(*) FROM profiles WHERE obs_time >= '2022-01-01' "
        "AND obs_time < '2023-01-01'",
        "SELECT count(*) FROM profiles WHERE obs_time >= '2022-01-01' "
        "AND obs_time <= '2022-12-31'",
        "SELECT count(*) FROM profiles WHERE obs_time BETWEEN '2022-01-01' AND '2022-12-31'",
        # A year taken apart rather than compared has no open end.
        "SELECT count(*) FROM profiles WHERE EXTRACT(year FROM obs_time) = 2022",
        "SELECT count(*) FROM profiles WHERE date_trunc('year', obs_time) = '2022-01-01'",
        # No bound on that year at all is a different problem, not this one.
        "SELECT count(*) FROM profiles",
    ],
)
def test_queries_that_do_not_run_past_the_year_pass(sql):
    assert open_ended_year("How many profiles were recorded in 2022?", sql) is None


@pytest.mark.parametrize(
    "question",
    [
        "How many profiles have been recorded since 2022?",
        "How many profiles were recorded after 2022?",
        "How many profiles from 2022 onwards?",
        "How many profiles between 2010 and 2015?",
        "How many profiles were recorded 2010-2015?",
        "How many profiles are there in total?",
    ],
)
def test_a_question_that_opens_the_range_is_left_alone(question):
    assert open_ended_year(question, "SELECT 1 FROM profiles WHERE obs_time >= '2022-01-01'") is None


def test_a_depth_or_a_float_id_is_not_a_year():
    """1000 decibars and a WMO number are not years, and 2900 is out of range."""
    sql = "SELECT 1 FROM profiles WHERE obs_time >= '2022-01-01'"

    assert open_ended_year("Mean temperature below 1000 decibars for float 2900107", sql) is None


def test_empty_input_is_not_a_finding():
    assert open_ended_year("", "SELECT 1") is None
    assert open_ended_year("in 2022", "") is None


def test_the_gold_reference_queries_are_never_flagged():
    """The check must not disagree with a single hand-written answer."""
    import csv

    with open("data/evaluation/ocean_questions.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["expected_sql"].strip()]

    assert rows
    for row in rows:
        assert open_ended_year(row["question"], row["expected_sql"]) is None, row["question"]
