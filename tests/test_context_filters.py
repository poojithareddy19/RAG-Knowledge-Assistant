"""The copied-filter check.

It exists because the model turns values from the retrieved summaries into
WHERE filters the question never asked for. What is pinned down: the real
cases are caught, values the question itself names are always allowed, short
codes and relative periods are left alone, and no reference query is flagged
against its own context.
"""

import pytest

from src.sqlgen.context_filters import copied_filter

# Retrieved context shaped like the real summaries.
CONTEXT = (
    "- ARGO float 1902367 is a PROVOR_MT platform and part of project Argo INDIA. "
    "It recorded 59 profiles from 2003-02-11 to 2004-12-30. It reported in the Bay of Bengal.\n"
    "- ARGO float 1900042 is an APEX platform and part of project US ARGO PROJECT.\n"
    "- The Global Drifter Program buoys in this database are of type SVPB and reported "
    "from 2023-01-01 to 2024-01-01."
)


@pytest.mark.parametrize(
    "question, sql, value",
    [
        # Seen live on "profiles per year with a running total".
        ("Profiles per year with a running cumulative total",
         "SELECT date_trunc('year', obs_time), count(*) FROM profiles p "
         "WHERE p.region = 'Bay of Bengal' GROUP BY 1",
         "Bay of Bengal"),
        ("Profiles per year with a running cumulative total",
         "SELECT 1 FROM profiles p JOIN floats f ON f.float_id = p.float_id "
         "WHERE f.platform = 'PROVOR_MT'",
         "PROVOR_MT"),
        # Seen in the benchmark.
        ("How many measurements belong to floats in the Southern Indian Ocean?",
         "SELECT COUNT(*) FROM profiles p JOIN floats f ON f.float_id = p.float_id "
         "WHERE f.project = 'US ARGO PROJECT' AND p.region = 'Southern Indian Ocean'",
         "US ARGO PROJECT"),
        # Recorded in the README.
        ("How many drifting buoys are in the database?",
         "SELECT count(*) FROM drifters WHERE buoy_type = 'SVPB'",
         "SVPB"),
        ("Number of drifter observations per month",
         "SELECT date_trunc('month', obs_time), count(*) FROM drifter_observations "
         "WHERE obs_time >= '2023-01-01' AND obs_time < '2024-01-01' GROUP BY 1",
         "2023"),
        ("Which floats are in the project?",
         "SELECT float_id FROM floats WHERE project IN ('Argo INDIA', 'X')",
         "Argo INDIA"),
    ],
)
def test_a_value_copied_from_the_context_is_caught(question, sql, value):
    reason = copied_filter(question, sql, CONTEXT)

    assert reason is not None
    assert value in reason


@pytest.mark.parametrize(
    "question, sql",
    [
        # The question names the value itself.
        ("How many profiles were recorded in the Bay of Bengal?",
         "SELECT count(*) FROM profiles WHERE region = 'Bay of Bengal'"),
        ("How many buoys are of type SVPB?",
         "SELECT count(*) FROM drifters WHERE buoy_type = 'SVPB'"),
        # A year the question names, and the bound after it.
        ("How many drifter observations were there in 2023?",
         "SELECT count(*) FROM drifter_observations "
         "WHERE obs_time >= '2023-01-01' AND obs_time < '2024-01-01'"),
        # A short code the question says in words.
        ("How many profiles are in delayed mode?",
         "SELECT count(*) FROM profiles WHERE data_mode = 'D'"),
        # A relative period: its bounds come from the archive's coverage.
        ("How many drifter observations in the last six months?",
         "SELECT count(*) FROM drifter_observations WHERE obs_time >= '2023-07-01'"),
        # A literal that is not in the context at all.
        ("How many floats use an ARVOR platform?",
         "SELECT count(*) FROM floats WHERE platform = 'ARVOR'"),
        # No filter values.
        ("How many profiles are there?", "SELECT count(*) FROM profiles"),
    ],
)
def test_values_the_question_supports_pass(question, sql):
    assert copied_filter(question, sql, CONTEXT) is None


def test_no_context_means_nothing_to_copy_from():
    sql = "SELECT count(*) FROM drifters WHERE buoy_type = 'SVPB'"

    assert copied_filter("How many buoys?", sql, "") is None


def test_date_trunc_units_are_not_filters():
    """'year' and 'month' are arguments, not compared values."""
    context = "- a summary that mentions every month and year"
    sql = "SELECT date_trunc('year', obs_time), count(*) FROM profiles GROUP BY 1"

    assert copied_filter("Number of profiles per year", sql, context) is None
