"""The profile-count check.

It exists because the model answered "number of profiles per region", "per
year" and "per year with a running total" by joining measurements and counting
rows. What is pinned down is that it catches those, asks for DISTINCT when the
join is genuinely needed, and says nothing about queries it has no reason to
doubt.
"""

import pytest

from src.sqlgen.counting import count_problem, measurements_undercounted, profiles_overcounted

# The model's SQL for the per-region question in the benchmark, verbatim.
PER_REGION = (
    "SELECT p.region, COUNT(*) AS profile_count FROM profiles p "
    "JOIN measurements m ON p.profile_id = m.profile_id "
    "WHERE m.pressure_dbar < 10 AND m.qc_flag = 1 AND m.temperature_c IS NOT NULL "
    "GROUP BY p.region ORDER BY p.region"
)


@pytest.mark.parametrize(
    "question",
    [
        "Number of profiles per region",
        "Number of profiles per year",
        "Profiles per year with a running cumulative total",
        "How many profiles are in the Arabian Sea and in 2022?",
    ],
)
def test_a_plain_profile_count_through_measurements_is_caught(question):
    reason = profiles_overcounted(question, PER_REGION)

    assert reason is not None
    assert "no join to measurements" in reason


def test_a_measured_condition_needs_distinct_not_no_join():
    question = "How many profiles have oxygen readings?"
    plain = (
        "SELECT COUNT(*) FROM profiles p JOIN measurements m "
        "ON m.profile_id = p.profile_id WHERE m.oxygen_umol_kg IS NOT NULL"
    )

    reason = profiles_overcounted(question, plain)

    assert reason is not None
    assert "COUNT(DISTINCT p.profile_id)" in reason


@pytest.mark.parametrize(
    "question, sql",
    [
        # Counted from profiles alone: the form the reference answers use.
        ("Number of profiles per region",
         "SELECT region, count(*) FROM profiles GROUP BY region ORDER BY region"),
        # A measured condition, counted properly.
        ("How many profiles have oxygen readings?",
         "SELECT COUNT(DISTINCT p.profile_id) FROM profiles p JOIN measurements m "
         "ON m.profile_id = p.profile_id WHERE m.oxygen_umol_kg IS NOT NULL"),
        ("How many profiles reach deeper than 1000 decibars?",
         "SELECT count(DISTINCT m.profile_id) FROM measurements m WHERE m.pressure_dbar > 1000"),
        # Not a profile count at all: measurement rows are the right thing to count.
        ("How many measurements are deeper than 1000 decibars?",
         "SELECT COUNT(*) FROM measurements WHERE pressure_dbar > 1000"),
        ("What is the average surface temperature per year?",
         "SELECT date_trunc('year', p.obs_time), avg(m.temperature_c) FROM profiles p "
         "JOIN measurements m ON m.profile_id = p.profile_id GROUP BY 1 ORDER BY 1"),
    ],
)
def test_queries_with_no_reason_to_doubt_pass(question, sql):
    assert profiles_overcounted(question, sql) is None


def test_empty_input_is_not_a_finding():
    assert profiles_overcounted("", "SELECT 1") is None
    assert profiles_overcounted("Number of profiles per region", "") is None


def test_the_gold_reference_queries_are_never_flagged():
    import csv

    with open("data/evaluation/ocean_questions.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["expected_sql"].strip()]

    assert rows
    for row in rows:
        assert profiles_overcounted(row["question"], row["expected_sql"]) is None, row["question"]


# ---------------------------------------------------------------------------
# Measurements counted as something else
# ---------------------------------------------------------------------------

# The model's SQL for this gold question in the live runs, after the copied
# project filter was repaired away.
SIO_QUESTION = "How many measurements belong to floats in the Southern Indian Ocean?"
SIO_SQL = (
    "SELECT COUNT(*) FROM profiles p JOIN floats f ON f.float_id = p.float_id "
    "WHERE p.region = 'Southern Indian Ocean'"
)


def test_a_measurement_count_that_never_reads_measurements_is_caught():
    reason = measurements_undercounted(SIO_QUESTION, SIO_SQL)

    assert reason is not None
    assert "never reads the measurements table" in reason
    assert count_problem(SIO_QUESTION, SIO_SQL) == reason


@pytest.mark.parametrize(
    "question, sql",
    [
        # The reference form.
        (SIO_QUESTION,
         "SELECT count(*) FROM measurements m JOIN profiles p ON p.profile_id = m.profile_id "
         "WHERE p.region = 'Southern Indian Ocean'"),
        ("How many measurements have been recorded?", "SELECT COUNT(*) FROM measurements"),
        ("How many BGC measurements are there?",
         "SELECT count(*) FROM measurements WHERE oxygen_umol_kg IS NOT NULL"),
        # Counting profiles is the other check's business, not this one's.
        ("How many profiles are there in total?", "SELECT count(*) FROM profiles"),
        # A question that mentions measurements without counting them.
        ("Which floats have measurements below 1000 decibars?",
         "SELECT DISTINCT float_id FROM profiles"),
    ],
)
def test_counts_that_read_measurements_or_are_not_measurement_counts_pass(question, sql):
    assert measurements_undercounted(question, sql) is None


def test_no_reference_query_trips_either_count_check():
    import csv

    with open("data/evaluation/ocean_questions.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["expected_sql"].strip()]

    for row in rows:
        assert count_problem(row["question"], row["expected_sql"]) is None, row["question"]
