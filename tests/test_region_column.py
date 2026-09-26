"""The region-column check.

It exists because the model wrote f.platform = 'Southern Indian Ocean' and
answered zero three times running. What is pinned down: a region name on any
column but region is caught, however it is compared, and a region name on a
region column, or a non-region value anywhere, passes.
"""

import pytest

from src.sqlgen.checks import mismatches
from src.sqlgen.region_column import region_misplaced

QUESTION = "How many measurements belong to floats in the Southern Indian Ocean?"
# The model's SQL in all three live runs, verbatim.
LIVE_SQL = (
    "SELECT COUNT(*) FROM measurements m JOIN profiles p ON p.profile_id = m.profile_id "
    "JOIN floats f ON f.float_id = p.float_id "
    "WHERE f.platform = 'Southern Indian Ocean' AND m.qc_flag = 1"
)


def test_the_live_query_is_caught():
    reason = region_misplaced(QUESTION, LIVE_SQL)

    assert reason is not None
    assert "f.platform" in reason
    assert "p.region = 'Southern Indian Ocean'" in reason


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1 FROM floats f WHERE f.project = 'Arabian Sea'",
        "SELECT 1 FROM drifters d WHERE d.buoy_type = 'Bay of Bengal'",
        "SELECT 1 FROM floats WHERE platform LIKE '%Arabian Sea%'",
        "SELECT 1 FROM floats WHERE platform IN ('Arabian Sea', 'APEX')",
        "SELECT 1 FROM floats f WHERE f.platform ILIKE 'bay of bengal'",
    ],
)
def test_a_region_name_on_any_other_column_is_caught(sql):
    assert region_misplaced("", sql) is not None


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) FROM profiles WHERE region = 'Arabian Sea'",
        "SELECT count(*) FROM profiles p WHERE p.region = 'Southern Indian Ocean'",
        "SELECT count(*) FROM drifter_observations o WHERE o.region IN ('Arabian Sea', 'Bay of Bengal')",
        'SELECT count(*) FROM profiles p WHERE p."region" = \'Bay of Bengal\'',
        # Not a region name at all.
        "SELECT count(*) FROM floats WHERE platform = 'APEX'",
        "SELECT count(*) FROM profiles WHERE data_mode = 'D'",
    ],
)
def test_region_names_on_region_columns_and_other_values_pass(sql):
    assert region_misplaced("", sql) is None


def test_it_runs_with_the_other_checks():
    assert any("f.platform" in reason for reason in mismatches(QUESTION, LIVE_SQL))


def test_no_reference_query_is_flagged():
    import csv

    with open("data/evaluation/ocean_questions.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["expected_sql"].strip()]

    for row in rows:
        assert region_misplaced(row["question"], row["expected_sql"]) is None, row["question"]
