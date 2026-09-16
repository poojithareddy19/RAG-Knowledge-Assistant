"""Result-set comparison.

Execution accuracy is only as trustworthy as this comparison, so the awkward
cases are pinned down here: float noise, row order, shape, and NULLs.
"""

from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytest

from src.evaluation.sql_metrics import results_match

GOLD_SET = "data/evaluation/ocean_questions.csv"


def test_identical_results_match():
    assert results_match([[1, 2.0]], [[1, 2.0]])


def test_row_order_does_not_matter():
    # No question specifies an order, and the reference query's ORDER BY is a
    # presentation choice rather than part of the answer.
    assert results_match(
        [["a", 1], ["b", 2]],
        [["b", 2], ["a", 1]],
    )


def test_float_noise_below_tolerance_still_matches():
    assert results_match([[26.1234567891]], [[26.1234567892]])


def test_a_real_numeric_difference_does_not_match():
    assert not results_match([[26.12]], [[26.13]])


def test_decimal_and_float_of_the_same_value_match():
    # Postgres returns numeric as Decimal and double precision as float.
    assert results_match([[Decimal("26.5")]], [[26.5]])


def test_an_extra_column_is_not_the_same_answer():
    assert not results_match([[1, 2]], [[1]])


def test_a_different_row_count_does_not_match():
    assert not results_match([[1], [2]], [[1]])


def test_nulls_compare_equal_to_nulls():
    assert results_match([[None, 1]], [[None, 1]])


def test_null_does_not_match_zero():
    assert not results_match([[None]], [[0]])


def test_datetimes_are_compared_by_value():
    assert results_match(
        [[datetime(2020, 1, 1)]],
        [[datetime(2020, 1, 1)]],
    )


def test_empty_results_match_each_other():
    # Both queries correctly finding nothing is agreement, not failure.
    assert results_match([], [])


def test_missing_results_never_match():
    assert not results_match(None, [[1]])


@pytest.fixture(scope="module")
def gold():
    return pd.read_csv(GOLD_SET)


def test_gold_set_has_the_expected_columns(gold):
    assert list(gold.columns) == ["question", "bucket", "expected_sql"]


def test_every_answerable_question_carries_a_reference_query(gold):
    answerable = gold[gold["bucket"] != "unanswerable"]

    missing = answerable[answerable["expected_sql"].isna()]

    assert missing.empty, f"no reference SQL for: {list(missing['question'])}"


def test_unanswerable_questions_carry_no_reference_query(gold):
    # A reference query would mean the question is answerable after all.
    unanswerable = gold[gold["bucket"] == "unanswerable"]

    assert unanswerable["expected_sql"].isna().all()


def test_the_gold_set_is_large_enough_to_mean_something(gold):
    assert len(gold) >= 40
    assert gold["bucket"].nunique() >= 5


def test_questions_are_unique(gold):
    assert gold["question"].duplicated().sum() == 0
