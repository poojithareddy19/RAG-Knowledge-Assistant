import numpy as np
import pytest

from src.analytics.trend_test import (
    compare_periods,
    linear_trend,
    mann_kendall,
)

rng = np.random.default_rng(42)


def test_detects_a_planted_trend():
    years = np.arange(2005, 2026, dtype=float)

    values = (
        27.0
        + 0.02 * (years - 2005)
        + rng.normal(0, 0.05, len(years))
    )

    mk = mann_kendall(values)

    assert mk["p_value"] < 0.05
    assert mk["tau"] > 0

    lin = linear_trend(years, values)

    assert lin["ci_low"] < 0.02 < lin["ci_high"]


def test_finds_nothing_in_pure_noise():
    years = np.arange(2005, 2026, dtype=float)

    values = (
        27.0
        + rng.normal(0, 0.5, len(years))
    )

    assert mann_kendall(values)["p_value"] > 0.05

    lin = linear_trend(years, values)

    assert lin["ci_low"] < 0 < lin["ci_high"]


def test_period_comparison_needs_enough_data():
    years = np.arange(2018, 2022, dtype=float)

    values = np.array(
        [27.0, 27.1, 27.2, 27.3]
    )

    with pytest.raises(ValueError):
        compare_periods(
            years,
            values,
            split_year=2021,
        )