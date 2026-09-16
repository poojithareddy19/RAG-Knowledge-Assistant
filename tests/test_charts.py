"""Chart shape selection.

The grouped case is the one that matters: a query like "mean temperature per
year per region" returns one row per year per region, and plotting that as a
single series draws a sawtooth that jumps between regions at every year.
"""

import pandas as pd

from src.charts.builder import spread_categories


def _long(regions, years):
    return pd.DataFrame(
        [
            (year, region, float(index))
            for index, (year, region) in enumerate(
                (y, r) for y in years for r in regions
            )
        ],
        columns=["year", "region", "mean_temp"],
    )


def test_grouped_result_becomes_one_column_per_group():
    wide = spread_categories(
        _long(["Arabian Sea", "Bay of Bengal"], [2020, 2021])
    )

    assert list(wide.columns) == ["year", "Arabian Sea", "Bay of Bengal"]
    assert len(wide) == 2


def test_too_many_groups_is_left_alone():
    # Fifteen overlapping lines is not a readable chart, so it stays a table.
    wide = spread_categories(_long([f"r{i}" for i in range(15)], [2020]))

    assert list(wide.columns) == ["year", "region", "mean_temp"]


def test_a_numeric_middle_column_is_not_a_group():
    df = pd.DataFrame(
        {
            "year": [2020, 2021],
            "float_id": [2900001, 2900002],
            "mean_temp": [26.0, 26.5],
        }
    )

    assert spread_categories(df).equals(df)


def test_two_column_results_are_untouched():
    df = pd.DataFrame({"year": [2020, 2021], "mean_temp": [26.0, 26.5]})

    assert spread_categories(df).equals(df)
