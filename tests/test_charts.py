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


# ---------------------------------------------------------------------------
# Gaps in a time series
# ---------------------------------------------------------------------------

from datetime import UTC, datetime  # noqa: E402

from src.charts.builder import break_at_gaps, find_gaps, render  # noqa: E402

# The shape of the real answer to "number of profiles per year": 2001 to 2009,
# then nothing until 2023.
YEARS = [*range(2001, 2010), 2023, 2024, 2025, 2026]


def test_a_hole_in_a_yearly_series_is_found():
    x = pd.Series(pd.to_datetime([f"{y}-01-01" for y in YEARS]))

    gaps = find_gaps(x)

    assert [(a.year, b.year) for a, b in gaps] == [(2009, 2023)]


def test_numeric_years_work_the_same_way():
    assert find_gaps(pd.Series(YEARS)) == [(2009, 2023)]


def test_a_regular_series_has_no_gaps():
    assert find_gaps(pd.Series(range(2001, 2015))) == []


def test_too_few_points_cannot_establish_a_step():
    assert find_gaps(pd.Series([2001, 2002, 2020])) == []


def test_a_category_axis_has_no_spacing_to_break():
    assert find_gaps(pd.Series(["Arabian Sea", "Bay of Bengal", "x", "y"])) == []


def test_the_break_is_an_empty_row_inside_the_gap():
    df = pd.DataFrame({"yr": YEARS, "n": range(len(YEARS))})

    broken = break_at_gaps(df, "yr", find_gaps(df["yr"]))

    hole = broken[broken["n"].isna()]
    assert len(broken) == len(df) + 1
    assert hole["yr"].tolist() == [2016]


def test_render_draws_the_real_shape_with_timezone_aware_years():
    """date_trunc on a timestamptz arrives as aware datetimes in an object
    column. It has to become a time axis for the gap check to see it."""
    rows = [[datetime(y, 1, 1, tzinfo=UTC), 10 * i] for i, y in enumerate(YEARS)]

    png, kind = render({"columns": ["yr", "profile_count"], "rows": rows})

    assert kind == "line"
    assert png.startswith(b"\x89PNG")


def test_a_one_year_hole_in_a_patchy_series_is_still_a_gap():
    """The Arabian Sea's years. Their median step is a year and a half, which
    let 2023 to 2025 through as if 2024 had been measured."""
    years = [2003, 2004, 2007, 2008, 2009, 2023, 2025]

    assert find_gaps(pd.Series(years)) == [(2004, 2007), (2009, 2023), (2023, 2025)]

    dates = pd.Series(pd.to_datetime([f"{y}-01-01" for y in years]))
    assert [(a.year, b.year) for a, b in find_gaps(dates)] == [
        (2004, 2007),
        (2009, 2023),
        (2023, 2025),
    ]


def test_a_monthly_series_breaks_at_a_missing_month():
    months = pd.Series(pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01", "2023-05-01"]))

    assert [(a.month, b.month) for a, b in find_gaps(months)] == [(3, 5)]


def test_an_irregular_numeric_axis_falls_back_to_the_median():
    depths = pd.Series([5.0, 10.0, 15.0, 20.0, 60.0])

    assert find_gaps(depths) == [(20.0, 60.0)]


def test_a_short_time_series_still_renders():
    """Three yearly points are too few to establish a step, so there are no
    gaps and nothing to shade. The band inset used to be computed anyway, from
    a step of None, and the chart route answered with a 500."""
    rows = [[datetime(y, 1, 1, tzinfo=UTC), n] for y, n in [(2021, 5), (2022, 7), (2023, 6)]]

    png, kind = render({"columns": ["yr", "profile_count"], "rows": rows})

    assert kind == "line"
    assert png.startswith(b"\x89PNG")


def test_the_band_uses_the_step_of_the_real_points():
    """The inserted break rows sit mid-gap, off the 1 January grid, so the
    step has to be read before they are added or it falls back to a median."""
    rows = [[datetime(y, 1, 1, tzinfo=UTC), 1] for y in YEARS]

    png, kind = render({"columns": ["yr", "n"], "rows": rows})

    assert kind == "line"
    assert png.startswith(b"\x89PNG")
