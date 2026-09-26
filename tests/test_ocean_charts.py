"""Dispatch and axis conventions for the domain plots.

Every frame here is built in the test, so the suite needs no database and no
network. The shapes are the ones the SQL generator actually returns.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.charts.ocean import pick_ocean_chart, render_ocean


def test_latitude_and_longitude_pick_a_trajectory():
    df = pd.DataFrame(
        {
            "float_id": [1900083, 1900083],
            "obs_time": pd.to_datetime(["2021-01-01", "2021-01-11"]),
            "latitude": [12.5, 12.9],
            "longitude": [68.0, 68.4],
        }
    )

    assert pick_ocean_chart(df) == "trajectory"


def test_time_pressure_and_a_value_pick_a_section():
    df = pd.DataFrame(
        {
            "obs_time": pd.to_datetime(["2021-01-01", "2021-02-01"]),
            "pressure_dbar": [10.0, 200.0],
            "temperature_c": [28.1, 18.4],
        }
    )

    assert pick_ocean_chart(df) == "section"


def test_pressure_and_a_value_pick_a_profile():
    df = pd.DataFrame(
        {
            "pressure_dbar": [5.0, 50.0, 500.0],
            "temperature_c": [28.5, 25.0, 10.0],
        }
    )

    assert pick_ocean_chart(df) == "profile"


def test_temperature_and_salinity_pick_a_ts_diagram():
    df = pd.DataFrame(
        {
            "temperature_c": [28.5, 25.0, 10.0],
            "salinity_psu": [35.1, 35.0, 34.6],
        }
    )

    assert pick_ocean_chart(df) == "ts_diagram"


def test_a_plain_aggregate_picks_nothing():
    df = pd.DataFrame({"region": ["Arabian Sea"], "count": [246]})

    assert pick_ocean_chart(df) is None


def test_an_empty_frame_picks_nothing():
    assert pick_ocean_chart(pd.DataFrame()) is None
    assert pick_ocean_chart(None) is None


def test_a_track_carrying_temperature_is_still_a_track():
    # First match wins, and a position is a stronger signal than a value.
    df = pd.DataFrame(
        {
            "latitude": [12.5],
            "longitude": [68.0],
            "temperature_c": [28.5],
            "salinity_psu": [35.1],
        }
    )

    assert pick_ocean_chart(df) == "trajectory"


def test_a_section_is_not_mistaken_for_a_profile():
    # Both carry pressure. A depth-time section needs more than one time: this
    # fixture used to be a single row, and one timestamp is one cast, which is
    # a profile. Two times keep what the test was written to pin.
    df = pd.DataFrame(
        {
            "obs_time": pd.to_datetime(["2021-01-01", "2021-02-01"]),
            "pressure_dbar": [10.0, 10.0],
            "salinity_psu": [35.1, 35.2],
        }
    )

    assert pick_ocean_chart(df) == "section"


def test_a_single_time_is_a_profile_not_a_section():
    df = pd.DataFrame(
        {
            "obs_time": pd.to_datetime(["2021-01-01"] * 3),
            "pressure_dbar": [5.0, 100.0, 500.0],
            "salinity_psu": [35.1, 35.3, 34.9],
        }
    )

    assert pick_ocean_chart(df) == "profile"


def test_asking_for_profiles_overlays_them_rather_than_drawing_a_section():
    """Casts at several times are both a section and a set of profiles. The
    question decides, and the problem statement asks for profile comparisons."""
    df = pd.DataFrame(
        {
            "profile_id": [1, 1, 2, 2],
            "obs_time": pd.to_datetime(["2023-03-01"] * 2 + ["2023-03-11"] * 2),
            "pressure_dbar": [5.0, 500.0, 5.0, 500.0],
            "salinity_psu": [35.1, 34.9, 35.2, 34.8],
        }
    )

    assert pick_ocean_chart(df) == "section"
    assert pick_ocean_chart(df, "Show me salinity profiles near the equator") == "profile"
    assert pick_ocean_chart(df, "compare the casts") == "profile"


def test_a_profile_that_carries_its_position_is_still_a_profile():
    """This was drawn as a map. Nearly every profile query returns where each
    cast was taken, and the map rule ran first, so the profile itself was never
    shown."""
    df = pd.DataFrame(
        {
            "latitude": [0.5, 0.5],
            "longitude": [80.1, 80.1],
            "pressure_dbar": [5.0, 500.0],
            "salinity_psu": [35.1, 34.9],
        }
    )

    assert pick_ocean_chart(df) == "profile"


def test_an_identifier_is_never_plotted_as_the_measurement():
    """profile_id is numeric and came first, so it used to be the value drawn
    against depth."""
    from src.charts.ocean import _value_column

    df = pd.DataFrame(
        {
            "profile_id": [7, 7],
            "cycle_number": [3, 3],
            "pressure_dbar": [5.0, 500.0],
            "salinity_psu": [35.1, 34.9],
        }
    )

    assert _value_column(df, ("pressure_dbar",)) == "salinity_psu"


def test_several_casts_get_one_trace_each():
    """One line through every row sorted by pressure zigzagged between casts."""
    df = pd.DataFrame(
        {
            "profile_id": [1, 1, 1, 2, 2, 2],
            "pressure_dbar": [5.0, 100.0, 500.0, 5.0, 100.0, 500.0],
            "salinity_psu": [35.1, 35.3, 34.9, 35.0, 35.4, 34.8],
        }
    )

    figure = render_ocean(df, "profile")

    assert len(figure.data) == 2
    assert all(list(trace.y) == [5.0, 100.0, 500.0] for trace in figure.data)


def test_too_many_casts_are_capped_and_the_title_says_so():
    from src.charts.ocean import MAX_PROFILES

    rows = []
    for cast in range(MAX_PROFILES + 5):
        for depth in (5.0, 500.0):
            rows.append({"profile_id": cast, "pressure_dbar": depth, "salinity_psu": 35.0})

    figure = render_ocean(pd.DataFrame(rows), "profile")

    assert len(figure.data) == MAX_PROFILES
    assert f"first {MAX_PROFILES} of {MAX_PROFILES + 5}" in figure.layout.title.text


def test_pressure_alone_is_not_a_profile():
    # Nothing was measured, so there is no value to plot against depth.
    df = pd.DataFrame({"pressure_dbar": [5.0, 50.0]})

    assert pick_ocean_chart(df) is None


def test_column_matching_is_case_insensitive_and_partial():
    df = pd.DataFrame(
        {
            "AVG_PRESSURE_DBAR": [5.0, 50.0],
            "avg_temperature_c": [28.5, 25.0],
        }
    )

    assert pick_ocean_chart(df) == "profile"


def test_the_profile_axis_is_reversed():
    df = pd.DataFrame(
        {
            "pressure_dbar": [5.0, 50.0, 500.0],
            "temperature_c": [28.5, 25.0, 10.0],
        }
    )

    figure = render_ocean(df, "profile")

    assert figure.layout.yaxis.autorange == "reversed"


def test_the_section_axis_is_reversed():
    df = pd.DataFrame(
        {
            "obs_time": pd.to_datetime(["2021-01-01", "2021-02-01"]),
            "pressure_dbar": [10.0, 200.0],
            "temperature_c": [28.1, 18.4],
        }
    )

    figure = render_ocean(df, "section")

    assert figure.layout.yaxis.autorange == "reversed"


def test_a_trajectory_draws_one_trace_per_float():
    df = pd.DataFrame(
        {
            "float_id": [1900083, 1900083, 1900162, 1900162],
            "obs_time": pd.to_datetime(
                ["2021-01-01", "2021-01-11", "2021-01-01", "2021-01-11"]
            ),
            "latitude": [12.5, 12.9, -5.0, -5.4],
            "longitude": [68.0, 68.4, 80.0, 80.3],
        }
    )

    figure = render_ocean(df, "trajectory")

    assert len(figure.data) == 2
    assert all(trace.mode == "lines+markers" for trace in figure.data)


def test_a_trajectory_without_a_float_id_draws_no_line():
    df = pd.DataFrame(
        {
            "latitude": [12.5, -5.0],
            "longitude": [68.0, 80.0],
        }
    )

    figure = render_ocean(df, "trajectory")

    assert len(figure.data) == 1
    assert figure.data[0].mode == "markers"


def test_a_trajectory_orders_its_points_by_time():
    df = pd.DataFrame(
        {
            "float_id": [1900083, 1900083, 1900083],
            "obs_time": pd.to_datetime(["2021-03-01", "2021-01-01", "2021-02-01"]),
            "latitude": [14.0, 12.0, 13.0],
            "longitude": [68.0, 66.0, 67.0],
        }
    )

    figure = render_ocean(df, "trajectory")

    assert list(figure.data[0].lat) == [12.0, 13.0, 14.0]


def test_a_ts_diagram_plots_points_not_lines():
    df = pd.DataFrame(
        {
            "temperature_c": [28.5, 25.0, 10.0],
            "salinity_psu": [35.1, 35.0, 34.6],
        }
    )

    figure = render_ocean(df, "ts_diagram")

    assert figure.data[0].mode == "markers"
    assert list(figure.data[0].x) == [35.1, 35.0, 34.6]
    assert list(figure.data[0].y) == [28.5, 25.0, 10.0]


def test_an_unknown_kind_is_an_error():
    with pytest.raises(ValueError):
        render_ocean(pd.DataFrame({"a": [1]}), "spectrogram")


def test_a_cast_cut_off_by_the_row_limit_is_not_drawn():
    """Ordered by cast then depth, a LIMIT falls inside the last cast. Drawn, it
    would show a float that stopped halfway down."""
    df = pd.DataFrame(
        {
            "profile_id": [1, 1, 1, 2, 2, 2, 3],
            "pressure_dbar": [5.0, 100.0, 500.0, 5.0, 100.0, 500.0, 5.0],
            "salinity_psu": [35.1, 35.3, 34.9, 35.0, 35.4, 34.8, 35.2],
        }
    )

    figure = render_ocean(df, "profile", truncated=True)

    assert len(figure.data) == 2
    assert "incomplete cast is not drawn" in figure.layout.title.text


def test_a_complete_result_keeps_every_cast():
    df = pd.DataFrame(
        {
            "profile_id": [1, 1, 2, 2],
            "pressure_dbar": [5.0, 500.0, 5.0, 500.0],
            "salinity_psu": [35.1, 34.9, 35.0, 34.8],
        }
    )

    assert len(render_ocean(df, "profile", truncated=False).data) == 2


def test_a_single_cast_that_filled_the_limit_is_labelled_as_possibly_short():
    """It cannot be dropped, since it is the only one, so the title says so
    rather than presenting half a cast as a whole one."""
    df = pd.DataFrame(
        {
            "profile_id": [1, 1, 1],
            "pressure_dbar": [5.0, 100.0, 500.0],
            "salinity_psu": [35.1, 35.3, 34.9],
        }
    )

    figure = render_ocean(df, "profile", truncated=True)

    assert len(figure.data) == 1
    assert "may stop short" in figure.layout.title.text


def test_a_result_that_filled_its_limit_is_recognised():
    """What tells the profile chart that its last cast may be cut off."""
    from src.utils.pipeline import _hit_limit

    assert _hit_limit("SELECT 1 FROM measurements LIMIT 500", 500)
    assert not _hit_limit("SELECT 1 FROM measurements LIMIT 500", 499)
    assert not _hit_limit("SELECT 1 FROM measurements", 500)
