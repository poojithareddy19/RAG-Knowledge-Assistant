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
    # Both carry pressure, so the section rule has to be tested first.
    df = pd.DataFrame(
        {
            "obs_time": pd.to_datetime(["2021-01-01"]),
            "pressure_dbar": [10.0],
            "salinity_psu": [35.1],
        }
    )

    assert pick_ocean_chart(df) == "section"


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
