"""Export round trips, with no database and no network.

The claim this module makes is that a downloaded file says what its variables
mean and which query produced them, so the tests reopen the bytes and read the
attributes back rather than checking that a file was written.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pandas as pd
import pytest

from src.utils.export import (
    column_descriptions,
    open_netcdf_bytes,
    to_csv_bytes,
    to_dataset,
    to_netcdf_bytes,
)

SQL = (
    "SELECT p.region, avg(m.temperature_c) AS mean_temp "
    "FROM measurements m JOIN profiles p ON p.profile_id = m.profile_id "
    "GROUP BY p.region"
)

RESULT = {
    "columns": ["region", "temperature_c", "pressure_dbar"],
    "rows": [
        ["Arabian Sea", 27.4, 5.0],
        ["Bay of Bengal", 28.1, 5.0],
        ["Southern Indian Ocean", 24.9, 5.0],
    ],
    "generated_sql": SQL,
}


def test_csv_round_trips_the_table():
    blob = to_csv_bytes(RESULT)
    frame = pd.read_csv(pd.io.common.BytesIO(blob))

    assert list(frame.columns) == ["region", "temperature_c", "pressure_dbar"]
    assert len(frame) == 3
    assert frame["temperature_c"].tolist() == [27.4, 28.1, 24.9]


def test_netcdf_round_trips_the_variables():
    dataset = open_netcdf_bytes(to_netcdf_bytes(RESULT))

    assert set(dataset.variables) >= {"region", "temperature_c", "pressure_dbar"}
    assert dataset["temperature_c"].values.tolist() == [27.4, 28.1, 24.9]


def test_a_known_column_carries_its_units():
    dataset = open_netcdf_bytes(to_netcdf_bytes(RESULT))

    assert dataset["temperature_c"].attrs["units"] == "degree_Celsius"
    assert dataset["pressure_dbar"].attrs["units"] == "dbar"


def test_the_sql_travels_in_the_global_attributes():
    dataset = open_netcdf_bytes(to_netcdf_bytes(RESULT))

    assert dataset.attrs["sql"] == SQL
    assert dataset.attrs["title"] == "ARGO query result"
    assert dataset.attrs["created"]


def test_the_title_is_settable():
    dataset = open_netcdf_bytes(
        to_netcdf_bytes(RESULT, title="surface temperature by region")
    )

    assert dataset.attrs["title"] == "surface temperature by region"


def test_a_column_with_no_catalog_entry_gets_no_units():
    result = {
        "columns": ["mean_temp"],
        "rows": [[25.4]],
        "generated_sql": "SELECT avg(temperature_c) AS mean_temp FROM measurements",
    }

    dataset = to_dataset(result)

    # An alias the catalog has never heard of. A guessed unit here would be a
    # wrong number waiting to be plotted.
    assert "units" not in dataset["mean_temp"].attrs
    assert dataset["mean_temp"].attrs["long_name"] == "mean_temp"


def test_a_catalog_column_gets_its_description_as_long_name():
    dataset = to_dataset(RESULT)

    assert "Celsius" in dataset["temperature_c"].attrs["long_name"]


def test_the_catalog_is_parsed_into_descriptions():
    catalog = column_descriptions()

    assert "temperature_c" in catalog
    # One bullet names several flag columns and describes them together.
    assert catalog["chlorophyll_qc"] == catalog["nitrate_qc"]


def test_a_profile_and_pressure_result_is_indexed_by_both():
    result = {
        "columns": ["profile_id", "pressure_dbar", "temperature_c"],
        "rows": [
            [1, 5.0, 28.5],
            [1, 50.0, 25.0],
            [2, 5.0, 27.9],
            [2, 50.0, 24.4],
        ],
        "generated_sql": "SELECT profile_id, pressure_dbar, temperature_c FROM measurements",
    }

    dataset = to_dataset(result)

    assert set(dataset.dims) == {"profile_id", "pressure_dbar"}
    assert dataset["temperature_c"].shape == (2, 2)


def test_a_plain_table_is_indexed_by_row():
    dataset = to_dataset(RESULT)

    assert set(dataset.dims) == {"obs"}


def test_repeated_profile_and_depth_falls_back_to_rows():
    # Two rows for one profile at one depth is a table, not a grid, and
    # from_dataframe cannot build a grid from repeated coordinates.
    result = {
        "columns": ["profile_id", "pressure_dbar", "temperature_c"],
        "rows": [[1, 5.0, 28.5], [1, 5.0, 28.6]],
        "generated_sql": "SELECT 1",
    }

    assert set(to_dataset(result).dims) == {"obs"}


def test_decimals_from_the_driver_become_floats():
    result = {
        "columns": ["region", "temperature_c"],
        "rows": [["Arabian Sea", Decimal("27.4")]],
        "generated_sql": "SELECT 1",
    }

    dataset = open_netcdf_bytes(to_netcdf_bytes(result))

    assert dataset["temperature_c"].values.tolist() == pytest.approx([27.4])


def test_timezone_aware_timestamps_survive_the_round_trip():
    stamp = dt.datetime(2003, 6, 1, 12, 0, tzinfo=dt.UTC)

    result = {
        "columns": ["obs_time", "temperature_c"],
        "rows": [[stamp, 27.4]],
        "generated_sql": "SELECT 1",
    }

    dataset = open_netcdf_bytes(to_netcdf_bytes(result))

    assert pd.Timestamp(dataset["obs_time"].values[0]) == pd.Timestamp("2003-06-01 12:00")


def test_an_empty_result_still_exports():
    result = {"columns": ["region"], "rows": [], "generated_sql": "SELECT 1"}

    assert to_csv_bytes(result).startswith(b"region")
    assert len(to_netcdf_bytes(result)) > 0
