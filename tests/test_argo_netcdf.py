"""Parsing tests against a synthetic file shaped like a real ARGO profile.

Building the file here rather than committing one keeps the suite offline and
makes each ARGO convention an explicit, readable setup line.
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from src.ingestion.argo_netcdf import read_file

FILL = 99999.0

# 1950-01-01 plus 25567 days.
JULD_2020 = 25567.0


def _chars(values, width):
    """Pad each string to ``width`` and split it into single bytes."""
    return np.array(
        [list(value.ljust(width)[:width]) for value in values],
        dtype="S1",
    )


@pytest.fixture
def argo_file(tmp_path):
    """Three profiles: real-time, delayed-mode, and one with a bad position."""
    path = tmp_path / "R1901393_001.nc"

    dataset = nc.Dataset(path, "w", format="NETCDF4")

    dataset.createDimension("N_PROF", 3)
    dataset.createDimension("N_LEVELS", 4)
    dataset.createDimension("STRING8", 8)
    dataset.createDimension("STRING32", 32)
    dataset.createDimension("STRING64", 64)

    def text(name, values, width, dim):
        var = dataset.createVariable(name, "S1", ("N_PROF", dim))
        var[:] = _chars(values, width)

    def flag(name, values):
        var = dataset.createVariable(name, "S1", ("N_PROF",))
        var[:] = np.array(values, dtype="S1")

    def numbers(name, values, dtype="f8"):
        var = dataset.createVariable(name, dtype, ("N_PROF",))
        var[:] = np.array(values)

    def grid(name, rows):
        var = dataset.createVariable(
            name,
            "f4",
            ("N_PROF", "N_LEVELS"),
            fill_value=FILL,
        )
        var[:] = np.array(rows, dtype="f4")

    def grid_qc(name, rows):
        var = dataset.createVariable(name, "S1", ("N_PROF", "N_LEVELS"))
        var[:] = np.array(rows, dtype="S1")

    text("PLATFORM_NUMBER", ["1901393"] * 3, 8, "STRING8")
    text("PLATFORM_TYPE", ["APEX"] * 3, 32, "STRING32")
    text("PROJECT_NAME", ["ARGO INDIA"] * 3, 64, "STRING64")

    flag("DATA_MODE", ["R", "D", "R"])
    flag("POSITION_QC", ["1", "1", "4"])
    flag("JULD_QC", ["1", "1", "1"])

    numbers("CYCLE_NUMBER", [1, 2, 3], dtype="i4")
    numbers("JULD", [JULD_2020] * 3)
    numbers("LATITUDE", [15.0, 15.0, 15.0])
    numbers("LONGITUDE", [65.0, 88.0, 65.0])

    # Level 2 has no pressure, level 3 measured nothing. Both are dropped.
    grid(
        "PRES",
        [
            [5.0, 50.0, FILL, 1000.0],
            [5.0, 50.0, FILL, 1000.0],
            [5.0, 50.0, FILL, 1000.0],
        ],
    )
    grid(
        "TEMP",
        [
            [28.5, 25.0, 10.0, FILL],
            [99.0, 99.0, 99.0, FILL],
            [28.5, 25.0, 10.0, FILL],
        ],
    )
    grid(
        "PSAL",
        [
            [35.1, 35.0, 34.0, FILL],
            [99.0, 99.0, 99.0, FILL],
            [35.1, 35.0, 34.0, FILL],
        ],
    )

    grid_qc("PRES_QC", [["1"] * 4] * 3)
    grid_qc(
        "TEMP_QC",
        [
            ["1", "4", "1", "1"],
            ["4", "4", "4", "4"],
            ["1", "1", "1", "1"],
        ],
    )
    grid_qc("PSAL_QC", [["1"] * 4] * 3)

    # Delayed mode: the adjusted values are the real ones.
    grid(
        "PRES_ADJUSTED",
        [
            [FILL] * 4,
            [5.0, 50.0, FILL, 1000.0],
            [FILL] * 4,
        ],
    )
    grid(
        "TEMP_ADJUSTED",
        [
            [FILL] * 4,
            [26.0, 24.0, 10.0, FILL],
            [FILL] * 4,
        ],
    )
    grid(
        "PSAL_ADJUSTED",
        [
            [FILL] * 4,
            [35.2, 35.1, 34.0, FILL],
            [FILL] * 4,
        ],
    )

    grid_qc("PRES_ADJUSTED_QC", [["1"] * 4] * 3)
    grid_qc("TEMP_ADJUSTED_QC", [["2"] * 4] * 3)
    grid_qc("PSAL_ADJUSTED_QC", [["1"] * 4] * 3)

    dataset.close()

    return path


def test_reads_both_usable_profiles(argo_file):
    profiles = read_file(argo_file)

    assert [p.cycle_number for p in profiles] == [1, 2]


def test_bad_position_profile_is_dropped(argo_file):
    # POSITION_QC 4 means the fix is bad, and region is derived from position.
    assert all(p.cycle_number != 3 for p in read_file(argo_file))


def test_metadata_is_decoded_from_char_arrays(argo_file):
    first = read_file(argo_file)[0]

    assert first.float_id == 1901393
    assert first.platform == "APEX"
    assert first.project == "ARGO INDIA"
    assert first.data_mode == "R"


def test_juld_is_decoded_against_the_argo_epoch(argo_file):
    first = read_file(argo_file)[0]

    assert first.obs_time == pd.Timestamp("2020-01-01", tz="UTC")


def test_region_comes_from_position(argo_file):
    arabian, bengal = read_file(argo_file)

    assert arabian.region == "Arabian Sea"
    assert bengal.region == "Bay of Bengal"


def test_levels_without_pressure_or_values_are_skipped(argo_file):
    first = read_file(argo_file)[0]

    assert [level.pressure_dbar for level in first.levels] == [5.0, 50.0]


def test_per_parameter_flags_are_kept(argo_file):
    surface, deep = read_file(argo_file)[0].levels

    assert surface.temperature_qc == 1
    assert deep.temperature_qc == 4
    assert deep.salinity_qc == 1


def test_overall_flag_is_the_worst_present(argo_file):
    surface, deep = read_file(argo_file)[0].levels

    assert surface.qc_flag == 1

    # Salinity is good here; the row is only as trustworthy as its temperature.
    assert deep.qc_flag == 4


def test_delayed_mode_reads_adjusted_values(argo_file):
    delayed = read_file(argo_file)[1]

    # The raw TEMP is 99.0. Reading it instead of TEMP_ADJUSTED is the classic
    # way to get ARGO wrong.
    assert [level.temperature_c for level in delayed.levels] == [26.0, 24.0]
    assert [level.temperature_qc for level in delayed.levels] == [2, 2]


def test_a_flag_of_eight_is_ranked_better_than_four():
    from src.ingestion.argo_netcdf import _overall_qc

    # Interpolated but usable must not lose to bad on numeric order alone.
    assert _overall_qc([(1.0, 8), (1.0, 4)]) == 4


def test_a_missing_parameter_contributes_no_flag():
    from src.ingestion.argo_netcdf import _overall_qc

    assert _overall_qc([(20.0, 1), (float("nan"), 9)]) == 1
