"""Selection rules for the GDAC index, tested without reaching the GDAC.

The index is 58 MB gzipped and the files behind it are a fleet's worth of
NetCDF, so the rules that decide what gets downloaded are tested against rows
built here. Every row in this module is shaped like a real index row.
"""

from __future__ import annotations

import gzip

from scripts.fetch_argo_index import (
    cycle,
    deduplicate,
    filter_index,
    float_id,
    only_floats,
    parse_index,
    spread,
)
from src.ingestion.regions import region_for

HEADER = "file,date,latitude,longitude,ocean,profiler_type,institution,date_update"


def _row(path, latitude, longitude, ocean="I", date="20210115123000"):
    return {
        "file": path,
        "date": date,
        "latitude": latitude,
        "longitude": longitude,
        "ocean": ocean,
        "profiler_type": "846",
        "institution": "IN",
        "date_update": "20220101000000",
    }


def test_keeps_a_row_inside_the_box():
    inside = _row("incois/2902086/profiles/D2902086_001.nc", "12.5", "68.0")

    assert filter_index([inside]) == [inside]


def test_drops_a_row_outside_the_box():
    # Ocean code I, but a South Atlantic position: the code alone is not enough.
    outside = _row("aoml/1901393/profiles/R1901393_001.nc", "-45.0", "20.0")

    assert filter_index([outside]) == []


def test_drops_a_row_with_no_position():
    missing = _row("aoml/1901393/profiles/R1901393_002.nc", "", "")

    assert filter_index([missing]) == []


def test_drops_a_row_whose_position_is_not_a_number():
    broken = _row("aoml/1901393/profiles/R1901393_003.nc", "n/a", "68.0")

    assert filter_index([broken]) == []


def test_drops_a_row_with_no_date():
    undated = _row("incois/2902086/profiles/D2902086_004.nc", "12.5", "68.0", date="")

    assert filter_index([undated]) == []


def test_drops_a_row_from_another_ocean():
    # A position inside the box, tagged Pacific. Both conditions must hold.
    pacific = _row("aoml/5904471/profiles/R5904471_001.nc", "12.5", "68.0", ocean="P")

    assert filter_index([pacific]) == []


def test_box_edges_are_inclusive():
    corner = _row("incois/2902086/profiles/D2902086_005.nc", "-40.0", "30.0")

    assert filter_index([corner]) == [corner]


def test_parse_index_skips_comments_and_reads_the_header():
    text = (
        "# Title : Profile directory file of the Argo GDAC\n"
        "# Description : ...\n"
        "\n"
        f"{HEADER}\n"
        "incois/2902086/profiles/D2902086_001.nc,20210115123000,12.5,68.0,I,846,IN,20220101000000\n"
    )

    rows = parse_index(text.encode())

    assert len(rows) == 1
    assert rows[0]["file"] == "incois/2902086/profiles/D2902086_001.nc"
    assert rows[0]["latitude"] == "12.5"


def test_parse_index_reads_a_gzipped_mirror():
    text = (
        "# Title : Profile directory file\n"
        f"{HEADER}\n"
        "incois/2902086/profiles/D2902086_001.nc,20210115123000,12.5,68.0,I,846,IN,20220101000000\n"
    )

    rows = parse_index(gzip.compress(text.encode()))

    assert [row["ocean"] for row in rows] == ["I"]


def test_parse_index_reads_the_bgc_header_too():
    text = (
        "# Title : Bio-profile directory file\n"
        "file,date,latitude,longitude,ocean,profiler_type,institution,"
        "parameters,parameter_data_mode,date_update\n"
        "coriolis/6901580/profiles/BD6901580_001.nc,20210115123000,12.5,68.0,I,"
        "846,IF,PRES TEMP PSAL DOXY,DDDD,20220101000000\n"
    )

    rows = parse_index(text.encode())

    assert rows[0]["parameters"] == "PRES TEMP PSAL DOXY"
    assert filter_index(rows) == rows


def test_float_id_and_cycle_come_from_the_path():
    path = "incois/2902086/profiles/D2902086_017.nc"

    assert float_id(path) == "2902086"
    assert cycle(path) == "017"


def test_a_descending_profile_is_a_different_cycle():
    assert cycle("incois/2902086/profiles/D2902086_017D.nc") == "017D"


def test_deduplicate_prefers_delayed_mode():
    realtime = _row("incois/2902086/profiles/R2902086_001.nc", "12.5", "68.0")
    delayed = _row("incois/2902086/profiles/D2902086_001.nc", "12.5", "68.0")

    assert deduplicate([realtime, delayed]) == [delayed]
    assert deduplicate([delayed, realtime]) == [delayed]


def test_deduplicate_reads_the_mode_behind_a_bgc_prefix():
    realtime = _row("coriolis/6901580/profiles/BR6901580_001.nc", "12.5", "68.0")
    delayed = _row("coriolis/6901580/profiles/BD6901580_001.nc", "12.5", "68.0")

    assert deduplicate([realtime, delayed]) == [delayed]


def test_deduplicate_keeps_separate_cycles():
    first = _row("incois/2902086/profiles/D2902086_001.nc", "12.5", "68.0")
    second = _row("incois/2902086/profiles/D2902086_002.nc", "12.6", "68.1")

    assert len(deduplicate([first, second])) == 2


def test_spread_takes_one_cycle_per_float_before_a_second():
    rows = [
        _row(f"incois/29020{i}/profiles/D29020{i}_{n:03d}.nc", "12.5", "68.0")
        for i in (11, 22, 33)
        for n in range(1, 4)
    ]

    selected = spread(rows, limit=3)

    assert len({float_id(row["file"]) for row in selected}) == 3


def test_spread_without_a_limit_returns_everything():
    rows = [
        _row(f"incois/2902011/profiles/D2902011_{n:03d}.nc", "12.5", "68.0")
        for n in range(1, 6)
    ]

    assert len(spread(rows)) == 5


def test_spread_visits_every_region_before_repeating_one():
    # Ids ascend with latitude here, the way WMO blocks do in the real index:
    # id order alone would return three southern floats and no Arabian Sea.
    rows = [
        _row("incois/1900001/profiles/D1900001_001.nc", "-20.0", "70.0"),
        _row("incois/1900002/profiles/D1900002_001.nc", "-18.0", "72.0"),
        _row("incois/1900003/profiles/D1900003_001.nc", "-16.0", "74.0"),
        _row("incois/2900004/profiles/D2900004_001.nc", "15.0", "65.0"),
        _row("incois/2900005/profiles/D2900005_001.nc", "15.0", "88.0"),
    ]

    selected = spread(rows, limit=3)

    assert {region_for(float(row["latitude"]), float(row["longitude"])) for row in selected} == {
        "Arabian Sea",
        "Bay of Bengal",
        "Southern Indian Ocean",
    }


def test_only_floats_keeps_every_cycle_of_the_named_floats():
    rows = [
        _row("incois/1900055/profiles/D1900055_001.nc", "12.5", "68.0"),
        _row("incois/1900055/profiles/D1900055_002.nc", "12.6", "68.1"),
        _row("incois/1900083/profiles/D1900083_001.nc", "12.7", "68.2"),
    ]

    kept = only_floats(rows, ["1900055"])

    assert [row["file"] for row in kept] == [
        "incois/1900055/profiles/D1900055_001.nc",
        "incois/1900055/profiles/D1900055_002.nc",
    ]


def test_only_floats_ignores_blank_ids():
    rows = [_row("incois/1900055/profiles/D1900055_001.nc", "12.5", "68.0")]

    assert only_floats(rows, ["", "  "]) == []
