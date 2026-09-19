"""Parsing the drifter feed, with no network and no database.

ERDDAP's CSV has two header rows and writes NaN for a missing measurement.
Both have bitten this parser, so both are pinned here.
"""

from __future__ import annotations

from scripts.load_drifters import parse_rows

HEADER = "ID,WMO,time,latitude,longitude,sst,ve,vn,typebuoy"
UNITS = ",,UTC,degrees_north,degrees_east,degree_C,,,"


def _feed(*rows):
    return "\n".join([HEADER, UNITS, *rows])


def test_a_normal_row_is_parsed():
    text = _feed(
        "300234063538040,5601522,2023-01-01T00:00:00Z,-37.739,47.024,18.853,0.2479,0.07506,SVPB"
    )

    rows = parse_rows(text)

    assert len(rows) == 1
    assert rows[0]["buoy_id"] == 300234063538040
    assert rows[0]["wmo"] == 5601522
    assert rows[0]["latitude"] == -37.739
    assert rows[0]["sst_c"] == 18.853
    assert rows[0]["buoy_type"] == "SVPB"


def test_the_units_row_is_not_read_as_data():
    """ERDDAP's second header line is units, and DictReader cannot tell."""
    rows = parse_rows(
        _feed("1,2,2023-01-01T00:00:00Z,-37.7,47.0,18.8,0.2,0.1,SVPB")
    )

    assert len(rows) == 1
    assert rows[0]["latitude"] == -37.7


def test_a_missing_measurement_becomes_none_not_nan():
    rows = parse_rows(
        _feed("1,2,2023-01-01T00:00:00Z,-37.7,47.0,NaN,NaN,NaN,SVPB")
    )

    assert rows[0]["sst_c"] is None
    assert rows[0]["eastward_velocity_m_s"] is None


def test_a_row_with_no_position_is_dropped():
    rows = parse_rows(
        _feed("1,2,2023-01-01T00:00:00Z,,,18.8,0.2,0.1,SVPB")
    )

    assert rows == []


def test_a_row_with_no_time_is_dropped():
    rows = parse_rows(_feed("1,2,,-37.7,47.0,18.8,0.2,0.1,SVPB"))

    assert rows == []


def test_a_row_with_no_buoy_id_is_dropped():
    rows = parse_rows(
        _feed(",2,2023-01-01T00:00:00Z,-37.7,47.0,18.8,0.2,0.1,SVPB")
    )

    assert rows == []


def test_an_empty_feed_is_empty():
    assert parse_rows("") == []
    assert parse_rows(HEADER) == []


def test_several_rows_keep_their_order():
    rows = parse_rows(
        _feed(
            "1,2,2023-01-01T00:00:00Z,-37.7,47.0,18.8,0.2,0.1,SVPB",
            "1,2,2023-01-01T06:00:00Z,-37.6,47.1,18.9,0.3,0.2,SVPB",
        )
    )

    assert [r["obs_time"] for r in rows] == [
        "2023-01-01T00:00:00Z",
        "2023-01-01T06:00:00Z",
    ]


def test_a_missing_wmo_is_none_rather_than_zero():
    rows = parse_rows(
        _feed("1,,2023-01-01T00:00:00Z,-37.7,47.0,18.8,0.2,0.1,SVPB")
    )

    assert rows[0]["wmo"] is None
