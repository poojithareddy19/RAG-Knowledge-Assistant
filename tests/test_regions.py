"""Region assignment, with and without the spatial lookup.

The point of these tests is the fallback. Ingestion runs against databases that
do not have PostGIS, do not have the regions table, or have it empty, and in
every one of those cases a profile still has to get a region rather than a
crash or a NULL. So the lookup is made to fail here on purpose and the answer
is compared with the rule it replaced.
"""

from __future__ import annotations

import pytest

from src.ingestion import regions
from src.ingestion.regions import fallback_count, region_by_bounds, region_for


@pytest.fixture(autouse=True)
def fresh():
    regions.reset()
    yield
    regions.reset()


@pytest.fixture
def no_database(monkeypatch):
    """Make the spatial lookup fail the way an absent database does."""

    def explode(*args, **kwargs):
        raise RuntimeError("no database here")

    monkeypatch.setattr("src.utils.db.fetch_all", explode)


@pytest.fixture
def polygons(monkeypatch):
    """A spatial lookup that claims everything is in the Mozambique Channel."""

    def answer(sql, params=None, readonly=True, timeout_ms=5000):
        return ["name"], [("Mozambique Channel",)]

    monkeypatch.setattr("src.utils.db.fetch_all", answer)


def test_the_inequalities_are_unchanged():
    assert region_by_bounds(15.0, 65.0) == "Arabian Sea"
    assert region_by_bounds(15.0, 88.0) == "Bay of Bengal"
    assert region_by_bounds(-20.0, 70.0) == "Southern Indian Ocean"
    assert region_by_bounds(5.0, 65.0) == "Southern Indian Ocean"


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [
        (15.0, 65.0),
        (15.0, 88.0),
        (-20.0, 70.0),
        (5.0, 65.0),
        (0.0, 100.0),
    ],
)
def test_without_a_spatial_lookup_the_inequality_answer_is_returned(
    no_database, latitude, longitude
):
    assert region_for(latitude, longitude) == region_by_bounds(latitude, longitude)


def test_a_fallback_is_counted(no_database):
    assert fallback_count() == 0

    region_for(15.0, 65.0)
    region_for(-20.0, 70.0)

    assert fallback_count() == 2


def test_a_polygon_hit_is_not_counted_as_a_fallback(polygons):
    assert region_for(-18.0, 41.0) == "Mozambique Channel"
    assert fallback_count() == 0


def test_the_polygon_answer_beats_the_inequalities(polygons):
    # The old rule puts this position in the Southern Indian Ocean. The
    # polygons are the whole reason for this task, so they win.
    assert region_by_bounds(-18.0, 41.0) == "Southern Indian Ocean"
    assert region_for(-18.0, 41.0) == "Mozambique Channel"


def test_an_empty_regions_table_falls_back(monkeypatch):
    def nothing(sql, params=None, readonly=True, timeout_ms=5000):
        return ["name"], []

    monkeypatch.setattr("src.utils.db.fetch_all", nothing)

    assert region_for(15.0, 65.0) == "Arabian Sea"
    assert fallback_count() == 1


def test_the_lookup_is_not_retried_after_it_fails(monkeypatch):
    """One failed query per profile would make a load unusably slow."""
    calls = []

    def explode(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("no database here")

    monkeypatch.setattr("src.utils.db.fetch_all", explode)

    for step in range(5):
        region_for(15.0 + step, 65.0)

    assert len(calls) == 1
    assert fallback_count() == 5


def test_repeated_positions_are_cached(polygons, monkeypatch):
    calls = []

    def counted(sql, params=None, readonly=True, timeout_ms=5000):
        calls.append(params)
        return ["name"], [("Arabian Sea",)]

    monkeypatch.setattr("src.utils.db.fetch_all", counted)

    for _ in range(4):
        region_for(15.00001, 65.00001)

    assert len(calls) == 1


def test_reset_clears_the_counter(no_database):
    region_for(15.0, 65.0)

    assert fallback_count() == 1

    regions.reset()

    assert fallback_count() == 0
