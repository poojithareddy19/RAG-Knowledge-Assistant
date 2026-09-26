"""Assign a named region to a profile position.

Shared by every loader, the Argo NetCDF loader and the drifter loader, so they
cannot disagree about which region a position belongs to.

The boundaries used to be three inequalities, which is coarse for a field that
nearly every query groups by: they put the Gulf of Aden in the Arabian Sea and
the whole Mozambique Channel in the Southern Indian Ocean. They are now real
basin polygons, looked up in PostGIS.

The inequalities are still here, and still correct as a fallback. A position in
open ocean matches no published basin polygon, and ingestion must not fail or
write a NULL region because a float drifted somewhere the IHO did not name. So
a miss falls back to the old rule and increments a counter, because a fallback
that nobody counts is a silent return to the behaviour this replaced. The
loaders print that counter when they finish.
"""

from __future__ import annotations

from functools import lru_cache

# Positions repeat far more than you would expect across a load: a float
# reports from nearly the same place for days. Rounding to three decimals is
# about 100 m, well inside any basin boundary, and turns most of a load's
# lookups into cache hits.
_PRECISION = 3

_fallbacks = 0

# None means "not tried yet". False latches after the first failure so a load
# against a database with no PostGIS does not pay for one failed query per
# profile.
_spatial_available: bool | None = None


def region_for(latitude: float, longitude: float) -> str:
    """The basin this position sits in.

    Polygons first, the historical inequalities when no polygon contains the
    point or the spatial lookup is unavailable.
    """
    global _fallbacks

    name = _from_polygons(
        round(float(latitude), _PRECISION),
        round(float(longitude), _PRECISION),
    )

    if name:
        return name

    _fallbacks += 1

    return region_by_bounds(latitude, longitude)


def region_by_bounds(latitude: float, longitude: float) -> str:
    """The original three inequalities, unchanged.

    Kept as a function rather than inlined into the fallback path so the
    regression test can assert the two agree without a database.
    """
    if latitude > 5 and longitude < 78:
        return "Arabian Sea"

    if latitude > 5:
        return "Bay of Bengal"

    return "Southern Indian Ocean"


def fallback_count() -> int:
    """How many positions have fallen back to the inequalities so far."""
    return _fallbacks


def reset() -> None:
    """Forget the counter and the availability latch. For tests and loaders."""
    global _fallbacks, _spatial_available

    _fallbacks = 0
    _spatial_available = None

    _from_polygons.cache_clear()


@lru_cache(maxsize=100_000)
def _from_polygons(latitude: float, longitude: float) -> str | None:
    """The region whose polygon contains this point, or None.

    Returns None rather than raising for every failure mode there is: no
    PostGIS, no regions table, an empty regions table, no database at all. The
    caller's fallback is the answer in all of them, and a loader that dies
    because a polygon table is missing is worse than one that is coarse.
    """
    global _spatial_available

    if _spatial_available is False:
        return None

    try:
        from src.utils.db import fetch_all

        _, rows = fetch_all(
            """
            SELECT name
            FROM regions
            WHERE ST_Contains(
                geom,
                ST_SetSRID(ST_MakePoint(%s, %s), 4326)
            )
            ORDER BY ST_Area(geom)
            LIMIT 1
            """,
            (longitude, latitude),
            readonly=True,
        )
    except Exception:
        _spatial_available = False

        return None

    _spatial_available = True

    return rows[0][0] if rows else None
