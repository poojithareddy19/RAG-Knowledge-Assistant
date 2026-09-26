"""The profile insert in the NetCDF loader.

What is pinned down: every profile row carries its position as a PostGIS point,
built from longitude then latitude, because a profile inserted without one is
invisible to every distance query and nothing else would notice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import load_argo_netcdf as loader  # noqa: E402

from src.ingestion.argo_netcdf import ArgoProfile  # noqa: E402

PROFILE = ArgoProfile(
    float_id=1900083,
    cycle_number=1,
    obs_time=pd.Timestamp("2001-08-26 09:08:08", tz="UTC"),
    latitude=-29.674,
    longitude=34.711,
    region="Southern Indian Ocean",
    data_mode="D",
    platform="APEX",
    project="Argo UK",
    levels=[],
)


def test_the_insert_sets_geom():
    assert "geom" in loader.INSERT_PROFILE
    assert "ST_MakePoint(%s, %s)" in loader.INSERT_PROFILE
    assert "::geography" in loader.INSERT_PROFILE


def test_the_row_fills_every_placeholder():
    assert loader.INSERT_PROFILE.count("%s") == len(loader.profile_row(PROFILE))


def test_the_point_is_longitude_then_latitude():
    """ST_MakePoint takes x, y. Swapped, a Mozambique Channel profile would be
    stored somewhere in Antarctica and every distance query would miss it."""
    row = loader.profile_row(PROFILE)

    assert row[-2:] == (34.711, -29.674)
    assert row[3:5] == (-29.674, 34.711)
