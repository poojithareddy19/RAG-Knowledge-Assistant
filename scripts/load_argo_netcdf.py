"""Load ARGO profile NetCDF files into the database.

    python scripts/load_argo_netcdf.py path/to/file.nc
    python scripts/load_argo_netcdf.py path/to/dap/dac/incois/

Accepts files or directories; directories are searched recursively for *.nc.
Re-running is safe: a profile already present is left alone rather than
duplicated, so a partial load can simply be repeated.
"""

import sys
from pathlib import Path

# Run from a plain checkout. `python scripts/x.py` puts scripts/ on sys.path,
# not the repository, so `src` is not importable without this line.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.ingestion.argo_netcdf import PARAMETERS, read_paths
from src.ingestion.regions import fallback_count
from src.utils.db import cursor

load_dotenv()


UPSERT_FLOAT = """
INSERT INTO floats (float_id, platform, project, first_seen, last_seen)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (float_id) DO UPDATE
SET platform   = COALESCE(
                     NULLIF(EXCLUDED.platform, ''),
                     floats.platform
                 ),
    project    = COALESCE(
                     NULLIF(EXCLUDED.project, ''),
                     floats.project
                 ),
    first_seen = LEAST(floats.first_seen, EXCLUDED.first_seen),
    last_seen  = GREATEST(floats.last_seen, EXCLUDED.last_seen)
"""

# geom is built here, on insert. 007 added the column and backfilled the rows
# that existed then, but on a new volume the migrations run before any data,
# so every profile this loader wrote had a NULL geom and silently dropped out
# of ST_DWithin distance queries. The drifter loader always did it this way.
INSERT_PROFILE = """
INSERT INTO profiles (
    float_id,
    cycle_number,
    obs_time,
    latitude,
    longitude,
    region,
    data_mode,
    geom
)
VALUES (
    %s, %s, %s, %s, %s, %s, %s,
    ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
)
ON CONFLICT (float_id, cycle_number) DO NOTHING
RETURNING profile_id
"""

# Derived from the parameter list so adding a sensor never means editing an
# INSERT and a value tuple in step.
MEASUREMENT_COLUMNS = (
    ["profile_id", "qc_flag"]
    + [parameter.column for parameter in PARAMETERS]
    + [parameter.qc_column for parameter in PARAMETERS]
)

INSERT_MEASUREMENTS = f"""
INSERT INTO measurements ({", ".join(MEASUREMENT_COLUMNS)})
VALUES ({", ".join(["%s"] * len(MEASUREMENT_COLUMNS))})
"""


def measurement_row(profile_id, level):
    """One level as a tuple matching ``MEASUREMENT_COLUMNS``."""
    return tuple(
        [profile_id, level.qc_flag]
        + [level.values[parameter.column] for parameter in PARAMETERS]
        + [level.flags[parameter.qc_column] for parameter in PARAMETERS]
    )


def profile_row(profile):
    """One profile as a tuple matching ``INSERT_PROFILE``.

    Longitude comes before latitude in the point: ST_MakePoint takes x, y.
    """
    return (
        profile.float_id,
        profile.cycle_number,
        profile.obs_time,
        profile.latitude,
        profile.longitude,
        profile.region,
        profile.data_mode,
        profile.longitude,
        profile.latitude,
    )


def load(targets):
    """Read every profile under ``targets`` and insert what is not there yet."""
    profiles = read_paths(targets)

    print(f"read {len(profiles)} profiles")

    floats = set()
    inserted = 0
    skipped = 0
    measurements = 0

    with cursor() as cur:
        for profile in profiles:
            observed = profile.obs_time.date()

            cur.execute(
                UPSERT_FLOAT,
                (
                    profile.float_id,
                    profile.platform,
                    profile.project,
                    observed,
                    observed,
                ),
            )

            floats.add(profile.float_id)

            cur.execute(INSERT_PROFILE, profile_row(profile))

            row = cur.fetchone()

            if row is None:
                skipped += 1
                continue

            profile_id = row[0]
            inserted += 1

            cur.executemany(
                INSERT_MEASUREMENTS,
                [
                    measurement_row(profile_id, level)
                    for level in profile.levels
                ],
            )

            measurements += len(profile.levels)

    print(
        f"{len(floats)} floats, {inserted} profiles inserted "
        f"({skipped} already present), {measurements} measurements"
    )

    # A fallback is a position no basin polygon contains, answered by the old
    # inequalities. Silence here would be a quiet return to the coarse rule.
    fallbacks = fallback_count()

    if fallbacks:
        print(
            f"{fallbacks} positions fell back to the latitude and longitude "
            "rule because no region polygon contained them"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("pass one or more .nc files or directories")

    load(sys.argv[1:])
