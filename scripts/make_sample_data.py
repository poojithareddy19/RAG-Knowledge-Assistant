# scripts/make_sample_data.py

"""Generate synthetic ARGO-shaped data with a known warming trend."""

import random
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from src.utils.db import cursor

load_dotenv()


REGIONS = [
    "Arabian Sea",
    "Bay of Bengal",
    "Southern Indian Ocean",
]

DEPTHS = [
    5,
    25,
    50,
    100,
    200,
    500,
    1000,
    1500,
]

# Planted signal: 0.02 C per year of surface warming
WARMING_PER_YEAR = 0.02

START = datetime(
    2005,
    1,
    1,
    tzinfo=timezone.utc,
)


def surface_temp(region, when):
    """Generate a surface temperature with a known warming trend."""

    base = {
        "Arabian Sea": 27.5,
        "Bay of Bengal": 28.5,
        "Southern Indian Ocean": 18.0,
    }[region]

    years = (when - START).days / 365.25

    season = 1.2 * random.uniform(0.8, 1.2)

    month_effect = season * (
        1 if when.month in (4, 5, 6, 10) else -1
    )

    return (
        base
        + WARMING_PER_YEAR * years
        + month_effect
        + random.gauss(0, 0.35)
    )


def temp_at_depth(surface, pressure):
    """Generate temperature decreasing with depth."""

    return (
        surface
        - 12.0 * (pressure / 1000.0) ** 0.45
        + random.gauss(0, 0.2)
    )


def main(n_floats=40, cycles_per_float=60):
    """Generate synthetic floats, profiles, and measurements."""

    with cursor() as cur:

        for i in range(n_floats):

            float_id = 2900000 + i
            region = REGIONS[i % len(REGIONS)]

            cur.execute(
                """
                INSERT INTO floats (
                    float_id,
                    platform,
                    project,
                    first_seen,
                    last_seen
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (float_id) DO NOTHING
                """,
                (
                    float_id,
                    "APEX",
                    "INCOIS",
                    START.date(),
                    datetime.now(timezone.utc).date(),
                ),
            )

            when = START + timedelta(
                days=random.randint(0, 120)
            )

            for cycle in range(1, cycles_per_float + 1):

                when += timedelta(days=10)

                if when > datetime.now(timezone.utc):
                    break

                lat = random.uniform(-30, 22)
                lon = random.uniform(45, 100)

                cur.execute(
                    """
                    INSERT INTO profiles (
                        float_id,
                        cycle_number,
                        obs_time,
                        latitude,
                        longitude,
                        region
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (float_id, cycle_number)
                    DO NOTHING
                    RETURNING profile_id
                    """,
                    (
                        float_id,
                        cycle,
                        when,
                        lat,
                        lon,
                        region,
                    ),
                )

                row = cur.fetchone()

                if row is None:
                    continue

                profile_id = row[0]

                sst = surface_temp(region, when)

                rows = []

                for pressure in DEPTHS:
                    rows.append(
                        (
                            profile_id,
                            pressure,
                            temp_at_depth(
                                sst,
                                pressure,
                            ),
                            35.0 + random.gauss(0, 0.25),
                            1,
                        )
                    )

                cur.executemany(
                    """
                    INSERT INTO measurements (
                        profile_id,
                        pressure_dbar,
                        temperature_c,
                        salinity_psu,
                        qc_flag
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    rows,
                )

    print(f"done: {n_floats} floats")


if __name__ == "__main__":
    main()