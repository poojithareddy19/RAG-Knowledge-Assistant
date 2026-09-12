"""Load an ARGO CSV export (ERDDAP tabledap) into the database."""

import os
import sys

import pandas as pd
from dotenv import load_dotenv

from src.utils.db import cursor

load_dotenv()


COLUMN_MAP = {
    "platform_number": "float_id",
    "cycle_number": "cycle_number",
    "time": "obs_time",
    "latitude": "latitude",
    "longitude": "longitude",
    "pres": "pressure_dbar",
    "temp": "temperature_c",
    "psal": "salinity_psu",
}


def region_for(lat, lon):
    if lat > 5 and lon < 78:
        return "Arabian Sea"

    if lat > 5:
        return "Bay of Bengal"

    return "Southern Indian Ocean"


def load(path_or_url):
    df = pd.read_csv(path_or_url)

    # ERDDAP can put a units row directly under the header.
    # Drop it if present.
    if not df.empty and df.iloc[0].astype(str).str.contains(
        "degree|PSU|decibar",
        case=False,
    ).any():
        df = df.iloc[1:]

    df = df.rename(columns=COLUMN_MAP)

    keep = [c for c in COLUMN_MAP.values() if c in df.columns]
    df = df[keep]

    numeric = [
        "float_id",
        "cycle_number",
        "latitude",
        "longitude",
        "pressure_dbar",
        "temperature_c",
        "salinity_psu",
    ]

    for column in numeric:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df["obs_time"] = pd.to_datetime(
        df["obs_time"],
        errors="coerce",
        utc=True,
    )

    before = len(df)

    df = df.dropna(
        subset=[
            "float_id",
            "cycle_number",
            "obs_time",
            "latitude",
            "longitude",
        ]
    )

    df = df[
        df["temperature_c"].between(-3, 40)
        | df["temperature_c"].isna()
    ]

    print(f"kept {len(df)} of {before} rows after cleaning")

    inserted = 0

    with cursor() as cur:
        for float_id, fgroup in df.groupby("float_id"):
            cur.execute(
                """
                INSERT INTO floats (
                    float_id,
                    platform,
                    project,
                    first_seen,
                    last_seen
                )
                VALUES (%s, 'unknown', 'ARGO', %s, %s)
                ON CONFLICT (float_id) DO UPDATE
                SET last_seen = EXCLUDED.last_seen
                """,
                (
                    int(float_id),
                    fgroup["obs_time"].min().date(),
                    fgroup["obs_time"].max().date(),
                ),
            )

            for cycle, cgroup in fgroup.groupby("cycle_number"):
                head = cgroup.iloc[0]

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
                    ON CONFLICT (float_id, cycle_number) DO NOTHING
                    RETURNING profile_id
                    """,
                    (
                        int(float_id),
                        int(cycle),
                        head["obs_time"],
                        float(head["latitude"]),
                        float(head["longitude"]),
                        region_for(
                            head["latitude"],
                            head["longitude"],
                        ),
                    ),
                )

                row = cur.fetchone()

                if row is None:
                    continue

                rows = [
                    (
                        row[0],
                        _f(measurement.get("pressure_dbar")),
                        _f(measurement.get("temperature_c")),
                        _f(measurement.get("salinity_psu")),
                        1,
                    )
                    for _, measurement in cgroup.iterrows()
                ]

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

                inserted += len(rows)

    print(f"inserted {inserted} measurements")


def _f(value):
    return None if pd.isna(value) else float(value)


if __name__ == "__main__":
    src = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("ARGO_CSV_URL")
    )

    if not src:
        sys.exit("pass a CSV path or set ARGO_CSV_URL")

    load(src)