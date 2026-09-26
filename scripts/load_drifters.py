"""Fetch Global Drifter Program buoys for the Indian Ocean and load them.

The second in-situ platform in this database. Argo floats profile the water
column; drifters ride the surface and report where the current took them. The
problem statement asks for a system extensible to other in-situ observations,
and this is that claim tested rather than asserted: a different instrument,
a different table shape, and nothing downstream rewritten to accommodate it.

Source is the Global Drifter Program 6-hourly quality controlled product,
served by Ifremer's ERDDAP. Public, no key.

Usage:
    python scripts/load_drifters.py --year 2023
    python scripts/load_drifters.py --year 2023 --refresh   # re-download

The download is cached under data/raw/drifters/ so reloading does not re-fetch.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run from a plain checkout. `python scripts/x.py` puts scripts/ on sys.path,
# not the repository, so `src` is not importable without this line.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import io
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.ingestion.regions import fallback_count, region_for
from src.utils.db import cursor

load_dotenv()

ERDDAP = "https://erddap.ifremer.fr/erddap/tabledap/drifter_6hour_qc.csv"

CACHE = Path("data/raw/drifters")

# The same Indian Ocean box the Argo selection uses, so the two platforms
# describe the same water rather than two different oceans.
BOX = {
    "lat_min": -40.0,
    "lat_max": 30.0,
    "lon_min": 30.0,
    "lon_max": 120.0,
}

COLUMNS = "ID,WMO,time,latitude,longitude,sst,ve,vn,typebuoy"

BATCH = 5000

UPSERT_DRIFTER = """
INSERT INTO drifters (buoy_id, wmo, buoy_type, first_seen, last_seen)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (buoy_id) DO UPDATE
SET wmo        = COALESCE(EXCLUDED.wmo, drifters.wmo),
    buoy_type  = COALESCE(NULLIF(EXCLUDED.buoy_type, ''), drifters.buoy_type),
    first_seen = LEAST(drifters.first_seen, EXCLUDED.first_seen),
    last_seen  = GREATEST(drifters.last_seen, EXCLUDED.last_seen)
"""

INSERT_OBSERVATION = """
INSERT INTO drifter_observations (
    buoy_id, obs_time, latitude, longitude, region,
    sst_c, eastward_velocity_m_s, northward_velocity_m_s, geom
)
VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s,
    ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
)
ON CONFLICT (buoy_id, obs_time) DO NOTHING
"""


def parse_rows(text: str) -> list[dict]:
    """Turn an ERDDAP CSV response into rows, dropping unusable ones.

    ERDDAP writes two header lines, the column names and then their units.
    The units line is data to ``csv.DictReader`` and has to go, or the first
    observation of every download is the string "degrees_north".

    A fix with no position or no time is dropped rather than stored: it cannot
    be placed in a region, drawn on a track, or ordered against anything.
    """
    lines = text.splitlines()

    if len(lines) < 2:
        return []

    reader = csv.DictReader(io.StringIO("\n".join([lines[0]] + lines[2:])))

    out = []

    for row in reader:
        latitude = _number(row.get("latitude"))
        longitude = _number(row.get("longitude"))
        buoy = _integer(row.get("ID"))

        if latitude is None or longitude is None or buoy is None:
            continue

        if not (row.get("time") or "").strip():
            continue

        out.append(
            {
                "buoy_id": buoy,
                "wmo": _integer(row.get("WMO")),
                "buoy_type": (row.get("typebuoy") or "").strip(),
                "obs_time": row["time"].strip(),
                "latitude": latitude,
                "longitude": longitude,
                "sst_c": _number(row.get("sst")),
                "eastward_velocity_m_s": _number(row.get("ve")),
                "northward_velocity_m_s": _number(row.get("vn")),
            }
        )

    return out


# ERDDAP reports a missing value two ways and this loader only caught one.
# Alongside NaN it writes the numeric fill value -999999, which parses as a
# perfectly good float and was stored as one: 189 of 139,971 observations, and
# enough to put the mean surface current speed in the Bay of Bengal at 6,243
# metres per second. No real value of any column this parses comes near it.
# Latitude bottoms out at -90, sea surface temperature at about -2, and a
# current at a couple of metres per second, so one threshold guards them all.
_FILL = -9999.0


def _number(value) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None

    # ERDDAP writes NaN for a missing measurement rather than an empty cell.
    if number != number:
        return None

    return None if number <= _FILL else number


def _integer(value) -> int | None:
    number = _number(value)

    return None if number is None else int(number)


def fetch(year: int, refresh: bool = False) -> str:
    """The year's observations inside the box, from cache when it is there."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"drifters_{year}.csv"

    if cached.exists() and not refresh:
        print(f"reading cached {cached}")

        return cached.read_text(encoding="utf-8")

    query = (
        f"{COLUMNS}"
        f"&latitude>={BOX['lat_min']}&latitude<={BOX['lat_max']}"
        f"&longitude>={BOX['lon_min']}&longitude<={BOX['lon_max']}"
        f"&time>={year}-01-01T00:00:00Z&time<={year + 1}-01-01T00:00:00Z"
    )

    print(f"downloading {year} from {ERDDAP}")

    response = requests.get(
        f"{ERDDAP}?{query}",
        timeout=1800,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()

    cached.write_text(response.text, encoding="utf-8")

    return response.text


def load(rows) -> tuple[int, int]:
    """Insert the buoys and their fixes. Returns (buoys, observations)."""
    buoys: dict[int, dict] = {}

    for row in rows:
        day = row["obs_time"][:10]
        seen = buoys.setdefault(
            row["buoy_id"],
            {
                "wmo": row["wmo"],
                "buoy_type": row["buoy_type"],
                "first": day,
                "last": day,
            },
        )
        seen["first"] = min(seen["first"], day)
        seen["last"] = max(seen["last"], day)

    inserted = 0

    with cursor() as cur:
        cur.executemany(
            UPSERT_DRIFTER,
            [
                (
                    buoy_id,
                    meta["wmo"],
                    meta["buoy_type"],
                    meta["first"],
                    meta["last"],
                )
                for buoy_id, meta in buoys.items()
            ],
        )

        # Batched rather than a round trip per fix. A year of six-hourly
        # positions is well over a hundred thousand rows, and one execute
        # each turns a minute of work into most of an hour.
        batch = []

        for row in rows:
            batch.append(
                (
                    row["buoy_id"],
                    row["obs_time"],
                    row["latitude"],
                    row["longitude"],
                    region_for(row["latitude"], row["longitude"]),
                    row["sst_c"],
                    row["eastward_velocity_m_s"],
                    row["northward_velocity_m_s"],
                    row["longitude"],
                    row["latitude"],
                )
            )

            if len(batch) >= BATCH:
                cur.executemany(INSERT_OBSERVATION, batch)
                inserted += len(batch)
                batch.clear()
                print(f"  {inserted} observations", flush=True)

        if batch:
            cur.executemany(INSERT_OBSERVATION, batch)
            inserted += len(batch)

    return len(buoys), inserted


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download instead of reading the cached CSV",
    )
    args = parser.parse_args(argv)

    rows = parse_rows(fetch(args.year, args.refresh))

    print(f"{len(rows)} usable observations in the Indian Ocean box")

    if not rows:
        raise SystemExit("nothing to load")

    buoys, inserted = load(rows)

    print(f"{buoys} buoys, {inserted} observations sent")

    fallbacks = fallback_count()

    if fallbacks:
        print(
            f"{fallbacks} positions fell back to the latitude and longitude "
            "rule because no region polygon contained them"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
