"""Select Indian Ocean profile files from the Argo GDAC index, and fetch them.

The GDAC publishes one index row per profile file for the whole global fleet,
which is far more than this project needs. This script is the step between that
index and ``scripts/load_argo_netcdf.py``: it decides which files are worth
downloading, writes the list, and optionally fetches them.

The work is split into pure functions and one ``main`` that does the network,
because only the pure half can be tested without reaching the GDAC, and the
selection rules are the part that is easy to get quietly wrong.

Usage:
    python scripts/fetch_argo_index.py --floats 40 --limit 800 --download
    python scripts/fetch_argo_index.py --bgc --floats 10 --limit 200 --download

Both limits matter. ``--floats`` decides how many floats the sample covers and
``--limit`` how many of their cycles are taken, and a file limit on its own
buys one cycle each from hundreds of floats, which is a map with no tracks on
it and a profile count that is 1 for every float in the fleet.

The BGC run is not optional. A core-only load leaves every biogeochemical
column NULL, and a NULL column reads as "no oxygen here" rather than as "this
float does not carry that sensor".
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import sys
from pathlib import Path

import requests

from src.ingestion.regions import region_for

GDAC = "https://data-argo.ifremer.fr"

CORE_INDEX = "ar_index_global_prof.txt.gz"
BGC_INDEX = "argo_bio-profile_index.txt.gz"

FILELIST = Path("data/raw/argo_filelist.txt")
DOWNLOAD_ROOT = Path("data/raw/argo")
INDEX_ROOT = Path("data/raw/argo_index")

# Delayed mode has been through a human's hands and its adjusted values are the
# ones worth loading, so it wins whenever both modes exist for a cycle.
_MODE_RANK = {"D": 0, "A": 1, "R": 2}


def parse_index(blob: bytes) -> list[dict[str, str]]:
    """Return the index rows as dicts, keyed by the file's own header row.

    Some mirrors serve the index gzipped and some serve it plain, and the
    comment block above the header is several lines of provenance that no CSV
    reader will skip on its own. Both are handled here so the caller never has
    to care which mirror it reached.

    Column positions are deliberately not hard coded. The core and BGC indexes
    agree on the first five columns and diverge after that, and reading the
    header is what lets one parser serve both.
    """
    if blob[:2] == b"\x1f\x8b":
        blob = gzip.decompress(blob)

    text = blob.decode("utf-8", "replace")

    lines = [
        line
        for line in io.StringIO(text)
        if not line.startswith("#") and line.strip()
    ]

    return list(csv.DictReader(lines))


def filter_index(
    rows,
    lat_min: float = -40.0,
    lat_max: float = 30.0,
    lon_min: float = 30.0,
    lon_max: float = 120.0,
    ocean: str = "I",
) -> list[dict[str, str]]:
    """Return the index rows inside the Indian Ocean box.

    Rows missing a latitude, longitude or date are dropped rather than
    guessed at, because a profile with no position cannot be assigned a
    region and would silently pollute every group by.

    The ocean code and the box are both required. The code alone reaches from
    the Southern Ocean to the Mozambique Channel, and the box alone would pick
    up Atlantic floats that happen to share a longitude.
    """
    kept = []

    for row in rows:
        if (row.get("ocean") or "").strip().upper() != ocean:
            continue

        if not (row.get("date") or "").strip():
            continue

        latitude = _number(row.get("latitude"))
        longitude = _number(row.get("longitude"))

        if latitude is None or longitude is None:
            continue

        if not lat_min <= latitude <= lat_max:
            continue

        if not lon_min <= longitude <= lon_max:
            continue

        kept.append(row)

    return kept


def deduplicate(rows) -> list[dict[str, str]]:
    """One row per float and cycle, preferring the best available data mode."""
    best: dict[tuple[str, str], dict[str, str]] = {}

    for row in rows:
        path = (row.get("file") or "").strip()

        if not path:
            continue

        key = (float_id(path), cycle(path))
        current = best.get(key)

        if current is None or _mode_rank(path) < _mode_rank(current["file"]):
            best[key] = row

    return [best[key] for key in sorted(best)]


def spread(rows, limit: int | None = None, floats: int | None = None) -> list[dict[str, str]]:
    """Take ``limit`` rows one float at a time, cycling through the floats.

    Taking the first N rows of the index instead would return N cycles of a
    single float, since the index is ordered by path. Every metric in this
    project groups by float or by region, so a sample of one float is not a
    sample at all.

    The floats themselves are visited region by region rather than in id
    order. WMO ids are handed out in blocks per deployment programme, so id
    order is geography order: the first forty ids inside the box turned out to
    be forty floats south of the equator and not one in the Arabian Sea.
    """
    by_float: dict[str, list[dict[str, str]]] = {}

    for row in rows:
        by_float.setdefault(float_id(row["file"]), []).append(row)

    order = _by_region(by_float)

    # Capping the floats before the files is what produces a time series. The
    # box holds thousands of floats, so a file limit alone would be spent on
    # one cycle each and no float would have a second point to plot.
    if floats is not None:
        order = order[:floats]

    out = []
    depth = 0

    while any(len(by_float[key]) > depth for key in order):
        for key in order:
            cycles = by_float[key]

            if depth >= len(cycles):
                continue

            out.append(cycles[depth])

            if limit is not None and len(out) >= limit:
                return out

        depth += 1

    return out


def _by_region(by_float) -> list[str]:
    """Float ids, interleaved across the regions their first row sits in.

    A float is classified once, by the position of its earliest selected row,
    which is enough to balance the sample. Where it drifted afterwards is a
    question for the database, not for the download list.
    """
    regions: dict[str, list[str]] = {}

    for key in sorted(by_float):
        first = by_float[key][0]

        region = region_for(
            _number(first.get("latitude")) or 0.0,
            _number(first.get("longitude")) or 0.0,
        )

        regions.setdefault(region, []).append(key)

    order = []
    depth = 0
    names = sorted(regions)

    while any(len(regions[name]) > depth for name in names):
        for name in names:
            if depth < len(regions[name]):
                order.append(regions[name][depth])

        depth += 1

    return order


def float_id(path: str) -> str:
    """The float's WMO id, the second segment of ``dac/id/profiles/file.nc``."""
    parts = path.split("/")

    return parts[1] if len(parts) > 1 else ""


def cycle(path: str) -> str:
    """The cycle token, kept as text so a descending profile stays distinct.

    ``R1901393_001.nc`` and ``R1901393_001D.nc`` are two different profiles
    from the same cycle, and parsing the token as an integer would merge them.
    """
    name = path.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0]

    return stem.split("_", 1)[1] if "_" in stem else stem


def _mode_rank(path: str) -> int:
    """Rank the data mode letter that prefixes the filename.

    BGC files carry a sensor prefix before the mode letter, ``BD6901580_001``
    rather than ``D6901580_001``, so the mode is the last letter of the
    leading alphabetic run rather than the first character.
    """
    name = path.rsplit("/", 1)[-1]

    letters = ""

    for character in name:
        if not character.isalpha():
            break

        letters += character

    return _MODE_RANK.get(letters[-1:].upper(), len(_MODE_RANK))


def _number(value) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def fetch(url: str, timeout: int = 300) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    return response.content


def fetch_index(name: str, refresh: bool = False, root: Path = INDEX_ROOT) -> bytes:
    """Read the index from disk, downloading it once if it is not there.

    The core index is 58 MB. Choosing a sample is iterative, and re-fetching
    that on every attempt is a cost paid to the GDAC for nothing.
    """
    cached = root / name

    if cached.exists() and not refresh:
        print(f"reading cached {cached}")

        return cached.read_bytes()

    print(f"downloading {GDAC}/{name}")

    blob = fetch(f"{GDAC}/{name}")

    root.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(blob)

    return blob


def download(rows, root: Path = DOWNLOAD_ROOT) -> tuple[int, int]:
    """Fetch each selected profile file, skipping what is already on disk."""
    fetched = 0
    present = 0

    for row in rows:
        relative = row["file"].strip()
        target = root / relative

        if target.exists():
            present += 1
            continue

        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            target.write_bytes(fetch(f"{GDAC}/dac/{relative}"))
        except requests.RequestException as exc:
            print(f"  failed {relative}: {exc}")
            continue

        fetched += 1

        if fetched % 10 == 0:
            print(f"  downloaded {fetched}")

    return fetched, present


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bgc",
        action="store_true",
        help="select from the biogeochemical index instead of the core index",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="keep at most this many files, spread across floats",
    )
    parser.add_argument(
        "--floats",
        type=int,
        default=None,
        help="sample at most this many distinct floats",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download the index instead of reading the cached copy",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="fetch the selected files into data/raw/argo/",
    )
    parser.add_argument(
        "--filelist",
        type=Path,
        default=FILELIST,
        help="where to write the selected paths",
    )
    args = parser.parse_args()

    index = BGC_INDEX if args.bgc else CORE_INDEX

    rows = parse_index(fetch_index(index, args.refresh))

    print(f"{len(rows)} rows in the index")

    inside = filter_index(rows)

    print(f"{len(inside)} rows in the Indian Ocean box")

    unique = deduplicate(inside)
    selected = spread(unique, args.limit, args.floats)

    floats = {float_id(row["file"]) for row in selected}

    print(f"{len(selected)} files selected across {len(floats)} floats")

    args.filelist.parent.mkdir(parents=True, exist_ok=True)

    existing = []

    if args.filelist.exists():
        existing = [
            line
            for line in args.filelist.read_text().splitlines()
            if line.strip()
        ]

    # Appended rather than overwritten, because the BGC run is a second pass
    # over a different index and must not erase the core selection.
    merged = list(dict.fromkeys(existing + [row["file"].strip() for row in selected]))

    args.filelist.write_text("\n".join(merged) + "\n")

    print(f"wrote {len(merged)} paths to {args.filelist}")

    if args.download:
        fetched, present = download(selected)
        print(f"downloaded {fetched} files ({present} already present)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
