"""Load basin polygons into the regions table from a GeoJSON file.

    python scripts/load_regions.py

The file is not downloaded and not committed. IHO Sea Areas is a third party
dataset with its own licence and its own terms, so the operator fetches it,
agrees to those terms, and puts it at data/raw/regions.geojson. A script that
quietly pulls a licensed dataset on someone's behalf makes that decision for
them, which is why the absence of the file is a clear error here rather than a
download.

Where to get it: the IHO Sea Areas layer from marineregions.org, exported as
GeoJSON in EPSG:4326. Any file works as long as each feature carries a name
property and a polygon geometry; the names this project expects are listed in
WANTED below, and the matching is case insensitive on a substring so that
"Arabian Sea" finds a feature named "Arabian Sea" whatever else the publisher
put around it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run from a plain checkout. `python scripts/x.py` puts scripts/ on sys.path,
# not the repository, so `src` is not importable without this line.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from pathlib import Path

from dotenv import load_dotenv

from src.utils.db import cursor

load_dotenv()

GEOJSON = Path("data/raw/regions.geojson")

# The three names the rest of the project uses. Anything else in the file is
# ignored rather than loaded, because a region this code has never heard of
# would appear in a GROUP BY that every saved query and every gold answer
# assumes has exactly three rows.
WANTED = (
    "Arabian Sea",
    "Bay of Bengal",
    "Indian Ocean",
)

# What the IHO calls a basin and what this schema calls it are not always the
# same word. The Southern Indian Ocean here is the open ocean south of the two
# named seas, which the IHO layer publishes as the Indian Ocean.
RENAME = {
    "Indian Ocean": "Southern Indian Ocean",
}

UPSERT = """
INSERT INTO regions (name, geom)
VALUES (
    %s,
    ST_Multi(
        ST_CollectionExtract(
            ST_MakeValid(
                ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)
            ),
            3
        )
    )
)
ON CONFLICT (name) DO UPDATE
SET geom = EXCLUDED.geom
"""


def read_features(path: Path = GEOJSON) -> list[dict]:
    """Every feature in the file, or a readable error explaining what is missing."""
    if not path.exists():
        raise SystemExit(
            f"missing {path}\n"
            "\n"
            "This script does not download the polygons. Fetch the IHO Sea "
            "Areas layer from https://www.marineregions.org/downloads.php , "
            "accept its licence, export it as GeoJSON in EPSG:4326 and save "
            f"it as {path}."
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}") from exc

    features = data.get("features")

    if not features:
        raise SystemExit(f"{path} has no features")

    return features


def name_of(feature: dict) -> str:
    """The feature's name, from whichever property the publisher used."""
    properties = feature.get("properties") or {}

    for key in ("name", "NAME", "Name", "sea", "SEA", "MRGID_name"):
        value = properties.get(key)

        if value:
            return str(value).strip()

    return ""


def select(features, wanted=WANTED) -> dict[str, dict]:
    """Map each wanted region to the feature that names it.

    Matching is a case insensitive substring both ways, because publishers
    wrap the name: "Arabian Sea" and "Arabian Sea (Persian Gulf excluded)"
    are the same basin for this purpose.
    """
    out: dict[str, dict] = {}

    for feature in features:
        label = name_of(feature).lower()

        if not label:
            continue

        for target in wanted:
            if target in out:
                continue

            if target.lower() in label or label in target.lower():
                out[target] = feature

    return out


def main() -> int:
    features = read_features()

    print(f"{len(features)} features in {GEOJSON}")

    chosen = select(features)

    missing = [name for name in WANTED if name not in chosen]

    if missing:
        raise SystemExit(
            "the file does not contain: "
            + ", ".join(missing)
            + "\nfound: "
            + ", ".join(sorted({name_of(f) for f in features if name_of(f)})[:20])
        )

    loaded = 0

    with cursor() as cur:
        for name, feature in chosen.items():
            stored = RENAME.get(name, name)

            cur.execute(
                UPSERT,
                (
                    stored,
                    json.dumps(feature["geometry"]),
                ),
            )

            print(f"loaded {stored}")
            loaded += 1

    print(f"{loaded} regions in the table")

    return 0


if __name__ == "__main__":
    sys.exit(main())
