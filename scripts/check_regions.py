"""Count profiles per region, to check a region backfill landed.

    python scripts/check_regions.py

Profiles carry the region and the observation time; measurements carry
neither, so this reads profiles. A region with no rows here has not been
backfilled, and a NULL region is a profile that fell outside every polygon.
"""

import sys
from pathlib import Path

# Run from a plain checkout. `python scripts/x.py` puts scripts/ on sys.path,
# not the repository, so `src` is not importable without this line.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.db import fetch_all

cols, rows = fetch_all("""
    select region,
           count(*) as profiles,
           count(distinct float_id) as floats,
           min(obs_time)::date as first_obs,
           max(obs_time)::date as last_obs,
           count(distinct extract(year from obs_time)) as years
    from profiles
    group by region
    order by profiles desc
""")

print(cols)
for r in rows:
    print(r)