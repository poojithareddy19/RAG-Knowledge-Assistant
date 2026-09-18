-- db/007_postgis.sql
-- Real basin polygons instead of three latitude and longitude inequalities.
--
-- Nearly every question this database answers groups by region, so the region
-- column is load bearing, and until now it was assigned by three straight
-- lines: north of 5 and west of 78 is the Arabian Sea, north of 5 is the Bay
-- of Bengal, everything else is the Southern Indian Ocean. That puts the Gulf
-- of Aden in the Arabian Sea, the Andaman Sea in the Bay of Bengal, and the
-- whole Mozambique Channel in the Southern Indian Ocean.
--
-- Distance comes free with the same change. The old "profiles within 200 km"
-- query spelled out the haversine formula twice and could not use an index,
-- because a computed expression over two columns is not something a B-tree can
-- help with. ST_DWithin on a geography column is one call and hits GIST.

CREATE EXTENSION IF NOT EXISTS postgis;

-- geography rather than geometry: the inputs are degrees on a sphere, and
-- geography measures in metres without anyone choosing a projection first.
ALTER TABLE profiles
    ADD COLUMN IF NOT EXISTS geom geography(Point, 4326);

UPDATE profiles
SET geom = ST_SetSRID(
        ST_MakePoint(longitude, latitude),
        4326
    )::geography
WHERE geom IS NULL;

CREATE INDEX IF NOT EXISTS idx_profiles_geom
    ON profiles
    USING GIST (geom);

-- The polygons themselves are not committed here. They are a third party
-- dataset with its own licence, loaded by scripts/load_regions.py from a file
-- the operator supplies.
CREATE TABLE IF NOT EXISTS regions (
    region_id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    -- geometry and not geography: these are polygons used for containment
    -- tests, never for distance, and geometry keeps ST_Contains cheap.
    geom geometry(MultiPolygon, 4326) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_regions_geom
    ON regions
    USING GIST (geom);

-- Explicit rather than relying on the default privileges in 003_roles.sql,
-- because this file is also run by hand against a database that already
-- exists, where those defaults were set before this table was thought of.
GRANT SELECT ON regions TO gda_ro;
GRANT SELECT ON profiles TO gda_ro;
