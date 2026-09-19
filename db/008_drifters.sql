-- db/008_drifters.sql
-- A second in-situ platform: surface drifting buoys.
--
-- Argo floats profile the water column and surface every ten days. Drifters
-- sit at the surface and report continuously as they are carried by the
-- current. Same ocean, different instrument, different shape of data, so they
-- get their own tables rather than being forced into profiles/measurements:
-- a drifter has no cycle, no depth levels and no descent, and pretending
-- otherwise would put NULLs down every column that makes a profile a profile.
--
-- What they share is what the questions are asked in: a position, a time, a
-- region and a temperature. That is enough for the trajectory chart, the
-- region grouping and the distance queries to work on both without changing
-- a line of either.

CREATE TABLE IF NOT EXISTS drifters (
    buoy_id BIGINT PRIMARY KEY,
    wmo BIGINT,
    buoy_type TEXT,
    first_seen DATE,
    last_seen DATE
);

CREATE TABLE IF NOT EXISTS drifter_observations (
    observation_id BIGSERIAL PRIMARY KEY,
    buoy_id BIGINT NOT NULL REFERENCES drifters(buoy_id),
    obs_time TIMESTAMPTZ NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    region TEXT,
    -- Sea surface bulk temperature, the only thing a standard drifter measures
    -- besides where it is.
    sst_c DOUBLE PRECISION,
    -- Surface current, derived from how the buoy itself moved.
    eastward_velocity_m_s DOUBLE PRECISION,
    northward_velocity_m_s DOUBLE PRECISION,
    geom geography(Point, 4326),
    -- One fix per buoy per timestamp. Re-running the loader is then safe.
    UNIQUE (buoy_id, obs_time)
);

CREATE INDEX IF NOT EXISTS idx_drifter_obs_time
    ON drifter_observations (obs_time);

CREATE INDEX IF NOT EXISTS idx_drifter_obs_buoy
    ON drifter_observations (buoy_id);

CREATE INDEX IF NOT EXISTS idx_drifter_obs_region
    ON drifter_observations (region);

CREATE INDEX IF NOT EXISTS idx_drifter_obs_geom
    ON drifter_observations
    USING GIST (geom);

GRANT SELECT ON drifters TO gda_ro;
GRANT SELECT ON drifter_observations TO gda_ro;
