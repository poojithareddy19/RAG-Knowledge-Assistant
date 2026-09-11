-- db/001_schema.sql
-- ARGO float data.
-- One float has many profiles; one profile has many measurements.

CREATE TABLE IF NOT EXISTS floats (
    float_id BIGINT PRIMARY KEY,
    platform TEXT,
    project TEXT,
    first_seen DATE,
    last_seen DATE
);

CREATE TABLE IF NOT EXISTS profiles (
    profile_id BIGSERIAL PRIMARY KEY,
    float_id BIGINT NOT NULL REFERENCES floats(float_id),
    cycle_number INTEGER,
    obs_time TIMESTAMPTZ NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    region TEXT,
    UNIQUE (float_id, cycle_number)
);

CREATE TABLE IF NOT EXISTS measurements (
    measurement_id BIGSERIAL PRIMARY KEY,
    profile_id BIGINT NOT NULL
        REFERENCES profiles(profile_id)
        ON DELETE CASCADE,
    pressure_dbar DOUBLE PRECISION,
    temperature_c DOUBLE PRECISION,
    salinity_psu DOUBLE PRECISION,
    qc_flag SMALLINT DEFAULT 1
);

-- pressure_dbar is roughly depth in metres.

-- Indexes chosen for the queries we actually run:
-- by time, by place, by float

CREATE INDEX IF NOT EXISTS idx_profiles_time
    ON profiles (obs_time);

CREATE INDEX IF NOT EXISTS idx_profiles_region
    ON profiles (region);

CREATE INDEX IF NOT EXISTS idx_profiles_float
    ON profiles (float_id);

CREATE INDEX IF NOT EXISTS idx_meas_profile
    ON measurements (profile_id);

CREATE INDEX IF NOT EXISTS idx_meas_pressure
    ON measurements (pressure_dbar);