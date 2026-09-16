-- db/004_argo_qc.sql
-- Columns the NetCDF loader can fill but a CSV export cannot.

-- Which variables the profile was read from: R real-time, D delayed,
-- A adjusted. In D and A the scientific values live in the ADJUSTED fields.
ALTER TABLE profiles
    ADD COLUMN IF NOT EXISTS data_mode CHAR(1);

-- ARGO flags each parameter separately: a level can have good temperature and
-- bad salinity. qc_flag stays the summary (the worst flag among parameters
-- that reported a value) so existing queries keep meaning what they meant.
ALTER TABLE measurements
    ADD COLUMN IF NOT EXISTS pressure_qc SMALLINT;

ALTER TABLE measurements
    ADD COLUMN IF NOT EXISTS temperature_qc SMALLINT;

ALTER TABLE measurements
    ADD COLUMN IF NOT EXISTS salinity_qc SMALLINT;

-- Almost every analytical query filters on good data, and after a real load
-- the bad rows are a small minority, so this index earns its keep.
CREATE INDEX IF NOT EXISTS idx_meas_qc
    ON measurements (qc_flag);
