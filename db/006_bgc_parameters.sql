-- db/006_bgc_parameters.sql
-- Biogeochemical parameters from BGC-ARGO floats.
--
-- Same grain as the core measurements: a BGC float reports these on the same
-- pressure levels as temperature and salinity, so they are columns here rather
-- than a separate table. Most floats are core-only, so these are NULL for the
-- large majority of rows, which is how the ARGO fleet actually looks.
--
-- Each parameter carries its own QC flag and is filtered by that flag alone.
-- qc_flag stays a summary of the CORE parameters only (pressure, temperature,
-- salinity). Folding BGC into it would mean one bad nitrate reading discarded
-- a perfectly good temperature, which would quietly change the meaning of
-- every query written before this table had BGC in it.

ALTER TABLE measurements
    ADD COLUMN IF NOT EXISTS oxygen_umol_kg DOUBLE PRECISION;

ALTER TABLE measurements
    ADD COLUMN IF NOT EXISTS oxygen_qc SMALLINT;

ALTER TABLE measurements
    ADD COLUMN IF NOT EXISTS chlorophyll_mg_m3 DOUBLE PRECISION;

ALTER TABLE measurements
    ADD COLUMN IF NOT EXISTS chlorophyll_qc SMALLINT;

ALTER TABLE measurements
    ADD COLUMN IF NOT EXISTS nitrate_umol_kg DOUBLE PRECISION;

ALTER TABLE measurements
    ADD COLUMN IF NOT EXISTS nitrate_qc SMALLINT;

ALTER TABLE measurements
    ADD COLUMN IF NOT EXISTS ph_total DOUBLE PRECISION;

ALTER TABLE measurements
    ADD COLUMN IF NOT EXISTS ph_qc SMALLINT;

ALTER TABLE measurements
    ADD COLUMN IF NOT EXISTS backscatter_700 DOUBLE PRECISION;

ALTER TABLE measurements
    ADD COLUMN IF NOT EXISTS backscatter_qc SMALLINT;

-- A BGC question always filters to rows where that parameter is present, and
-- those are a small minority, so a partial index per parameter is worth far
-- more than its size.

CREATE INDEX IF NOT EXISTS idx_meas_oxygen
    ON measurements (profile_id)
    WHERE oxygen_umol_kg IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_meas_chlorophyll
    ON measurements (profile_id)
    WHERE chlorophyll_mg_m3 IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_meas_nitrate
    ON measurements (profile_id)
    WHERE nitrate_umol_kg IS NOT NULL;
