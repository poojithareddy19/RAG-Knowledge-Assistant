# Database Schema

Ocean measurements from ARGO floats. Read-only. PostgreSQL.

## floats

One row per physical float.

- `float_id` (bigint, PK) - WMO identifier
- `platform` (text)
- `project` (text)
- `first_seen` (date)
- `last_seen` (date)

## profiles

One row per surfacing event. A float has many profiles.

- `profile_id` (bigint, PK)
- `float_id` (bigint, FK -> floats.float_id)
- `cycle_number` (int) - increments per float
- `obs_time` (timestamptz) - when the profile was recorded
- `latitude`, `longitude` (double)
- `region` (text) - exactly one of:
  - `Arabian Sea`
  - `Bay of Bengal`
  - `Southern Indian Ocean`
- `data_mode` (char) - `R` real-time, `D` delayed, `A` adjusted. `D` is the
  quality-controlled science record. NULL for rows loaded from CSV.

## measurements

One row per depth level within a profile.

- `measurement_id` (bigint, PK)
- `profile_id` (bigint, FK -> profiles.profile_id)
- `pressure_dbar` (double) - approximately depth in metres
- `temperature_c` (double) - degrees Celsius
- `salinity_psu` (double) - practical salinity units
- `qc_flag` (smallint) - overall quality of the row: the worst flag among the
  parameters that reported a value. 1 means good; ignore rows where
  `qc_flag <> 1`
- `pressure_qc`, `temperature_qc`, `salinity_qc` (smallint) - the ARGO flag for
  that one parameter. Use these when a question is about a single parameter and
  you do not want a bad reading of another one to discard the row

ARGO quality flags: 1 good, 2 probably good, 3 probably bad, 4 bad,
5 changed, 8 interpolated, 9 missing.

## Conventions

- `"surface"` means `pressure_dbar < 10`
- `"deep"` means `pressure_dbar > 1000`
- Always exclude rows with `temperature_c IS NULL` from averages
- Join path: `measurements -> profiles -> floats`
- For yearly aggregates use `date_trunc('year', obs_time)`
- Prefer `qc_flag = 1`. Fall back to `temperature_qc = 1` only when the
  question is about temperature alone

## Examples

### Average surface temperature per year in the Arabian Sea

```sql
SELECT
    date_trunc('year', p.obs_time) AS yr,
    avg(m.temperature_c) AS mean_temp
FROM measurements m
JOIN profiles p
    ON p.profile_id = m.profile_id
WHERE p.region = 'Arabian Sea'
  AND m.pressure_dbar < 10
  AND m.qc_flag = 1
  AND m.temperature_c IS NOT NULL
GROUP BY yr
ORDER BY yr;
```

### Mean surface temperature per region, delayed-mode profiles only

```sql
SELECT
    p.region,
    avg(m.temperature_c) AS mean_temp
FROM measurements m
JOIN profiles p
    ON p.profile_id = m.profile_id
WHERE p.data_mode = 'D'
  AND m.pressure_dbar < 10
  AND m.temperature_qc = 1
  AND m.temperature_c IS NOT NULL
GROUP BY p.region
ORDER BY p.region;
```