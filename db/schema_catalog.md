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
- `pressure_dbar` (double) - how deep the float descended, approximately
  metres. Not the depth of the seabed, which this database does not record
- `temperature_c` (double) - degrees Celsius
- `salinity_psu` (double) - practical salinity units
- `qc_flag` (smallint) - overall quality of the row: the worst flag among the
  parameters that reported a value. 1 means good; ignore rows where
  `qc_flag <> 1`
- `pressure_qc`, `temperature_qc`, `salinity_qc` (smallint) - the ARGO flag for
  that one parameter. Use these when a question is about a single parameter and
  you do not want a bad reading of another one to discard the row

### Biogeochemical columns

Only BGC-ARGO floats carry these sensors, so they are NULL on most rows. A
question about one of them must exclude NULLs or it will average nothing.

- `oxygen_umol_kg` (double) - dissolved oxygen, micromoles per kilogram
- `chlorophyll_mg_m3` (double) - chlorophyll-a, milligrams per cubic metre
- `nitrate_umol_kg` (double) - nitrate, micromoles per kilogram
- `ph_total` (double) - pH on the total scale
- `backscatter_700` (double) - particle backscattering at 700 nm, per metre
- `oxygen_qc`, `chlorophyll_qc`, `nitrate_qc`, `ph_qc`, `backscatter_qc`
  (smallint) - the ARGO flag for that parameter

`qc_flag` summarises the CORE parameters only (pressure, temperature,
salinity). It says nothing about a BGC value, so filter a BGC question on that
parameter's own flag, for example `oxygen_qc = 1`.

ARGO quality flags: 1 good, 2 probably good, 3 probably bad, 4 bad,
5 changed, 8 interpolated, 9 missing.

<!-- drifters -->
## drifters

Surface drifting buoys from the Global Drifter Program. A second in-situ
platform, not Argo floats. A drifter rides the surface and reports where the
current carried it; it has no cycles and no depth levels, so it does not join
to `profiles` or `measurements`.

- `buoy_id` (bigint, PK) - Global Drifter Program identifier
- `wmo` (bigint) - WMO number, NULL for buoys that never got one
- `buoy_type` (text) - hull and drogue type, for example `SVPB`
- `first_seen`, `last_seen` (date)

## drifter_observations

One row per buoy per six-hour fix.

- `observation_id` (bigint, PK)
- `buoy_id` (bigint, FK -> drifters.buoy_id)
- `obs_time` (timestamptz)
- `latitude`, `longitude` (double)
- `region` (text) - same three regions as `profiles`. Note it sits on the
  observation and not on `drifters`: a buoy drifts, so it can report from
  the Arabian Sea in March and the Southern Indian Ocean by September.
  Group by `drifter_observations.region`, never by a region on `drifters`
- `sst_c` (double) - sea surface temperature in degrees Celsius. This is the
  only temperature a drifter measures, and it is at the surface, so it is
  comparable with `measurements.temperature_c` where `pressure_dbar < 10` and
  with nothing deeper
- `eastward_velocity_m_s`, `northward_velocity_m_s` (double) - the two
  components of the surface current in metres per second, derived from how the
  buoy itself moved. This is the only current speed in the database, and it is
  at the surface only. Speed is the hypotenuse of the two,
  `sqrt(power(eastward_velocity_m_s, 2) + power(northward_velocity_m_s, 2))`,
  never their sum. Either may be NULL where the source reported no velocity,
  so exclude NULLs before averaging
- `geom` (geography Point) - the position, for distance queries

There is no QC flag column here: the source product is already quality
controlled, so there is no per-row flag to filter on. Do not invent one, and do
not apply the `qc_flag = 1` rule to this table.
<!-- /drifters -->

## Conventions

- `"surface"` means `pressure_dbar < 10`
- `"deep"` means `pressure_dbar > 1000`
- `"near the equator"` means `latitude BETWEEN -5 AND 5`. It is a latitude
  band, not a region name, and no `region` value means the equator
- A named month and year, such as "March 2023", means that whole calendar
  month: `obs_time >= '2023-03-01' AND obs_time < '2023-04-01'`
- "BGC parameters" means the biogeochemical columns: `oxygen_umol_kg`,
  `chlorophyll_mg_m3`, `nitrate_umol_kg`, `ph_total`, `backscatter_700`
- Always exclude rows with `temperature_c IS NULL` from averages
- Join path for Argo: `measurements -> profiles -> floats`
<!-- drifters -->
- Join path for buoys: `drifter_observations -> drifters`
- The two platforms do not join to each other. A question comparing them is
  answered by aggregating each separately, for example with two CTEs, because
  a float profile and a buoy fix have no shared key
- `qc_flag = 1` applies to `measurements` only, never to `drifter_observations`
- Alias `drifter_observations` as `obs` and `drifters` as `d`. Do not alias it
  as `do`: that is a PostgreSQL keyword and the validator rejects the query
<!-- /drifters -->
- For yearly aggregates use `date_trunc('year', obs_time)`
- Prefer `qc_flag = 1`. Fall back to `temperature_qc = 1` only when the
  question is about temperature alone
- For a biogeochemical question filter on that parameter's own flag and
  exclude its NULLs, never on `qc_flag`

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

### Mean surface oxygen per region, from the floats that measure it

```sql
SELECT
    p.region,
    avg(m.oxygen_umol_kg) AS mean_oxygen
FROM measurements m
JOIN profiles p
    ON p.profile_id = m.profile_id
WHERE m.pressure_dbar < 10
  AND m.oxygen_qc = 1
  AND m.oxygen_umol_kg IS NOT NULL
GROUP BY p.region
ORDER BY p.region;
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