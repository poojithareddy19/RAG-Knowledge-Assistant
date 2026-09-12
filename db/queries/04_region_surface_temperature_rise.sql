-- Question: Regions ordered by how much surface temperature rose from their first to their last year.

WITH yearly AS (
    SELECT
        p.region,
        EXTRACT(YEAR FROM p.obs_time) AS year,
        AVG(m.temperature_c) AS mean_surface_temperature
    FROM measurements m
    JOIN profiles p
        ON p.profile_id = m.profile_id
    WHERE m.pressure_dbar < 10
      AND m.qc_flag = 1
      AND m.temperature_c IS NOT NULL
    GROUP BY p.region, year
),
with_first_last AS (
    SELECT
        region,
        year,
        mean_surface_temperature,
        FIRST_VALUE(mean_surface_temperature) OVER (
            PARTITION BY region
            ORDER BY year
        ) AS first_year_temperature,
        LAST_VALUE(mean_surface_temperature) OVER (
            PARTITION BY region
            ORDER BY year
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS last_year_temperature
    FROM yearly
)
SELECT DISTINCT
    region,
    first_year_temperature,
    last_year_temperature,
    last_year_temperature - first_year_temperature AS temperature_rise
FROM with_first_last
ORDER BY temperature_rise DESC;