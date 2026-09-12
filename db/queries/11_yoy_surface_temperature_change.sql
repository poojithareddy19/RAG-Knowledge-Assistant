-- Question: Year-over-year change in mean surface temperature.

WITH yearly AS (
    SELECT
        EXTRACT(YEAR FROM p.obs_time) AS year,
        AVG(m.temperature_c) AS mean_surface_temperature
    FROM measurements m
    JOIN profiles p
        ON p.profile_id = m.profile_id
    WHERE m.pressure_dbar < 10
      AND m.qc_flag = 1
      AND m.temperature_c IS NOT NULL
    GROUP BY year
)
SELECT
    year,
    mean_surface_temperature,
    mean_surface_temperature
        - LAG(mean_surface_temperature) OVER (ORDER BY year)
        AS yoy_change
FROM yearly
ORDER BY year;