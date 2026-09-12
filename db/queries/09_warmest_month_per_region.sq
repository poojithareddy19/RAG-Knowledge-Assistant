-- Question: The warmest month on record per region.

WITH monthly AS (
    SELECT
        p.region,
        EXTRACT(MONTH FROM p.obs_time) AS month,
        AVG(m.temperature_c) AS mean_temperature
    FROM measurements m
    JOIN profiles p
        ON p.profile_id = m.profile_id
    WHERE m.pressure_dbar < 10
      AND m.qc_flag = 1
      AND m.temperature_c IS NOT NULL
    GROUP BY p.region, month
)
SELECT DISTINCT ON (region)
    region,
    month,
    mean_temperature
FROM monthly
ORDER BY region, mean_temperature DESC;