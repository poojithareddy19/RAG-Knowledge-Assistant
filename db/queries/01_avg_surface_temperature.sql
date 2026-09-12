-- Question: Average surface temperature per year, per region.

SELECT
    date_trunc('year', p.obs_time) AS year,
    p.region,
    AVG(m.temperature_c) AS mean_surface_temperature
FROM measurements m
JOIN profiles p
    ON p.profile_id = m.profile_id
WHERE m.pressure_dbar < 10
  AND m.qc_flag = 1
  AND m.temperature_c IS NOT NULL
GROUP BY year, p.region
ORDER BY year, p.region;