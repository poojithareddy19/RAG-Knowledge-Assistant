-- Question: Monthly mean temperature at 100 dbar for one region, 2015 onward.

SELECT
    date_trunc('month', p.obs_time) AS month,
    AVG(m.temperature_c) AS mean_temperature
FROM measurements m
JOIN profiles p
    ON p.profile_id = m.profile_id
WHERE p.region = 'Arabian Sea'
  AND m.pressure_dbar BETWEEN 95 AND 105
  AND p.obs_time >= '2015-01-01'
  AND m.qc_flag = 1
  AND m.temperature_c IS NOT NULL
GROUP BY month
ORDER BY month;