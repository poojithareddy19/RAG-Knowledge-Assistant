-- Question: Percentage of measurements failing quality control, per year.

SELECT
    EXTRACT(YEAR FROM p.obs_time) AS year,
    COUNT(*) FILTER (WHERE m.qc_flag <> 1) * 100.0
        / COUNT(*) AS qc_failure_percentage
FROM measurements m
JOIN profiles p
    ON p.profile_id = m.profile_id
GROUP BY year
ORDER BY year;