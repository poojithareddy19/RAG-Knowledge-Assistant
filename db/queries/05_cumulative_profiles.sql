-- Question: Profiles per year, with a running cumulative total.

SELECT
    EXTRACT(YEAR FROM obs_time) AS year,
    COUNT(*) AS profile_count,
    SUM(COUNT(*)) OVER (
        ORDER BY EXTRACT(YEAR FROM obs_time)
    ) AS cumulative_profiles
FROM profiles
GROUP BY year
ORDER BY year;