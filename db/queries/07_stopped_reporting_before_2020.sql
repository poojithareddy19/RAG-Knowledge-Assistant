-- Question: Floats that stopped reporting before 2020.

SELECT
    float_id,
    MAX(obs_time) AS last_reported
FROM profiles
GROUP BY float_id
HAVING MAX(obs_time) < '2020-01-01'
ORDER BY last_reported;