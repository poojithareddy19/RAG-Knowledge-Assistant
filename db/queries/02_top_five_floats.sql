-- Question: The five floats with the most profiles.

SELECT
    float_id,
    COUNT(*) AS profile_count
FROM profiles
GROUP BY float_id
ORDER BY profile_count DESC
LIMIT 5;