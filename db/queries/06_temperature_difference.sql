-- Question: The temperature difference between surface and 500 dbar per profile.

WITH surface AS (
    SELECT
        p.profile_id,
        AVG(m.temperature_c) AS surface_temperature
    FROM measurements m
    JOIN profiles p
        ON p.profile_id = m.profile_id
    WHERE m.pressure_dbar < 10
      AND m.qc_flag = 1
      AND m.temperature_c IS NOT NULL
    GROUP BY p.profile_id
),
deep AS (
    SELECT
        p.profile_id,
        AVG(m.temperature_c) AS temperature_500_dbar
    FROM measurements m
    JOIN profiles p
        ON p.profile_id = m.profile_id
    WHERE m.pressure_dbar BETWEEN 495 AND 505
      AND m.qc_flag = 1
      AND m.temperature_c IS NOT NULL
    GROUP BY p.profile_id
)
SELECT
    s.profile_id,
    s.surface_temperature,
    d.temperature_500_dbar,
    s.surface_temperature - d.temperature_500_dbar
        AS temperature_difference
FROM surface s
JOIN deep d
    ON s.profile_id = d.profile_id
ORDER BY s.profile_id;