-- Question: Profiles within 200 km of a given coordinate (15.0, 65.0).

WITH target AS (
    SELECT
        15.0::double precision AS lat,
        65.0::double precision AS lon
)
SELECT
    p.profile_id,
    p.float_id,
    p.obs_time,
    p.latitude,
    p.longitude,
    6371.0 * 2 * ASIN(
        SQRT(
            POWER(
                SIN(RADIANS(p.latitude - target.lat) / 2),
                2
            )
            + COS(RADIANS(target.lat))
            * COS(RADIANS(p.latitude))
            * POWER(
                SIN(RADIANS(p.longitude - target.lon) / 2),
                2
            )
        )
    ) AS distance_km
FROM profiles p
CROSS JOIN target
WHERE 6371.0 * 2 * ASIN(
    SQRT(
        POWER(SIN(RADIANS(p.latitude - target.lat) / 2), 2)
        + COS(RADIANS(target.lat))
        * COS(RADIANS(p.latitude))
        * POWER(SIN(RADIANS(p.longitude - target.lon) / 2), 2)
    )
) <= 200
ORDER BY distance_km;