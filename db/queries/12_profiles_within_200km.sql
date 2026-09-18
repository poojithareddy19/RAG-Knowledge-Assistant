-- Question: Profiles within 200 km of a given coordinate (15.0, 65.0).

WITH target AS (
    SELECT ST_SetSRID(
        ST_MakePoint(65.0, 15.0),
        4326
    )::geography AS at
)
SELECT
    p.profile_id,
    p.float_id,
    p.obs_time,
    p.latitude,
    p.longitude,
    ST_Distance(p.geom, target.at) / 1000.0 AS distance_km
FROM profiles p
CROSS JOIN target
WHERE ST_DWithin(p.geom, target.at, 200000)
ORDER BY distance_km;
