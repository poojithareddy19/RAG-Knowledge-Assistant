-- Question: Median salinity per region.

SELECT
    p.region,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY m.salinity_psu)
        AS median_salinity
FROM measurements m
JOIN profiles p
    ON p.profile_id = m.profile_id
WHERE m.qc_flag = 1
  AND m.salinity_psu IS NOT NULL
GROUP BY p.region
ORDER BY p.region;