"""Cluster ocean profiles into data-driven water-mass groups."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils.db import fetch_all


PROFILE_FEATURES = """
SELECT
    p.profile_id,
    p.region,
    avg(
        CASE
            WHEN m.pressure_dbar < 10
            THEN m.temperature_c
        END
    ) AS sst,
    avg(
        CASE
            WHEN m.pressure_dbar < 10
            THEN m.salinity_psu
        END
    ) AS sss,
    avg(
        CASE
            WHEN m.pressure_dbar BETWEEN 400 AND 600
            THEN m.temperature_c
        END
    ) AS t500
FROM profiles p
JOIN measurements m
    ON m.profile_id = p.profile_id
WHERE m.qc_flag = 1
GROUP BY
    p.profile_id,
    p.region
HAVING avg(
    CASE
        WHEN m.pressure_dbar < 10
        THEN m.temperature_c
    END
) IS NOT NULL
"""


def load_features():
    """Load profile-level temperature and salinity features."""
    cols, rows = fetch_all(PROFILE_FEATURES)
    df = pd.DataFrame(rows, columns=cols)

    return df.dropna(
        subset=["sst", "sss", "t500"]
    ).astype(
        {
            "sst": float,
            "sss": float,
            "t500": float,
        }
    )


def choose_k(x_scaled, k_range=(2, 8), seed=42):
    """Sweep k and keep the best silhouette score."""
    scores = {}

    for k in range(k_range[0], k_range[1] + 1):
        labels = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=seed,
        ).fit_predict(x_scaled)

        scores[k] = float(
            silhouette_score(x_scaled, labels)
        )

    best = max(scores, key=scores.get)

    return best, scores


def cluster_profiles(k_range=(2, 8), seed=42):
    """Cluster profiles using scaled oceanographic features."""
    df = load_features()

    features = ["sst", "sss", "t500"]

    scaler = StandardScaler()
    x = scaler.fit_transform(df[features])

    k, scores = choose_k(
        x,
        k_range,
        seed,
    )

    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=seed,
    )

    df["cluster"] = model.fit_predict(x)

    centroids = pd.DataFrame(
        scaler.inverse_transform(
            model.cluster_centers_
        ),
        columns=features,
    ).round(2)

    centroids["n_profiles"] = (
        df["cluster"]
        .value_counts()
        .sort_index()
        .values
    )

    # Which region dominates each cluster?
    crosstab = pd.crosstab(
        df["cluster"],
        df["region"],
        normalize="index",
    )

    return {
        "chosen_k": k,
        "silhouette_scores": {
            str(a): round(b, 3)
            for a, b in scores.items()
        },
        "best_silhouette": round(
            scores[k],
            3,
        ),
        "centroids": centroids.to_dict(
            "records"
        ),
        "region_mix": crosstab.round(
            2
        ).to_dict("index"),
        "labelled": df,
    }


def describe(result):
    """Create plain-language cluster descriptions for the README."""
    lines = [
        f"k={result['chosen_k']} chosen by silhouette "
        f"({result['best_silhouette']})"
    ]

    for i, c in enumerate(result["centroids"]):
        mix = result["region_mix"].get(i, {})
        top = max(
            mix,
            key=mix.get,
        ) if mix else "mixed"

        lines.append(
            f"Cluster {i}: "
            f"surface {c['sst']} C, "
            f"salinity {c['sss']} PSU, "
            f"500m {c['t500']} C, "
            f"{c['n_profiles']} profiles, "
            f"mostly {top} "
            f"({mix.get(top, 0):.0%})"
        )

    return "\n".join(lines)