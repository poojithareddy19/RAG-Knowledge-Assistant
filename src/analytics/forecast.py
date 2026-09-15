"""Quantile forecasting of regional monthly surface temperature."""

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.utils.db import fetch_all


MONTHLY = """
SELECT
    date_trunc('month', p.obs_time)::date AS month,
    avg(m.temperature_c) AS mean_temp,
    count(*) AS n_obs
FROM measurements m
JOIN profiles p
    ON p.profile_id = m.profile_id
WHERE p.region = %s
    AND m.pressure_dbar < 10
    AND m.qc_flag = 1
    AND m.temperature_c IS NOT NULL
GROUP BY month
HAVING count(*) >= 10
ORDER BY month
"""


def monthly_frame(region):
    """Load monthly regional surface-temperature observations."""
    cols, rows = fetch_all(MONTHLY, (region,))
    df = pd.DataFrame(rows, columns=cols)

    df["month"] = pd.to_datetime(df["month"])
    df["mean_temp"] = df["mean_temp"].astype(float)

    return df


def make_features(df):
    """Create lag, rolling, and calendar features."""
    out = (
        df.copy()
        .sort_values("month")
        .reset_index(drop=True)
    )

    for lag in (1, 2, 3, 12):
        out[f"lag_{lag}"] = out["mean_temp"].shift(lag)

    out["roll_mean_3"] = (
        out["mean_temp"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    out["roll_std_3"] = (
        out["mean_temp"]
        .shift(1)
        .rolling(3)
        .std()
    )

    out["roll_mean_12"] = (
        out["mean_temp"]
        .shift(1)
        .rolling(12)
        .mean()
    )

    out["month_num"] = out["month"].dt.month
    out["year"] = out["month"].dt.year

    # December and January are adjacent.
    out["month_sin"] = np.sin(
        2 * np.pi * out["month_num"] / 12
    )
    out["month_cos"] = np.cos(
        2 * np.pi * out["month_num"] / 12
    )

    return out.dropna().reset_index(drop=True)


FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_12",
    "roll_mean_3",
    "roll_std_3",
    "roll_mean_12",
    "month_sin",
    "month_cos",
]


def pinball_loss(y_true, y_pred, quantile):
    """Calculate pinball loss for a given quantile."""
    diff = (
        np.asarray(y_true)
        - np.asarray(y_pred)
    )

    return float(
        np.mean(
            np.maximum(
                quantile * diff,
                (quantile - 1) * diff,
            )
        )
    )


def fit_quantile(
    x_train,
    y_train,
    quantile,
    seed=42,
):
    """Fit a LightGBM quantile-regression model."""
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=quantile,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=10,
        random_state=seed,
        verbose=-1,
    )

    model.fit(x_train, y_train)

    return model


def seasonal_naive(df_test, df_all):
    """Use the same month from the previous year as the baseline."""
    del df_all

    preds = []

    for _, row in df_test.iterrows():
        preds.append(row["lag_12"])

    return np.array(preds)


def leave_one_year_out(
    region,
    quantiles=(0.1, 0.5, 0.9),
):
    """Evaluate quantile forecasts using leave-one-year-out validation."""
    df = make_features(
        monthly_frame(region)
    )

    years = sorted(
        df["year"].unique()
    )

    if len(years) < 4:
        raise ValueError(
            "need at least 4 years of monthly data"
        )

    losses = {
        q: []
        for q in quantiles
    }

    baseline_losses = []

    for held_out in years[1:]:
        train = df[
            df["year"] != held_out
        ]

        test = df[
            df["year"] == held_out
        ]

        if len(test) < 3 or len(train) < 24:
            continue

        for q in quantiles:
            model = fit_quantile(
                train[FEATURES],
                train["mean_temp"],
                q,
            )

            preds = model.predict(
                test[FEATURES]
            )

            losses[q].append(
                pinball_loss(
                    test["mean_temp"],
                    preds,
                    q,
                )
            )

        base = seasonal_naive(
            test,
            df,
        )

        baseline_losses.append(
            pinball_loss(
                test["mean_temp"],
                base,
                0.5,
            )
        )

    if not baseline_losses:
        raise ValueError(
            "no valid leave-one-year-out folds"
        )

    summary = {
        f"pinball_q{int(q * 100)}": round(
            float(np.mean(losses[q])),
            4,
        )
        for q in quantiles
    }

    summary["baseline_pinball_q50"] = round(
        float(np.mean(baseline_losses)),
        4,
    )

    summary["model_beats_baseline"] = (
        summary["pinball_q50"]
        < summary["baseline_pinball_q50"]
    )

    summary["improvement_pct"] = round(
        100
        * (
            summary["baseline_pinball_q50"]
            - summary["pinball_q50"]
        )
        / summary["baseline_pinball_q50"],
        1,
    )

    summary["folds"] = len(
        baseline_losses
    )

    return summary


def explain(
    region,
    quantile=0.5,
    max_display=8,
):
    """Return SHAP feature importance for the median model."""
    import shap

    df = make_features(
        monthly_frame(region)
    )

    model = fit_quantile(
        df[FEATURES],
        df["mean_temp"],
        quantile,
    )

    explainer = shap.TreeExplainer(model)

    values = explainer.shap_values(
        df[FEATURES]
    )

    importance = (
        pd.DataFrame(
            {
                "feature": FEATURES,
                "mean_abs_shap": np.abs(values).mean(
                    axis=0
                ),
            }
        )
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .head(max_display)
    )

    return importance.round(
        4
    ).to_dict("records")