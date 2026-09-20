"""Trend detection on regional time series. Inference, not prediction."""

import numpy as np
from scipy import stats

from src.utils.db import fetch_all

YEARLY_SURFACE = """
SELECT date_trunc('year', p.obs_time)::date AS yr,
       avg(m.temperature_c) AS mean_temp,
       count(*) AS n_obs
FROM measurements m
JOIN profiles p ON p.profile_id = m.profile_id
WHERE p.region = %s
  AND m.pressure_dbar < 10
  AND m.qc_flag = 1
  AND m.temperature_c IS NOT NULL
GROUP BY yr
HAVING count(*) >= 30
ORDER BY yr
"""


def yearly_series(region):
    _, rows = fetch_all(YEARLY_SURFACE, (region,))

    years = np.array(
        [r[0].year for r in rows],
        dtype=float,
    )
    temps = np.array(
        [float(r[1]) for r in rows],
    )
    counts = np.array(
        [int(r[2]) for r in rows],
    )

    return years, temps, counts


def mann_kendall(values):
    """Non-parametric monotonic trend test. Returns S, tau, z, p."""
    n = len(values)

    if n < 4:
        raise ValueError("need at least 4 points")

    s = 0

    for i in range(n - 1):
        s += np.sum(
            np.sign(values[i + 1 :] - values[i])
        )

    # Variance with a correction for tied values
    _, tie_counts = np.unique(
        values,
        return_counts=True,
    )

    tie_term = np.sum(
        tie_counts
        * (tie_counts - 1)
        * (2 * tie_counts + 5)
    )

    var_s = (
        n * (n - 1) * (2 * n + 5) - tie_term
    ) / 18.0

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p = 2 * (1 - stats.norm.cdf(abs(z)))

    tau = s / (0.5 * n * (n - 1))

    return {
        "S": float(s),
        "tau": float(tau),
        "z": float(z),
        "p_value": float(p),
    }


def linear_trend(years, values, alpha=0.05):
    """Least-squares slope with a confidence interval. Gives effect size."""
    res = stats.linregress(years, values)

    n = len(years)

    t_crit = stats.t.ppf(
        1 - alpha / 2,
        df=n - 2,
    )

    margin = t_crit * res.stderr

    return {
        "slope_per_year": float(res.slope),
        "ci_low": float(res.slope - margin),
        "ci_high": float(res.slope + margin),
        "r_squared": float(res.rvalue**2),
        "p_value": float(res.pvalue),
        "n_years": n,
    }


def compare_periods(years, values, split_year):
    """Welch's t-test between two eras. Unequal variances assumed."""
    early = values[years < split_year]
    late = values[years >= split_year]

    if len(early) < 3 or len(late) < 3:
        raise ValueError(
            "need at least 3 years on each side"
        )

    t_stat, p = stats.ttest_ind(
        late,
        early,
        equal_var=False,
    )

    pooled_sd = np.sqrt(
        (
            np.var(early, ddof=1)
            + np.var(late, ddof=1)
        )
        / 2
    )

    cohens_d = (
        (late.mean() - early.mean()) / pooled_sd
        if pooled_sd
        else 0.0
    )

    return {
        "mean_before": float(early.mean()),
        "mean_after": float(late.mean()),
        "difference": float(
            late.mean() - early.mean()
        ),
        "t_statistic": float(t_stat),
        "p_value": float(p),
        "cohens_d": float(cohens_d),
    }


def report(region, alpha=0.05, split_year=2015):
    years, temps, counts = yearly_series(region)

    out = {
        "region": region,
        "years_covered": (
            f"{int(years.min())}-{int(years.max())}"
        ),
        "total_observations": int(counts.sum()),
        "mann_kendall": mann_kendall(temps),
        "linear_trend": linear_trend(
            years,
            temps,
            alpha,
        ),
    }

    try:
        out["period_comparison"] = compare_periods(
            years,
            temps,
            split_year,
        )
    except ValueError as exc:
        out["period_comparison"] = {
            "skipped": str(exc)
        }

    mk_p = out["mann_kendall"]["p_value"]
    slope = out["linear_trend"]["slope_per_year"]
    lo = out["linear_trend"]["ci_low"]
    hi = out["linear_trend"]["ci_high"]

    if mk_p < alpha:
        direction = (
            "warming" if slope > 0 else "cooling"
        )

        out["conclusion"] = (
            f"Significant {direction} trend "
            f"(Mann-Kendall p={mk_p:.4f}). "
            f"Estimated {slope:+.4f} C/year, "
            f"95% CI [{lo:+.4f}, {hi:+.4f}]. "
            f"Over a decade that is {slope * 10:+.2f} C."
        )
    else:
        out["conclusion"] = (
            f"No statistically significant trend "
            f"(p={mk_p:.4f}). "
            f"Point estimate {slope:+.4f} C/year "
            f"but the interval [{lo:+.4f}, {hi:+.4f}] "
            f"includes zero."
        )

    return out