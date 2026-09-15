"""Paired comparison of two pipeline configurations."""

import numpy as np
from scipy import stats


def required_pairs(
    baseline_rate,
    expected_lift,
    alpha=0.05,
    power=0.80,
):
    """Rough paired sample size for a difference in proportions."""
    p1 = baseline_rate
    p2 = min(baseline_rate + expected_lift, 0.999)

    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)

    p_bar = (p1 + p2) / 2

    numerator = (
        z_a * np.sqrt(2 * p_bar * (1 - p_bar))
        + z_b
        * np.sqrt(
            p1 * (1 - p1) + p2 * (1 - p2)
        )
    ) ** 2

    n = numerator / max((p2 - p1) ** 2, 1e-12)

    # Pairing typically buys back roughly a third.
    return int(np.ceil(n * 0.66))


def mcnemar(a_success, b_success):
    """Paired test on two boolean lists of equal length."""
    a = np.asarray(a_success, dtype=bool)
    b = np.asarray(b_success, dtype=bool)

    if a.shape != b.shape:
        raise ValueError(
            "both arms must cover the same questions"
        )

    b_only = int(np.sum(~a & b))
    a_only = int(np.sum(a & ~b))

    discordant = a_only + b_only

    if discordant == 0:
        return {
            "p_value": 1.0,
            "note": "identical outcomes",
            "a_only": 0,
            "b_only": 0,
        }

    # Exact binomial test: more honest than chi-square
    # on small counts.
    p = stats.binomtest(
        b_only,
        discordant,
        0.5,
    ).pvalue

    return {
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "discordant": discordant,
        "p_value": float(p),
    }


def summarise(
    a_success,
    b_success,
    alpha=0.05,
    labels=("A", "B"),
):
    """Summarise paired comparison of two pipeline arms."""
    a = np.asarray(a_success, dtype=bool)
    b = np.asarray(b_success, dtype=bool)

    rate_a = a.mean()
    rate_b = b.mean()

    test = mcnemar(a, b)

    lift = rate_b - rate_a
    significant = test["p_value"] < alpha

    winner = labels[1] if lift > 0 else labels[0]

    if significant:
        decision = (
            f"Adopt {winner}: "
            f"{abs(lift) * 100:.1f} point difference, "
            f"p={test['p_value']:.4f}."
        )
    else:
        decision = (
            f"Keep {labels[0]}: "
            f"the {abs(lift) * 100:.1f} point difference "
            f"is not significant "
            f"(p={test['p_value']:.4f}), "
            f"so it could be noise."
        )

    return {
        "n_pairs": int(len(a)),
        f"{labels[0]}_rate": round(float(rate_a), 3),
        f"{labels[1]}_rate": round(float(rate_b), 3),
        "absolute_lift": round(float(lift), 3),
        "significant_at_alpha": significant,
        **test,
        "decision": decision,
    }