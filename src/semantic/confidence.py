"""Confidence estimation for an answer drawn from the summaries.

How well the retrieved summaries support an answer is estimated from retrieval
signals alone, with no extra model call. Three interpretable components are
combined with configurable weights:

  1. mean_similarity   average cosine similarity of the top-k summaries. A
                       direct measure of how close the index is to the question.
  2. support           fraction of the top-k whose similarity exceeds
                       ``support_floor``. Distinguishes one lucky hit from
                       broad corroboration.
  3. spread            1 minus the normalised variance of the top-k scores. High
                       variance means the pool mixes relevant and irrelevant
                       text, which lowers confidence.

Multiplicative rather than additive: mean similarity is the primary signal and
the other two can only modulate it, never rescue it. Every number shown in the
UI traces back to a retrieval statistic, which is why this is preferred over
asking the model to rate its own answer; self-reported confidence is poorly
calibrated.
"""
from __future__ import annotations

from statistics import pvariance

from src.utils.config import get_config
from src.utils.schemas import Confidence, Summary

MIN_MEAN_SIM = 0.28  # Hard floor: below this, do not answer at all


def estimate_confidence(retrieved: list[Summary]) -> Confidence:
    cfg = get_config().semantic.confidence
    weights = cfg.weights
    floor = float(cfg.support_floor)

    if not retrieved:
        return Confidence(score=0.0, percent=0, components={}, n_supporting=0)

    scores = [max(0.0, min(1.0, hit.score)) for hit in retrieved]
    mean_sim = sum(scores) / len(scores)

    n_supporting = sum(1 for s in scores if s >= floor)
    support = n_supporting / len(scores)

    # Variance of scores in [0,1] is at most 0.25; normalize against that.
    spread = 1.0 - min(1.0, pvariance(scores) / 0.25) if len(scores) > 1 else 1.0

    score = combine_signals(
        mean_sim=mean_sim,
        support=support,
        spread=spread,
        weights={
            "mean_similarity": float(weights.mean_similarity),
            "support": float(weights.support),
            "spread": float(weights.spread),
        },
    )

    return Confidence(
        score=round(score, 4),
        percent=int(round(score * 100)),
        components={
            "mean_similarity": round(mean_sim, 4),
            "support": round(support, 4),
            "spread": round(spread, 4),
        },
        n_supporting=n_supporting,
    )


def combine_signals(mean_sim, support, spread, weights):
    """Combine retrieval signals into a 0-1 confidence score."""
    if mean_sim < MIN_MEAN_SIM:
        return 0.0

    w_sim = weights.get("mean_similarity", 0.6)
    w_sup = weights.get("support", 0.25)
    w_spr = weights.get("spread", 0.15)

    # Normalize secondary signals into a 0.5-1.0 multiplier band.
    # They can penalize a good primary signal but cannot manufacture one.
    secondary = (
        w_sup * support + w_spr * spread
    ) / max(w_sup + w_spr, 1e-9)

    multiplier = 0.5 + 0.5 * secondary

    score = (mean_sim**w_sim) * multiplier

    return round(min(max(score, 0.0), 1.0), 4)
