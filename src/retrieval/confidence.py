"""Confidence estimation.

We estimate how well the retrieved context supports an answer using only
retrieval signals (no extra LLM call). Three interpretable components are
combined with configurable weights:

  1. mean_similarity  — average cosine sim of the top-k chunks. Direct measure
                        of how close the corpus is to the question.
  2. support          — fraction of top-k chunks whose sim exceeds
                        ``support_floor``. Distinguishes "one lucky hit" from
                        "broad corroboration".
  3. spread           — 1 - normalized variance of the top-k scores. High
                        variance means the pool is a mix of relevant and
                        irrelevant text, which lowers confidence.

    confidence = w1*mean_similarity + w2*support + w3*spread     (clamped 0-1)

This is intentionally simple and explainable — every number shown in the UI can
be traced back to a retrieval statistic. See docs/design_decisions.md for why we
prefer this over, e.g., asking the LLM to self-report confidence (which is
poorly calibrated).
"""
from __future__ import annotations

from statistics import pvariance

from src.utils.config import get_config
from src.utils.schemas import Confidence, RetrievedChunk


def estimate_confidence(retrieved: list[RetrievedChunk]) -> Confidence:
    cfg = get_config().confidence
    weights = cfg.weights
    floor = float(cfg.support_floor)

    if not retrieved:
        return Confidence(score=0.0, percent=0, components={}, n_supporting=0)

    scores = [max(0.0, min(1.0, rc.score)) for rc in retrieved]
    mean_sim = sum(scores) / len(scores)

    n_supporting = sum(1 for s in scores if s >= floor)
    support = n_supporting / len(scores)

    # Variance of scores in [0,1] is at most 0.25; normalize against that.
    spread = 1.0 - min(1.0, pvariance(scores) / 0.25) if len(scores) > 1 else 1.0

    raw = (
        float(weights.mean_similarity) * mean_sim
        + float(weights.support) * support
        + float(weights.spread) * spread
    )
    score = max(0.0, min(1.0, raw))

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
