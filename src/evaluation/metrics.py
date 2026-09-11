"""Retrieval evaluation metrics.

Pure functions over ranked results, so they are trivially unit-testable and need
no LLM. Each takes the ordered list of retrieved ids and the set of relevant
("gold") ids for a question.

Definitions
-----------
Recall@K     fraction of relevant items that appear in the top-K.
Precision@K  fraction of the top-K that are relevant.
Hit@K        1 if any relevant item is in the top-K, else 0.
MRR          1 / rank of the first relevant item (0 if none).
nDCG@K       discounted cumulative gain normalized by the ideal ordering.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _relevance_flags(retrieved: Sequence[str], relevant: set[str], k: int) -> list[int]:
    return [1 if doc in relevant else 0 for doc in retrieved[:k]]


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(_relevance_flags(retrieved, relevant, k))
    return hits / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(_relevance_flags(retrieved, relevant, k)) / k


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(_relevance_flags(retrieved, relevant, k)) else 0.0


def mrr(retrieved: Sequence[str], relevant: set[str]) -> float:
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    flags = _relevance_flags(retrieved, relevant, k)
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(flags))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(rows: Iterable[dict]) -> dict:
    """Mean each metric across a list of per-question metric dicts."""
    rows = list(rows)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}
