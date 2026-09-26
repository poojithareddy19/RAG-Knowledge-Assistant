"""Unit tests for the deterministic core: retrieval metrics and confidence.

These run without any model, database or API key, so they are fast and
CI-friendly.

Run: pytest -q
"""
import pytest

from src.evaluation import metrics as M
from src.semantic.confidence import estimate_confidence
from src.utils.schemas import Summary


# --- retrieval metrics ------------------------------------------------------
def test_recall_precision_hit():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "e"}
    assert M.recall_at_k(retrieved, relevant, 4) == pytest.approx(0.5)
    assert M.precision_at_k(retrieved, relevant, 4) == pytest.approx(0.25)
    assert M.hit_at_k(retrieved, relevant, 4) == 1.0
    assert M.hit_at_k(retrieved, {"z"}, 4) == 0.0


def test_mrr_and_ndcg():
    assert M.mrr(["x", "y", "gold"], {"gold"}) == pytest.approx(1 / 3)
    assert M.mrr(["gold"], {"gold"}) == 1.0
    assert M.mrr(["a", "b"], {"gold"}) == 0.0
    # first-position relevant => perfect nDCG
    assert M.ndcg_at_k(["gold", "x"], {"gold"}, 2) == pytest.approx(1.0)


def test_recall_never_exceeds_one_when_a_subject_is_returned_twice():
    """Two hits on the same subject are still one subject found."""
    retrieved = ["float:1901393", "float:1901393", "float:2900999"]

    assert M.recall_at_k(retrieved, {"float:1901393"}, 5) == 1.0


def test_recall_counts_distinct_relevant_subjects():
    retrieved = ["float:1", "float:1", "float:2"]

    assert M.recall_at_k(retrieved, {"float:1", "float:2", "float:3"}, 5) == 2 / 3


# --- confidence -------------------------------------------------------------
def _hit(score, i):
    return Summary(kind="float", subject=str(i), text="t", score=score, rank=i)


def test_confidence_monotonic():
    strong = estimate_confidence([_hit(0.9, 1), _hit(0.85, 2), _hit(0.8, 3)])
    weak = estimate_confidence([_hit(0.2, 1), _hit(0.1, 2), _hit(0.05, 3)])
    assert strong.score > weak.score
    assert 0.0 <= weak.score <= strong.score <= 1.0
    assert estimate_confidence([]).score == 0.0


def test_a_summary_looked_up_by_name_scores_as_certain():
    """A question that names a float gets that float at similarity 1.0, and
    the confidence has to reflect that rather than average it away."""
    named = estimate_confidence([_hit(1.0, 1)])

    assert named.score > 0.9
    assert named.n_supporting == 1
