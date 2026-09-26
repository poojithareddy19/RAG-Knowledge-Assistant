"""Retrieval evaluation over the summaries, with a fake index and a gold file."""

from src.evaluation.evaluator import evaluate_retrieval, gold_keys
from src.utils.schemas import Summary


class FakeIndex:
    """Returns a fixed ranking whatever is asked."""

    def __init__(self, ranking):
        self.ranking = ranking

    def search(self, question, k=None):
        return [
            Summary(kind=kind, subject=subject, text="", score=0.9)
            for kind, subject in self.ranking
        ][:k]


def _gold(tmp_path, rows):
    path = tmp_path / "gold.csv"
    lines = ["question,expected_answer,relevant_subjects"]
    lines += [f'"{q}","{a}","{s}"' for q, a, s in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_gold_keys_are_kind_and_subject():
    assert gold_keys("float:1900083; region:Arabian Sea") == {
        "float:1900083",
        "region:Arabian Sea",
    }
    assert gold_keys("") == set()


def test_a_hit_in_the_top_k_scores_and_its_rank_is_reported(tmp_path):
    gold = _gold(tmp_path, [("tell me about float 1900083", "x", "float:1900083")])
    index = FakeIndex([("region", "Arabian Sea"), ("float", "1900083")])

    report = evaluate_retrieval(index, gold, k=5)

    assert report["n_questions"] == 1
    assert report["aggregate"]["hit@k"] == 1.0
    assert report["aggregate"]["mrr"] == 0.5


def test_unanswerable_questions_are_counted_but_not_scored(tmp_path):
    """There is nothing for retrieval to find, so scoring the miss would
    punish the index for a question the generation metrics should check."""
    gold = _gold(
        tmp_path,
        [
            ("tell me about float 1900083", "x", "float:1900083"),
            ("what is the wind speed", "declined", ""),
        ],
    )
    index = FakeIndex([("float", "1900083")])

    report = evaluate_retrieval(index, gold, k=5)

    assert report["n_questions"] == 1
    assert report["n_unanswerable"] == 1
    assert report["aggregate"]["hit@k"] == 1.0
