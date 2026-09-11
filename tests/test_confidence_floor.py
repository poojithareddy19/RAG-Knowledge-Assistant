from src.retrieval.confidence import combine_signals


WEIGHTS = {
    "mean_similarity": 0.6,
    "support": 0.25,
    "spread": 0.15,
}


def test_low_similarity_cannot_pass_on_secondary_signals():
    # This is the exact case the old additive formula got wrong.
    score = combine_signals(
        mean_sim=0.10,
        support=1.0,
        spread=1.0,
        weights=WEIGHTS,
    )

    assert score == 0.0


def test_strong_retrieval_scores_high():
    score = combine_signals(
        mean_sim=0.82,
        support=1.0,
        spread=0.9,
        weights=WEIGHTS,
    )

    assert score > 0.7


def test_weak_secondary_signals_penalise_but_do_not_zero():
    strong = combine_signals(
        0.70,
        1.0,
        1.0,
        WEIGHTS,
    )

    weak = combine_signals(
        0.70,
        0.2,
        0.1,
        WEIGHTS,
    )

    assert weak < strong
    assert weak > 0.0