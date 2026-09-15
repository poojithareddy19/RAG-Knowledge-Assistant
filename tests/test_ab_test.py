from src.evaluation.ab_test import mcnemar, required_pairs, summarise


def test_identical_arms_are_not_significant():
    outcomes = [True, False, True, True, False]

    result = mcnemar(outcomes, outcomes)

    assert result["p_value"] == 1.0


def test_clear_improvement_is_detected():
    a = [False] * 20 + [True] * 5
    b = [True] * 20 + [True] * 5

    assert summarise(a, b)["significant_at_alpha"] is True


def test_sample_size_grows_as_effect_shrinks():
    assert required_pairs(0.6, 0.05) > required_pairs(0.6, 0.20)