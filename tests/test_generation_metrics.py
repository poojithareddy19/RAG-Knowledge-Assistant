"""The deterministic half of generation evaluation, with no model and no index.

The judged metrics are not tested here because a test that needs a model to
decide whether it passed is not a test.
"""

from src.evaluation.evaluator import gold_keys
from src.evaluation.generation_metrics import (
    _cited_subjects,
    _content_words,
    _gold_subjects,
    _lexical_support,
    summarise,
)
from src.utils.schemas import Summary


def _source(kind, subject, text=""):
    return Summary(kind=kind, subject=subject, text=text, score=0.7)


def test_gold_subjects_reads_one_subject():
    row = {"relevant_subjects": "float:1901393"}

    assert _gold_subjects(row) == {"float:1901393"}


def test_gold_subjects_reads_several_and_forgives_spaces():
    row = {"relevant_subjects": "float:1901393; region:Arabian Sea"}

    assert _gold_subjects(row) == {"float:1901393", "region:Arabian Sea"}


def test_a_region_name_keeps_its_own_spaces():
    assert gold_keys("region:Bay of Bengal") == {"region:Bay of Bengal"}


def test_citation_is_correct_when_the_gold_subject_is_among_the_sources():
    cited = _cited_subjects([_source("float", "1901393"), _source("region", "Arabian Sea")])

    assert "float:1901393" in cited


def test_citation_is_wrong_when_only_the_kind_matches():
    """An answer about one float written from another float's summary is not
    grounded, however plausible it reads."""
    cited = _cited_subjects([_source("float", "2900999")])

    assert not ({"float:1901393"} & cited)


def test_lexical_support_is_one_when_every_word_came_from_the_summary():
    answer = "Arabian Sea and Bay of Bengal"
    sources = [_source("float", "1", "It reported in the Arabian Sea and Bay of Bengal.")]

    assert _lexical_support(answer, sources) == 1.0


def test_lexical_support_is_zero_for_an_answer_from_nowhere():
    sources = [_source("float", "1", "It recorded 142 profiles from 2015 to 2021.")]

    assert _lexical_support("Antarctic penguins migrate annually", sources) == 0.0


def test_lexical_support_of_an_empty_answer_is_zero_not_an_error():
    assert _lexical_support("", [_source("float", "1", "anything")]) == 0.0


def test_stop_words_do_not_prop_up_the_score():
    """Otherwise an answer made of 'the of and is' would look well supported."""
    assert _content_words("the of and is it") == set()


def test_a_declined_answer_is_not_counted_as_a_wrong_citation():
    """Refusing must not be scored as answering badly, or the system is pushed
    towards answering when it should decline."""
    rows = [
        {"answered": True, "citation_correct": True, "lexical_support": 0.9},
        {"answered": False, "citation_correct": None, "lexical_support": None},
    ]

    out = summarise(rows)

    assert out["answered_rate"] == 0.5
    assert out["citation_correct"] == 1.0
    assert out["n_citation_correct"] == 1


def test_answered_rate_is_reported_beside_every_quality_score():
    """A system that refuses everything scores perfectly on the rest, so the
    denominator has to travel with the numbers."""
    rows = [{"answered": False, "citation_correct": None, "lexical_support": None}] * 3

    out = summarise(rows)

    assert out["answered_rate"] == 0.0
    assert "citation_correct" not in out


def test_an_unreachable_judge_is_unknown_rather_than_a_failure():
    rows = [
        {"answered": True, "citation_correct": True, "faithfulness": True},
        {"answered": True, "citation_correct": True, "faithfulness": None},
    ]

    out = summarise(rows)

    assert out["faithfulness"] == 1.0
    assert out["n_faithfulness"] == 1
