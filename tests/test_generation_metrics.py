"""The deterministic half of generation evaluation, with no model and no index.

The judged metrics are not tested here because a test that needs a model to
decide whether it passed is not a test.
"""

from types import SimpleNamespace

from src.evaluation.generation_metrics import (
    _cited_pages,
    _content_words,
    _gold_pages,
    _lexical_support,
    summarise,
)


def _source(doc, page, text=""):
    return SimpleNamespace(chunk=SimpleNamespace(doc_name=doc, page=page, text=text))


def test_gold_pages_reads_one_document():
    row = {
        "relevant_documents": "argo_quality_control_manual.pdf",
        "relevant_pages": "12",
    }

    assert _gold_pages(row) == {("argo_quality_control_manual.pdf", "12")}


def test_gold_pages_reads_several_pairwise():
    row = {"relevant_documents": "a.pdf;b.pdf", "relevant_pages": "1;2"}

    assert _gold_pages(row) == {("a.pdf", "1"), ("b.pdf", "2")}


def test_gold_pages_survives_a_missing_page():
    row = {"relevant_documents": "a.pdf;b.pdf", "relevant_pages": "1"}

    assert ("a.pdf", "1") in _gold_pages(row)


def test_citation_is_correct_when_the_gold_page_is_among_the_sources():
    cited = _cited_pages([_source("a.pdf", 12), _source("b.pdf", 3)])

    assert ("a.pdf", "12") in cited


def test_citation_is_wrong_when_only_the_document_matches():
    """Right manual, wrong page, is not a grounded answer."""
    cited = _cited_pages([_source("a.pdf", 99)])

    assert not ({("a.pdf", "12")} & cited)


def test_lexical_support_is_one_when_every_word_came_from_the_passage():
    answer = "Three metres per second"
    sources = [_source("a.pdf", 1, "the drift speed should not exceed three metres per second")]

    assert _lexical_support(answer, sources) == 1.0


def test_lexical_support_is_zero_for_an_answer_from_nowhere():
    sources = [_source("a.pdf", 1, "quality control flags and pressure")]

    assert _lexical_support("Antarctic penguins migrate annually", sources) == 0.0


def test_lexical_support_of_an_empty_answer_is_zero_not_an_error():
    assert _lexical_support("", [_source("a.pdf", 1, "anything")]) == 0.0


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
