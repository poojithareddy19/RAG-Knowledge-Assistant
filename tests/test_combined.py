"""The route that answers from the manuals and the database at once.

Three rules are being protected. A compound question must reach both backends
split rather than whole. The synthesis step must never be the reason an answer
is lost, so every failure of it falls back to both answers side by side. And
the combined confidence must not be flattered by its stronger half.
"""

from __future__ import annotations

import pytest

from src.generation import synthesis
from src.generation.synthesis import combine
from src.router import planner
from src.router.classifier import rule_route
from src.router.planner import split

COMPOUND = (
    "what does the QC manual say about a flag of 4 "
    "and how many profiles have one"
)


# ------------------------------------------------------------------
# Routing
# ------------------------------------------------------------------


def test_a_compound_question_routes_to_both():
    assert rule_route(COMPOUND) == "both"


def test_a_manual_question_mentioning_a_measurement_is_not_combined():
    # It names a document and a measurement but asks for no quantity, so it is
    # one question and the database has nothing to add.
    question = "what does the policy document say about temperature limits"

    assert rule_route(question) is None


def test_a_plain_count_is_still_a_data_question():
    assert rule_route("how many profiles are there per region") == "data"


def test_a_chart_word_still_wins():
    # A depth-time section is a picture, and "section" is also a manual word.
    assert rule_route("plot a section of average temperature") == "chart"


# ------------------------------------------------------------------
# Splitting
# ------------------------------------------------------------------


@pytest.fixture
def stub_split(monkeypatch):
    seen = {}

    def fake(question, timeout=60):
        seen["question"] = question
        return seen.get("reply", {})

    monkeypatch.setattr(planner, "_call_model", fake)

    return seen


def test_a_compound_question_is_split_into_two(stub_split):
    stub_split["reply"] = {
        "documents": "what does the QC manual say about a flag of 4",
        "data": "how many profiles have a QC flag of 4",
    }

    documents, data = split(COMPOUND)

    assert documents == "what does the QC manual say about a flag of 4"
    assert data == "how many profiles have a QC flag of 4"


def test_a_failed_split_sends_the_whole_question_to_both(monkeypatch):
    def explode(*args, **kwargs):
        raise TimeoutError("model unreachable")

    monkeypatch.setattr(planner, "_call_model", explode)

    assert split(COMPOUND) == (COMPOUND, COMPOUND)


def test_an_empty_half_falls_back_to_the_whole_question(stub_split):
    stub_split["reply"] = {"documents": "", "data": "how many profiles"}

    documents, data = split(COMPOUND)

    assert documents == COMPOUND
    assert data == "how many profiles"


def test_a_fragment_too_short_to_be_a_question_is_rejected(stub_split):
    stub_split["reply"] = {"documents": "flag 4", "data": "count"}

    assert split(COMPOUND) == (COMPOUND, COMPOUND)


# ------------------------------------------------------------------
# Synthesis
# ------------------------------------------------------------------


@pytest.fixture
def stub_combine(monkeypatch):
    seen = {}

    def fake(prompt, timeout=90):
        seen["prompt"] = prompt
        return seen.get("reply", "A flag of 4 is bad data [manual.pdf p3]. 12 profiles carry one.")

    monkeypatch.setattr(synthesis, "_call_model", fake)

    return seen


def test_both_halves_are_written_into_one_answer(stub_combine):
    text, how = combine(
        COMPOUND,
        "A flag of 4 means bad data [manual.pdf p3].",
        "12 row(s) returned.",
        ["n"],
        [[12]],
    )

    assert how == "model"
    assert "flag of 4" in text


def test_the_rows_are_shown_to_the_model(stub_combine):
    combine(COMPOUND, "doc answer", "data answer", ["region", "n"], [["Bay", 7]])

    assert "region | n" in stub_combine["prompt"]
    assert "Bay | 7" in stub_combine["prompt"]


def test_a_long_result_is_capped_before_the_model_sees_it(stub_combine):
    rows = [[i] for i in range(50)]

    combine(COMPOUND, "doc answer", "data answer", ["n"], rows)

    assert "30 more row(s)" in stub_combine["prompt"]


def test_a_failed_synthesis_keeps_both_answers(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("model unreachable")

    monkeypatch.setattr(synthesis, "_call_model", explode)

    text, how = combine(COMPOUND, "the manual says X", "the data says Y", ["n"], [[1]])

    assert how == "stapled"
    # Neither half may be lost because the sentence could not be written.
    assert "the manual says X" in text
    assert "the data says Y" in text


def test_an_empty_synthesis_keeps_both_answers(stub_combine):
    stub_combine["reply"] = "   "

    text, how = combine(COMPOUND, "the manual says X", "the data says Y")

    assert how == "stapled"
    assert "the manual says X" in text


def test_one_missing_half_is_never_sent_to_the_model(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("nothing to combine, so nothing to call")

    monkeypatch.setattr(synthesis, "_call_model", explode)

    text, how = combine(COMPOUND, "", "the data says Y")

    assert how == "stapled"
    assert "the data says Y" in text


def test_the_prompt_forbids_inventing_numbers(stub_combine):
    combine(COMPOUND, "doc answer", "data answer", ["n"], [[1]])

    assert "Do not round, total, average or compare" in synthesis.SYSTEM
    assert "must appear in the result exactly" in synthesis.SYSTEM
    # A count from the database carries no manual citation, or the answer
    # claims the manual is the source of a number it never mentions.
    assert "never put a citation after it" in synthesis.SYSTEM


# ------------------------------------------------------------------
# Confidence
# ------------------------------------------------------------------


def _confidence(documents, data):
    from src.utils.pipeline import RAGService

    return RAGService._combined_confidence(documents, data)


def test_confidence_is_the_weaker_of_the_two_halves():
    documents = {"answered": True, "confidence": 0.9}
    data = {"answered": True, "confidence": 0.0}

    assert _confidence(documents, data) == 0.0


def test_a_half_that_refused_does_not_count_against_confidence():
    documents = {"answered": False, "confidence": 0.0}
    data = {"answered": True, "confidence": 1.0}

    assert _confidence(documents, data) == 1.0


def test_confidence_is_zero_when_neither_half_answered():
    assert _confidence({"answered": False}, {"answered": False}) == 0.0
