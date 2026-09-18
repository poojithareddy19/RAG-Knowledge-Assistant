"""Follow up rewriting, with the model call stubbed.

The rule this module has to keep is that a rewrite never costs the user an
answer. Empty history, a model that times out and a model that replies with
nonsense all have to degrade to the question as typed, so each has a test.
"""

from __future__ import annotations

import pytest

from src.router import rewriter
from src.router.rewriter import build_prompt, rewrite

HISTORY = [
    (
        "average surface temperature in the Arabian Sea in 2021",
        "The average surface temperature was 27.4 C.",
    ),
]


@pytest.fixture
def stub(monkeypatch):
    """Capture the prompt and return a canned rewrite."""
    seen = {}

    def fake(prompt, timeout=30):
        seen["prompt"] = prompt
        return seen.get("reply", "average surface temperature in the Arabian Sea in 2022")

    monkeypatch.setattr(rewriter, "_call_model", fake)

    return seen


def test_empty_history_returns_the_question_unchanged(monkeypatch):
    def explode(prompt, timeout=30):
        raise AssertionError("the model must not be called without history")

    monkeypatch.setattr(rewriter, "_call_model", explode)

    assert rewrite("and in 2022?", []) == "and in 2022?"


def test_a_model_failure_returns_the_question_unchanged(monkeypatch):
    def explode(prompt, timeout=30):
        raise TimeoutError("model unreachable")

    monkeypatch.setattr(rewriter, "_call_model", explode)

    assert rewrite("and in 2022?", HISTORY) == "and in 2022?"


def test_an_empty_reply_returns_the_question_unchanged(stub):
    stub["reply"] = "   "

    assert rewrite("and in 2022?", HISTORY) == "and in 2022?"


def test_the_history_and_the_follow_up_both_reach_the_prompt(stub):
    rewrite("and in 2022?", HISTORY)

    prompt = stub["prompt"]

    assert "average surface temperature in the Arabian Sea in 2021" in prompt
    assert "The average surface temperature was 27.4 C." in prompt
    assert "and in 2022?" in prompt


def test_the_rewritten_question_is_returned(stub):
    out = rewrite("and in 2022?", HISTORY)

    assert out == "average surface temperature in the Arabian Sea in 2022"


def test_quotes_and_a_label_are_stripped_from_the_reply(stub):
    stub["reply"] = '"average surface temperature in the Arabian Sea in 2022"'

    assert rewrite("and in 2022?", HISTORY) == (
        "average surface temperature in the Arabian Sea in 2022"
    )


def test_only_the_first_line_of_a_chatty_reply_is_used(stub):
    stub["reply"] = (
        "average surface temperature in the Arabian Sea in 2022\n"
        "I resolved 'and' to the previous region."
    )

    assert rewrite("and in 2022?", HISTORY) == (
        "average surface temperature in the Arabian Sea in 2022"
    )


def test_history_is_capped_at_max_turns(stub, monkeypatch):
    monkeypatch.setattr(
        rewriter,
        "_settings",
        lambda: {"enabled": True, "max_turns": 2},
    )

    history = [(f"question {n}", f"answer {n}") for n in range(1, 6)]

    rewrite("and in 2022?", history)

    prompt = stub["prompt"]

    assert "question 5" in prompt
    assert "question 4" in prompt
    assert "question 3" not in prompt


def test_disabling_the_feature_skips_the_model(monkeypatch):
    def explode(prompt, timeout=30):
        raise AssertionError("the model must not be called when disabled")

    monkeypatch.setattr(rewriter, "_call_model", explode)
    monkeypatch.setattr(
        rewriter,
        "_settings",
        lambda: {"enabled": False, "max_turns": 4},
    )

    assert rewrite("and in 2022?", HISTORY) == "and in 2022?"


def test_the_prompt_forbids_adding_constraints():
    prompt = build_prompt("and in 2022?", HISTORY)

    assert "Add no new constraints" in prompt
    assert "Output only the rewritten question" in prompt


def test_a_turn_with_no_answer_yet_is_still_usable(stub):
    rewrite("and in 2022?", [("what about the Bay of Bengal", "")])

    assert "what about the Bay of Bengal" in stub["prompt"]
