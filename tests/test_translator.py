"""Language detection and translation, with the model call stubbed.

Two rules are being protected. An English question must cost nothing, because
detection is the thing that runs on every query. And a translation that fails
must leave the user exactly where they were, never worse.
"""

from __future__ import annotations

import pytest

from src.router import translator
from src.router.translator import from_english, looks_english, to_english

HINDI = "अरब सागर में औसत सतही तापमान क्या है"
TAMIL = "அரபிக் கடலில் சராசரி மேற்பரப்பு வெப்பநிலை என்ன"
BENGALI = "আরব সাগরে গড় পৃষ্ঠ তাপমাত্রা কত"
ENGLISH = "What is the average surface temperature in the Arabian Sea?"


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setattr(
        translator,
        "_settings",
        lambda: {
            "enabled": True,
            "always_translate": False,
            "answer_in_source_language": True,
        },
    )


@pytest.fixture
def stub(monkeypatch):
    seen = {}

    def fake(system, text, timeout=60):
        seen.setdefault("calls", []).append((system, text))
        return seen.get("reply", ENGLISH)

    monkeypatch.setattr(translator, "_call_model", fake)

    return seen


def test_english_is_recognised_without_a_model_call(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("an English question must not cost a model call")

    monkeypatch.setattr(translator, "_call_model", explode)

    assert to_english(ENGLISH) == (ENGLISH, None)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (HINDI, "Hindi or Marathi"),
        (TAMIL, "Tamil"),
        (BENGALI, "Bengali"),
        ("الطقس في بحر العرب", "Arabic or Urdu"),
    ],
)
def test_a_non_latin_script_is_detected(text, expected):
    english, language = looks_english(text)

    assert english is False
    assert language == expected


def test_digits_and_punctuation_do_not_decide_the_language():
    english, language = looks_english("float 1900083, 2003-2004 (surface)?")

    assert english is True
    assert language is None


def test_a_hindi_question_is_translated_and_its_language_reported(stub):
    question, language = to_english(HINDI)

    assert question == ENGLISH
    assert language == "Hindi or Marathi"
    assert len(stub["calls"]) == 1


def test_a_failed_translation_returns_the_question_unchanged(monkeypatch):
    def explode(*args, **kwargs):
        raise TimeoutError("model unreachable")

    monkeypatch.setattr(translator, "_call_model", explode)

    assert to_english(HINDI) == (HINDI, None)


def test_an_empty_translation_returns_the_question_unchanged(stub):
    stub["reply"] = "   "

    assert to_english(HINDI) == (HINDI, None)


def test_a_preamble_is_stripped_from_the_translation(stub):
    stub["reply"] = f'Translation: "{ENGLISH}"'

    question, _ = to_english(HINDI)

    assert question == ENGLISH


def test_disabling_the_feature_skips_everything(monkeypatch):
    monkeypatch.setattr(translator, "_settings", lambda: {"enabled": False})

    def explode(*args, **kwargs):
        raise AssertionError("must not be called when disabled")

    monkeypatch.setattr(translator, "_call_model", explode)

    assert to_english(HINDI) == (HINDI, None)


def test_always_translate_sends_english_through_the_model_too(monkeypatch, stub):
    monkeypatch.setattr(
        translator,
        "_settings",
        lambda: {"enabled": True, "always_translate": True},
    )
    stub["reply"] = ENGLISH

    question, language = to_english(ENGLISH)

    assert len(stub["calls"]) == 1
    # The model returned the question unchanged, so there is nothing to
    # translate the answer back into.
    assert language is None


def test_the_answer_goes_back_into_the_source_language(stub):
    stub["reply"] = "औसत सतही तापमान 27.4 डिग्री सेल्सियस है"

    out = from_english("The average surface temperature is 27.4 C", "Hindi")

    assert out == "औसत सतही तापमान 27.4 डिग्री सेल्सियस है"


def test_an_english_answer_is_left_alone_when_there_is_no_language(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("must not translate without a source language")

    monkeypatch.setattr(translator, "_call_model", explode)

    assert from_english("27.4 C", None) == "27.4 C"


def test_a_failed_back_translation_returns_the_english_answer(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("model unreachable")

    monkeypatch.setattr(translator, "_call_model", explode)

    assert from_english("27.4 C", "Hindi") == "27.4 C"


def test_the_back_translation_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(
        translator,
        "_settings",
        lambda: {"enabled": True, "answer_in_source_language": False},
    )

    def explode(*args, **kwargs):
        raise AssertionError("must not be called when switched off")

    monkeypatch.setattr(translator, "_call_model", explode)

    assert from_english("27.4 C", "Hindi") == "27.4 C"


def test_the_prompt_tells_the_model_not_to_answer_the_question(stub):
    to_english(HINDI)

    system, _ = stub["calls"][0]

    assert "Do not answer the question" in system
    assert "Leave identifiers alone" in system
