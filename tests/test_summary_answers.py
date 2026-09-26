"""The summaries route, with a fake index and a fake model.

What is pinned down here is the contract: the gate declines before the model
is called, the model's own abstention is honoured, the citations name the
float or region, and an unreachable model degrades to the summaries rather
than to an error.
"""

from __future__ import annotations

import pytest

from src.generation import answer_generator as ag
from src.generation.prompt import SYSTEM_PROMPT, build_context_block, build_messages
from src.utils.schemas import Summary

FLOAT = Summary(
    kind="float",
    subject="1901393",
    text="ARGO float 1901393 is an APEX platform. It recorded 142 profiles.",
    score=0.82,
)

REGION = Summary(
    kind="region",
    subject="Arabian Sea",
    text="The Arabian Sea holds 58 floats and 3,204 profiles in this database.",
    score=0.71,
)


class FakeIndex:
    def __init__(self, hits):
        self.hits = hits
        self.asked = []

    def search(self, question, k=None):
        self.asked.append(question)
        return [Summary(**h.to_dict()) for h in self.hits]


class FakeLLM:
    def __init__(self, text="Float 1901393 recorded 142 profiles [1].", fail=None):
        self.text = text
        self.fail = fail
        self.calls = 0

    def generate(self, system_prompt, user_prompt):
        self.calls += 1

        if self.fail:
            raise self.fail

        from src.generation.llm import LLMResponse

        return LLMResponse(text=self.text, tokens={"total": 10})


@pytest.fixture(autouse=True)
def config(monkeypatch):
    from src.utils.config import AttrDict

    cfg = AttrDict(
        {
            "generation": {"provider": "ollama", "model": "fake", "expose_prompt": True},
            "embeddings": {"model_name": "fake-embedder"},
            "semantic": {
                "answer_threshold": 0.45,
                "confidence": {
                    "weights": {"mean_similarity": 0.6, "support": 0.3, "spread": 0.1},
                    "support_floor": 0.55,
                },
            },
        }
    )

    monkeypatch.setattr(ag, "get_config", lambda: cfg)
    monkeypatch.setattr("src.semantic.confidence.get_config", lambda: cfg)

    return cfg


def _generator(hits, llm=None):
    return ag.AnswerGenerator(FakeIndex(hits), llm=llm or FakeLLM())


# --- the prompt ---------------------------------------------------------------


def test_the_context_block_numbers_each_summary_and_names_its_subject():
    hits = [Summary(**FLOAT.to_dict()), Summary(**REGION.to_dict())]
    hits[0].rank, hits[1].rank = 1, 2

    block = build_context_block(hits)

    assert block.startswith("[1] (float 1901393)\n")
    assert "[2] (region Arabian Sea)\n" in block


def test_the_prompt_forbids_prior_knowledge_and_asks_for_citations():
    system, user = build_messages("how many profiles?", [FLOAT])

    assert system == SYSTEM_PROMPT
    assert "ONLY information in the summaries" in system
    assert "INSUFFICIENT_CONTEXT" in system
    assert "Question: how many profiles?" in user


# --- the gate -----------------------------------------------------------------


def test_an_answer_cites_the_summaries_it_was_written_from():
    llm = FakeLLM()
    answer = _generator([FLOAT, REGION], llm).answer("what did float 1901393 record")

    assert answer.answered
    assert llm.calls == 1
    assert [(s.kind, s.subject, s.rank) for s in answer.sources] == [
        ("float", "1901393", 1),
        ("region", "Arabian Sea", 2),
    ]
    assert answer.prompt and "SYSTEM:" in answer.prompt


def test_nothing_retrieved_declines_without_calling_the_model():
    llm = FakeLLM()
    answer = _generator([], llm).answer("what is the wind speed")

    assert not answer.answered
    assert llm.calls == 0
    assert answer.confidence.score == 0.0
    assert "not something it holds" in answer.reason


def test_weak_retrieval_declines_without_calling_the_model():
    llm = FakeLLM()
    weak = Summary(kind="float", subject="1", text="t", score=0.30)

    answer = _generator([weak], llm).answer("something off topic")

    assert not answer.answered
    assert llm.calls == 0
    assert "below the 45% threshold" in answer.reason
    # The closest summaries still travel with the refusal, so the reader can
    # see what was nearly good enough.
    assert answer.sources[0].subject == "1"


def test_the_models_own_abstention_is_honoured():
    llm = FakeLLM(text="INSUFFICIENT_CONTEXT")

    answer = _generator([FLOAT], llm).answer("what colour is the float")

    assert not answer.answered
    assert answer.answer == ag.DECLINE_MESSAGE
    assert "judged the retrieved summaries insufficient" in answer.reason


def test_an_unreachable_model_returns_the_summaries_rather_than_an_error():
    llm = FakeLLM(fail=ConnectionError("connection refused"))

    answer = _generator([FLOAT, REGION], llm).answer("tell me about float 1901393")

    assert not answer.answered
    assert answer.error == "connection refused"
    assert "float 1901393" in answer.answer
    assert "could not reach the language model" in answer.answer


def test_the_service_dictionary_carries_citations_as_plain_data():
    """The API and the UI read this shape, so it must not leak dataclasses."""
    from src.utils import pipeline

    svc = pipeline.RAGService.__new__(pipeline.RAGService)
    svc.generator = _generator([FLOAT])

    result = svc.answer_from_summaries("what did float 1901393 record")

    assert result["answered"] is True
    assert result["refused"] is False
    assert result["sources"][0] == {
        "kind": "float",
        "subject": "1901393",
        "text": FLOAT.text,
        "score": 0.82,
        "rank": 1,
    }
