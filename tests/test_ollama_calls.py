"""What every call to Ollama sends.

Two settings have to travel with every request, from every call site, with the
same value. ``num_ctx``: Ollama cuts a prompt longer than its window instead of
refusing it, and reloads the model when the window changes between calls.
``keep_alive``: without it the model is unloaded five minutes after the last
call and the next question pays to load it again. One call site that forgets
either undoes it for everyone, so each is checked here.
"""

from __future__ import annotations

import pytest

from src.generation import llm


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


@pytest.fixture
def sent(monkeypatch):
    """Every JSON body posted to Ollama, whichever HTTP client posted it."""
    bodies = []

    def post(url, json=None, timeout=None, **_):
        bodies.append(json)
        return FakeResponse(
            {
                "response": '{"route": "data", "reason": "x"}',
                "message": {"content": "an answer"},
            }
        )

    import httpx
    import requests

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(llm, "num_ctx", lambda: 6144)
    monkeypatch.setattr(llm, "keep_alive", lambda: "30m")

    for module in ("src.router.classifier", "src.router.rewriter", "src.sqlgen.generator"):
        mod = __import__(module, fromlist=["x"])
        monkeypatch.setattr(mod, "num_ctx", lambda: 6144)
        monkeypatch.setattr(mod, "keep_alive", lambda: "30m")

    return bodies


def _check(body):
    assert body["keep_alive"] == "30m"
    assert body["options"]["num_ctx"] == 6144


def test_the_router_call(sent):
    from src.router.classifier import model_route

    model_route("is the water warmer near Goa")

    _check(sent[-1])


def test_the_rewrite_call(sent):
    from src.router.rewriter import _call_model

    _call_model("a prompt")

    _check(sent[-1])


def test_the_sql_call(sent):
    from src.sqlgen.generator import _complete

    _complete("a prompt", "llama3.1:latest", 30, 400)

    _check(sent[-1])


def test_the_answer_call(sent, monkeypatch):
    from src.utils.config import AttrDict

    monkeypatch.setattr(
        llm,
        "get_config",
        lambda: AttrDict(
            {
                "generation": {"model": "llama3.1:latest", "temperature": 0.0},
                "secrets": {"ollama_base_url": "http://localhost:11434"},
            }
        ),
    )

    llm.OllamaLLM().generate("system", "user")

    _check(sent[-1])


def test_warm_up_loads_each_model_at_the_real_window(sent, monkeypatch):
    """A load at a different window would be thrown away by the first question."""
    from src.utils.config import AttrDict

    monkeypatch.setattr(
        llm,
        "get_config",
        lambda: AttrDict({"secrets": {"ollama_base_url": "http://localhost:11434"}}),
    )

    took = llm.warm_up(["llama3.1:latest", "qwen2.5-coder:7b"])

    assert [b["model"] for b in sent] == ["llama3.1:latest", "qwen2.5-coder:7b"]
    assert all("prompt" not in b for b in sent)
    for body in sent:
        _check(body)
    assert set(took) == {"llama3.1:latest", "qwen2.5-coder:7b"}


def test_keep_alive_reads_the_config(monkeypatch):
    from src.utils.config import AttrDict

    monkeypatch.setattr(llm, "get_config", lambda: AttrDict({"ollama": {"keep_alive": "2h"}}))
    assert llm.keep_alive() == "2h"

    monkeypatch.setattr(llm, "get_config", lambda: AttrDict({"ollama": {"keep_alive": -1}}))
    assert llm.keep_alive() == -1


def test_the_answer_call_caps_its_length(sent, monkeypatch):
    """generation.max_tokens was read and never sent, so answers had no cap."""
    from src.utils.config import AttrDict

    monkeypatch.setattr(
        llm,
        "get_config",
        lambda: AttrDict(
            {
                "generation": {"model": "m", "temperature": 0.0, "max_tokens": 321},
                "secrets": {"ollama_base_url": "http://localhost:11434"},
            }
        ),
    )

    llm.OllamaLLM().generate("system", "user")

    assert sent[-1]["options"]["num_predict"] == 321


def test_each_role_reads_its_model_from_config(monkeypatch):
    """router.model was read by the warm-up and ignored by the router."""
    from src.utils.config import AttrDict

    monkeypatch.setattr(
        llm,
        "get_config",
        lambda: AttrDict(
            {
                "generation": {"model": "gen-model"},
                "router": {"model": "router-model"},
                "sql": {},
                "secrets": {"ollama_base_url": "http://ollama:11434/"},
            }
        ),
    )

    assert llm.model_for("router") == "router-model"
    assert llm.model_for("generation") == "gen-model"
    # No model of its own: the generation model.
    assert llm.model_for("sql") == "gen-model"
    assert llm.ollama_base_url() == "http://ollama:11434"


def test_the_router_call_uses_the_router_model(sent, monkeypatch):
    import src.router.classifier as classifier

    monkeypatch.setattr(classifier, "model_for", lambda role: f"{role}-model")
    monkeypatch.setattr(classifier, "ollama_base_url", lambda: "http://x")

    classifier.model_route("is the water warmer near Goa")

    assert sent[-1]["model"] == "router-model"
