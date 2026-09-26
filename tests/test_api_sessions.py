"""The API's in-process conversation memory.

No client and no service here: importing the module is enough, because the
service is built lazily on the first request. What is worth pinning down is the
capping rule, since an uncapped history grows the rewrite prompt on every turn.
"""

from __future__ import annotations

import pytest

from src.api import main


@pytest.fixture(autouse=True)
def clean_history():
    main._history.clear()
    yield
    main._history.clear()


def test_a_session_remembers_its_exchanges():
    main.remember("s1", "average temperature in the Arabian Sea", "27.4 C")

    assert main._history["s1"] == [
        ("average temperature in the Arabian Sea", "27.4 C")
    ]


def test_sessions_do_not_see_each_other():
    main.remember("s1", "question one", "answer one")
    main.remember("s2", "question two", "answer two")

    assert len(main._history["s1"]) == 1
    assert main._history["s1"][0][0] == "question one"


def test_history_is_capped_at_max_turns(monkeypatch):
    monkeypatch.setattr(main, "_max_turns", lambda: 2)

    for n in range(1, 6):
        main.remember("s1", f"question {n}", f"answer {n}")

    assert [asked for asked, _ in main._history["s1"]] == [
        "question 4",
        "question 5",
    ]


def test_the_request_schema_accepts_a_session_id():
    request = main.AskRequest(question="and in 2022?", session_id="abc")

    assert request.session_id == "abc"


def test_a_session_id_is_optional():
    assert main.AskRequest(question="how many floats are there").session_id is None


def test_the_generic_chart_reaches_the_client_as_base64(monkeypatch):
    """The pipeline holds PNG bytes, which JSON cannot carry. The React page
    draws the generic chart from this field, so it must survive the trip."""
    import base64

    class FakeService:
        def answer_from_data(self, question, want_chart=False):
            return {
                "answer": "3 rows",
                "confidence": 1.0,
                "chart_png": b"\x89PNG fake bytes",
                "chart_kind": "bar",
            }

    monkeypatch.setattr(main, "service", lambda: FakeService())

    response = main.ask(main.AskRequest(question="bar chart of profiles per region",
                                        route_override="chart"))

    assert base64.b64decode(response.chart_png_base64) == b"\x89PNG fake bytes"
    assert response.chart_kind == "bar"


def test_an_override_request_is_logged(monkeypatch):
    """The override paths bypass RAGService.answer, which is where logging
    happens, so they used to leave no line in interactions.jsonl."""
    logged = []

    class FakeService:
        def answer_from_summaries(self, question):
            return {"answer": "ok", "confidence": 0.9, "sources": []}

    monkeypatch.setattr(main, "service", lambda: FakeService())
    monkeypatch.setattr(main, "log_result", lambda q, r: logged.append((q, r)))

    response = main.ask(main.AskRequest(question="tell me about float 1900083",
                                        route_override="summaries"))

    assert response.route_decided_by == "override"
    assert len(logged) == 1
    question, result = logged[0]
    assert question == "tell me about float 1900083"
    assert result["route"] == "summaries"
    assert result["elapsed_ms"] is not None


def test_the_number_of_sessions_is_capped(monkeypatch):
    monkeypatch.setattr(main, "MAX_SESSIONS", 3)

    for n in range(5):
        main.remember(f"s{n}", "q", "a")

    assert list(main._history) == ["s2", "s3", "s4"]


def test_a_returning_session_is_kept_over_older_ones(monkeypatch):
    monkeypatch.setattr(main, "MAX_SESSIONS", 2)

    main.remember("old", "q", "a")
    main.remember("new", "q", "a")
    main.remember("old", "q2", "a2")
    main.remember("newest", "q", "a")

    assert list(main._history) == ["old", "newest"]
