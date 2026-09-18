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
