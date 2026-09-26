"""What period the archive covers, as the SQL prompt is told it.

The problem statement asks for "the last 6 months". Before this the model had
no idea the archive was a snapshot, wrote now() - interval '6 months', and got a
window starting after the last profile.
"""

import datetime as dt

import src.sqlgen.schema_context as schema_context
from src.sqlgen.schema_context import coverage_note, data_coverage

SPANS = [
    ("Argo profiles", dt.date(2001, 7, 24), dt.date(2025, 10, 5)),
    ("Drifting buoy fixes", dt.date(2023, 1, 1), dt.date(2024, 1, 1)),
]


def test_both_platforms_are_stated_with_their_own_range():
    note = coverage_note(SPANS)

    assert "2001-07-24 to 2025-10-05" in note
    assert "2023-01-01 to 2024-01-01" in note


def test_relative_periods_are_anchored_to_the_data_not_the_clock():
    note = coverage_note(SPANS)

    assert "not relative to today" in note
    assert "never now() or current_date" in note


def test_the_worked_window_is_six_months_back_from_the_latest_date():
    """A concrete window the model can copy, computed rather than described."""
    note = coverage_note(SPANS)

    assert "obs_time >= '2025-04-05' AND obs_time <= '2025-10-05'" in note


def test_no_spans_means_no_section():
    assert coverage_note([]) == ""


def test_an_unreachable_database_degrades_to_nothing_and_is_not_retried(monkeypatch):
    """The prompt is built in CI and in these tests with no database. That must
    cost one fast failure, not a thirty second pool wait on every prompt."""
    calls = {"n": 0}

    def down():
        calls["n"] += 1
        raise ConnectionError("no database")

    monkeypatch.setattr(schema_context, "_read_coverage", down)
    monkeypatch.setattr(schema_context, "_COVERAGE", {"text": None, "at": 0.0})

    assert data_coverage() == ""
    assert data_coverage() == ""
    assert calls["n"] == 1


def test_the_note_reaches_the_sql_prompt(monkeypatch):
    import src.sqlgen.generator as generator

    monkeypatch.setattr(generator, "data_coverage", lambda: coverage_note(SPANS))

    assert "WHAT PERIOD THE DATABASE COVERS" in generator.build_prompt(
        "Compare BGC parameters in the Arabian Sea for the last 6 months"
    )
