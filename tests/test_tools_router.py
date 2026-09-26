"""Which questions go to an MCP tool, and with what arguments.

Written against the problem statement's own examples, because those are the
questions an assessor will type first.
"""

import pytest

from src.router.tools import NEEDS_LOCATION, coordinates, match_tool, with_location


@pytest.mark.parametrize(
    "question, expected",
    [
        ("What are the nearest ARGO floats to 10.5N 65.2E?", (10.5, 65.2)),
        ("closest floats to 5S 80E", (-5.0, 80.0)),
        ("nearest floats to 12.5°N, 70.25°E", (12.5, 70.25)),
        ("nearest floats to 3 N 45 W", (3.0, -45.0)),
        ("nearest floats to lat 12.3 lon 75.1", (12.3, 75.1)),
        ("nearest floats to latitude: -20 longitude = 60.5", (-20.0, 60.5)),
    ],
)
def test_positions_are_read_in_every_common_notation(question, expected):
    assert coordinates(question) == expected


def test_south_and_west_are_negative():
    assert coordinates("floats closest to 12S 40W") == (-12.0, -40.0)


def test_an_impossible_position_is_not_a_position():
    assert coordinates("nearest floats to 95N 65E") is None
    assert coordinates("nearest floats to lat 10 lon 200") is None


def test_the_problem_statements_third_example_goes_to_the_mcp_tool():
    tool, arguments = match_tool("What are the nearest ARGO floats to 10.5N 65.2E?")

    assert tool == "nearest_floats"
    assert arguments == {"latitude": 10.5, "longitude": 65.2}


def test_a_bare_pair_is_accepted_once_the_question_asks_for_the_nearest():
    assert match_tool("Which floats are closest to (-20.5, 70.0)?") == (
        "nearest_floats",
        {"latitude": -20.5, "longitude": 70.0},
    )


def test_asking_for_the_nearest_without_saying_where_asks_for_a_location():
    """Better than guessing where "here" is."""
    assert match_tool("What are the nearest ARGO floats to this location?") == (
        NEEDS_LOCATION,
        {},
    )


@pytest.mark.parametrize(
    "question",
    [
        # A date the tool's fixed shape cannot honour.
        "nearest floats to 10N 65E in 2020",
        "nearest floats to 10N 65E since March",
        # A measured quantity or a statistic.
        "average salinity of the floats nearest 10N 65E",
        # The other two problem statement examples belong to the SQL path.
        "Show me salinity profiles near the equator in March 2023",
        "Compare BGC parameters in the Arabian Sea for the last 6 months",
        # Ordinary questions.
        "How many floats are in the database?",
        "Show the track of float 1900083",
        "What is the average salinity in the profiles of float 1900083?",
    ],
)
def test_anything_the_tool_would_answer_only_partly_is_left_to_sql(question):
    """Declining is the safe direction: the SQL path can honour the extra
    condition, and a tool returning the right floats for the wrong year is a
    wrong answer delivered confidently."""
    assert match_tool(question) is None


def test_a_request_for_one_floats_profile_uses_the_fixed_lookup():
    assert match_tool("Show the profile of float 1900083") == (
        "get_profile",
        {"float_id": "1900083"},
    )


def test_a_cycle_number_is_passed_through():
    assert match_tool("plot profiles for float 1901393 cycle 12") == (
        "get_profile",
        {"float_id": "1901393", "cycle": 12},
    )


def test_the_panel_position_fills_in_this_location():
    sent = with_location("What are the nearest ARGO floats to this location?", (10.5, 65.25))

    assert sent.endswith("(lat 10.5000 lon 65.2500)")
    assert match_tool(sent) == ("nearest_floats", {"latitude": 10.5, "longitude": 65.25})


def test_a_position_typed_in_the_question_beats_the_panel():
    question = "nearest floats to 5S 80E"

    assert with_location(question, (10.0, 65.0)) == question


def test_the_panel_is_ignored_when_switched_off():
    question = "What are the nearest ARGO floats to this location?"

    assert with_location(question, None) == question


def test_a_question_that_does_not_mention_a_place_is_sent_as_typed():
    """A position set for one question must not narrow the next."""
    question = "How many floats are in the database?"

    assert with_location(question, (10.0, 65.0)) == question


def test_a_result_that_filled_its_limit_is_recognised():
    from src.utils.pipeline import _hit_limit

    assert _hit_limit("SELECT 1 FROM measurements LIMIT 500", 500)
    assert not _hit_limit("SELECT 1 FROM measurements LIMIT 500", 499)
    assert not _hit_limit("SELECT 1 FROM measurements", 500)
