import pytest

from src.sqlgen.scope import out_of_scope


@pytest.mark.parametrize(
    "question",
    [
        "What is the wind speed over the Arabian Sea?",
        "What is the average rainfall in Chennai?",
        "What is the seafloor depth beneath each float?",
        "How deep is the ocean under float 1900083?",
        "What is the significant wave height in the Bay of Bengal?",
        "How many whales were observed by the floats?",
        "Which floats detected plastic pollution?",
    ],
)
def test_quantity_with_no_column_is_refused(question):
    assert out_of_scope(question) is not None


@pytest.mark.parametrize(
    "question",
    [
        "What is the current speed at 1000 decibars?",
        "What is the current speed at 500 m?",
        "How fast are the currents at depth?",
        "What is the velocity of the current below the thermocline?",
        "Show me deep ocean currents in the Arabian Sea",
    ],
)
def test_current_below_the_surface_is_refused(question):
    reason = out_of_scope(question)

    assert reason is not None
    assert "surface current" in reason


@pytest.mark.parametrize(
    "question",
    [
        "What is the average surface current speed?",
        "Average surface current speed in each region",
        "What is the current speed measured by the drifting buoys?",
        "Which region has the fastest surface currents?",
    ],
)
def test_surface_current_is_still_answerable(question):
    """The gate closes a hole, it does not close the feature.

    The buoy velocities are real columns holding a real quantity. Refusing
    every question that says "current" would trade a hallucination for a
    false refusal, which the same metric counts against us.
    """
    assert out_of_scope(question) is None


@pytest.mark.parametrize(
    "question",
    [
        # "current" as the ordinary English word, nowhere near the ocean sense
        "How many profiles were recorded in the current year?",
        "Which floats are currently reporting?",
        # depth words on a quantity the database holds all the way down
        "What is the average salinity below 1000 decibars?",
        "How many measurements are deeper than 1000 decibars?",
        "What is the average temperature at each pressure level?",
        "What is the deepest pressure recorded?",
        # the ordinary answerable set
        "How many floats are in the database?",
        "What is the average surface temperature in the Arabian Sea?",
        "Average sea surface temperature from the drifting buoys",
    ],
)
def test_answerable_questions_pass_through(question):
    assert out_of_scope(question) is None


def test_empty_question_is_not_refused_here():
    """An empty question is a different failure and belongs to a different
    layer. This gate only knows about quantities."""
    assert out_of_scope("") is None
    assert out_of_scope("   ") is None
