import pytest

from src.sqlgen.scope import error_is_the_answer, out_of_scope


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


# --- when a failure is the answer ------------------------------------------

CATALOG = {
    "measurements": {"pressure_dbar", "temperature_c", "profile_id"},
    "profiles": {"profile_id", "float_id", "region", "obs_time"},
    "floats": {"float_id", "platform"},
    "drifters": {"buoy_id", "buoy_type"},
    "drifter_observations": {"buoy_id", "sst_c", "region", "obs_time"},
}

FAMILIES = {
    "argo": frozenset({"floats", "profiles", "measurements"}),
    "drifter": frozenset({"drifters", "drifter_observations"}),
}


def test_a_column_the_platform_does_not_have_is_not_repaired():
    """The failure that started this: a buoy has no depth.

    Left to repair, the model substituted sea surface temperature and kept the
    aliases min_pressure and max_pressure, answering "how deep did each buoy
    dive" with three rows of temperatures.
    """
    reason = error_is_the_answer(
        "SELECT MIN(o.pressure_dbar) FROM drifter_observations o",
        "UndefinedColumn: column o.pressure_dbar does not exist",
        CATALOG,
        FAMILIES,
    )

    assert reason is not None
    assert "pressure_dbar" in reason


def test_a_wrong_alias_on_a_real_column_is_still_repaired():
    """pressure_dbar exists on the Argo tables; the query reached it through
    the wrong alias. That is a mistake, and repairing it is the point."""
    assert (
        error_is_the_answer(
            "SELECT p.pressure_dbar FROM measurements m JOIN profiles p ON p.profile_id = m.profile_id",
            "UndefinedColumn: column p.pressure_dbar does not exist",
            CATALOG,
            FAMILIES,
        )
        is None
    )


def test_an_unqualified_column_is_judged_the_same_way():
    assert (
        error_is_the_answer(
            "SELECT COUNT(DISTINCT float_id) FROM measurements m",
            'UndefinedColumn: column "float_id" does not exist',
            CATALOG,
            FAMILIES,
        )
        is None
    )


def test_a_quantity_on_no_platform_is_not_repaired():
    reason = error_is_the_answer(
        "SELECT o.wind_speed FROM drifter_observations o",
        "UndefinedColumn: column o.wind_speed does not exist",
        CATALOG,
        FAMILIES,
    )

    assert reason is not None


def test_an_error_that_is_not_about_a_column_is_left_alone():
    """Syntax errors and missing FROM entries are ordinary mistakes."""
    assert (
        error_is_the_answer(
            "SELECT 1 FROM measurements m",
            'UndefinedTable: missing FROM-clause entry for table "p"',
            CATALOG,
            FAMILIES,
        )
        is None
    )


def test_without_a_catalog_the_guard_does_not_guess():
    """No catalog means the check cannot be made, and the repair proceeds as
    it did before the guard existed."""
    assert (
        error_is_the_answer(
            "SELECT MIN(o.pressure_dbar) FROM drifter_observations o",
            "UndefinedColumn: column o.pressure_dbar does not exist",
            {},
            FAMILIES,
        )
        is None
    )
