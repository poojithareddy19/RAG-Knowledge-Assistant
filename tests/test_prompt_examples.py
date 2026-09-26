"""Which questions are shown which worked examples.

Every example is a pattern the model can match. The drifting buoy examples
name `drifter_observations`, a table most questions must not go near, so they
are shown only to questions that could plausibly be about the buoys.
"""

from src.sqlgen.generator import (
    BGC_EXAMPLE,
    DRIFTER_EXAMPLE,
    PROFILE_EXAMPLE,
    bgc_example_applies,
    build_prompt,
    cache_version,
    drifter_example_applies,
    profile_example_applies,
)

MARK = "=== DRIFTING BUOY EXAMPLE ==="


def test_buoy_questions_get_the_examples():
    for q in [
        "Average sea surface temperature from the drifting buoys",
        "Average surface current speed in each region",
        "How many buoys reported in the Bay of Bengal?",
        "What is the northward velocity of the drifters?",
    ]:
        assert drifter_example_applies(q), q
        assert MARK in build_prompt(q), q


def test_argo_questions_do_not():
    for q in [
        "How many floats are in the database?",
        "What is the average salinity below 1000 decibars?",
        "Average surface temperature per year per region",
        "How many measurements failed quality control?",
    ]:
        assert not drifter_example_applies(q), q
        assert MARK not in build_prompt(q), q


def test_a_harmless_false_positive_is_allowed():
    """"current year" is not an ocean current, and the cost of being wrong
    here is one longer prompt rather than a wrong table."""
    assert drifter_example_applies("How many profiles in the current year?")


def test_the_other_examples_are_shown_to_everyone():
    prompt = build_prompt("How many floats are in the database?")

    assert "=== EXAMPLE ===" in prompt
    assert "=== WINDOW FUNCTION EXAMPLES ===" in prompt


def test_editing_the_buoy_examples_still_invalidates_the_cache():
    """The digest is taken with an empty question, which is shown no buoy
    example, so they have to be folded in explicitly or an edit to them would
    silently keep serving SQL written before it."""
    import src.sqlgen.generator as g

    before = cache_version()
    original = g.DRIFTER_EXAMPLE
    try:
        g.DRIFTER_EXAMPLE = original + "\n-- a change\n"
        assert cache_version() != before
    finally:
        g.DRIFTER_EXAMPLE = original

    assert cache_version() == before


def test_examples_can_still_be_turned_off_wholesale():
    prompt = build_prompt("Average surface current speed", include_examples=False)

    assert MARK not in prompt
    assert DRIFTER_EXAMPLE not in prompt


# --- vertical profiles ------------------------------------------------------

PROFILE_MARK = "=== VERTICAL PROFILE EXAMPLE ==="


def test_a_request_for_profiles_of_a_quantity_gets_the_profile_example():
    question = "Show me salinity profiles near the equator in March 2023"

    assert profile_example_applies(question)
    assert PROFILE_MARK in build_prompt(question)


def test_counting_profiles_is_not_asking_for_their_levels():
    """"Profiles per year" is in the gold set and is a count. Teaching it to
    return one row per depth level would break it."""
    for question in (
        "Profiles per year with a running cumulative total",
        "How many profiles were recorded in 2003?",
        "Number of profiles per region",
    ):
        assert not profile_example_applies(question), question
        assert PROFILE_MARK not in build_prompt(question), question


def test_the_profile_example_returns_the_cast_key_the_chart_groups_on():
    """Without profile_id the chart can only draw one line through every cast."""
    assert "p.profile_id" in PROFILE_EXAMPLE
    assert "ORDER BY p.profile_id, m.pressure_dbar" in PROFILE_EXAMPLE


# --- BGC ----------------------------------------------------------------------


def test_naming_bgc_parameters_gets_the_bgc_example():
    """The problem statement's second example. Told in prose what BGC meant,
    the model compared temperature and salinity instead."""
    question = "Compare BGC parameters in the Arabian Sea for the last 6 months"

    assert bgc_example_applies(question)
    assert "=== BGC EXAMPLE ===" in build_prompt(question)


def test_single_bgc_parameter_questions_are_not_shown_the_group_example():
    """The gold set's bgc bucket asks about one parameter at a time and never
    says "BGC", so the benchmark does not move."""
    for question in (
        "What is the average pH of the water?",
        "What is the average dissolved oxygen at 500 decibars?",
    ):
        assert not bgc_example_applies(question), question


def test_each_bgc_parameter_is_averaged_under_its_own_qc_flag():
    for flag in ("oxygen_qc", "chlorophyll_qc", "nitrate_qc", "ph_qc", "backscatter_qc"):
        assert f"m.{flag} = 1" in BGC_EXAMPLE


def _example_sql(example):
    return example.split("SQL:", 1)[1].strip()


def test_every_gated_example_passes_the_validator():
    """An example the validator would reject teaches the model to write SQL
    that is then refused."""
    from src.sqlgen.validator import validate

    tables = ["floats", "profiles", "measurements", "drifters", "drifter_observations"]

    for example in (PROFILE_EXAMPLE, BGC_EXAMPLE):
        assert validate(_example_sql(example), tables)


def test_the_profile_example_asks_for_enough_rows_for_high_resolution_casts():
    """A high-resolution float reports about a thousand levels a cast. At the
    default limit of 500 one cast was cut in half and drawn as complete."""
    assert "LIMIT 5000" in PROFILE_EXAMPLE
