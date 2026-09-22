"""Which questions are shown which worked examples.

Every example is a pattern the model can match. The drifting buoy examples
name `drifter_observations`, a table most questions must not go near, so they
are shown only to questions that could plausibly be about the buoys.
"""

from src.sqlgen.generator import (
    DRIFTER_EXAMPLE,
    build_prompt,
    cache_version,
    drifter_example_applies,
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
