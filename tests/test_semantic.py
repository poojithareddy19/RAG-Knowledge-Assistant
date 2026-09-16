"""Summary wording and prompt assembly.

The wording is the part that decides whether retrieval finds the right float,
so it is tested directly rather than through a database.
"""

from datetime import date

from src.semantic.index import Summary, as_context
from src.semantic.summaries import summarise_float, summarise_region
from src.sqlgen.generator import build_prompt, cache_version

FLOAT_ROW = {
    "float_id": 1901393,
    "platform": "APEX",
    "project": "ARGO INDIA",
    "profiles": 142,
    "first_obs": date(2015, 3, 2),
    "last_obs": date(2021, 11, 18),
    "regions": ["Arabian Sea", "Bay of Bengal"],
    "min_pressure": 4.0,
    "max_pressure": 1998.0,
    "mean_surface_temp": 28.41,
    "mean_surface_salinity": 36.12,
}

REGION_ROW = {
    "region": "Arabian Sea",
    "floats": 58,
    "profiles": 3204,
    "first_obs": date(2005, 1, 1),
    "last_obs": date(2023, 6, 30),
    "mean_surface_temp": 28.4,
    "mean_surface_salinity": 36.1,
}


def test_float_summary_names_the_things_a_question_would_name():
    text = summarise_float(FLOAT_ROW)

    assert "1901393" in text
    assert "an APEX platform" in text
    assert "ARGO INDIA" in text
    assert "Arabian Sea and Bay of Bengal" in text
    assert "2015-03-02" in text
    assert "2021-11-18" in text


def test_float_summary_survives_a_float_with_no_measurements():
    sparse = {
        "float_id": 7,
        "platform": None,
        "project": None,
        "profiles": 1,
        "first_obs": date(2020, 1, 1),
        "last_obs": date(2020, 1, 1),
        "regions": None,
        "min_pressure": None,
        "max_pressure": None,
        "mean_surface_temp": None,
        "mean_surface_salinity": None,
    }

    text = summarise_float(sparse)

    assert "ARGO float 7." in text
    assert "1 profile on 2020-01-01" in text


def test_unknown_metadata_is_left_out():
    # The CSV loader writes "unknown". Repeating it would only teach the
    # retriever that every float is unknown.
    row = dict(FLOAT_ROW, platform="unknown", project="unknown")

    assert "unknown" not in summarise_float(row).lower()


def test_counts_are_pluralised_and_grouped():
    assert "3,204 profiles" in summarise_region(REGION_ROW)
    assert "58 floats" in summarise_region(REGION_ROW)


def test_region_summary_reports_coverage_and_surface_values():
    text = summarise_region(REGION_ROW)

    assert "Arabian Sea" in text
    assert "from 2005-01-01 to 2023-06-30" in text
    assert "28.4 degrees Celsius" in text


def test_context_is_rendered_as_a_bullet_list():
    summaries = [
        Summary("float", "1901393", "Float A.", 0.9),
        Summary("region", "Arabian Sea", "Region B.", 0.7),
    ]

    assert as_context(summaries) == "- Float A.\n- Region B."


def test_prompt_omits_the_context_block_when_there_is_none():
    prompt = build_prompt("how many floats?", context="")

    assert "WHAT IS IN THE DATABASE" not in prompt
    assert "=== SCHEMA ===" in prompt


def test_prompt_includes_context_and_the_rule_against_quoting_it():
    prompt = build_prompt(
        "how many floats?",
        context="- ARGO float 1901393 is an APEX platform.",
    )

    assert "WHAT IS IN THE DATABASE" in prompt
    assert "1901393" in prompt
    assert "compute every number with SQL" in prompt


def test_question_lands_after_the_context():
    prompt = build_prompt("how many floats?", context="- Something.")

    assert prompt.index("Something.") < prompt.index("how many floats?")


def test_different_context_cannot_share_a_cache_entry():
    assert cache_version("- A.") != cache_version("- B.")


def test_no_context_keeps_the_bare_prompt_version():
    from src.sqlgen.generator import PROMPT_VERSION

    assert cache_version("") == PROMPT_VERSION
    assert cache_version("   ") == PROMPT_VERSION
