"""Summary wording and prompt assembly.

The wording is the part that decides whether retrieval finds the right float,
so it is tested directly rather than through a database.
"""

from datetime import date

from src.semantic.index import IDENTIFIER, Summary, as_context
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


def test_a_core_only_float_is_not_described_as_bgc():
    assert "BGC" not in summarise_float(FLOAT_ROW)
    assert "oxygen" not in summarise_float(FLOAT_ROW)


def test_bgc_sensors_are_named_when_the_float_carries_them():
    bgc = dict(FLOAT_ROW, oxygen_n=5000, chlorophyll_n=4800, ph_n=0)

    text = summarise_float(bgc)

    assert "BGC float" in text
    assert "dissolved oxygen and chlorophyll" in text
    assert "pH" not in text


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
    assert "Compute every number with SQL" in " ".join(prompt.split())


def test_prompt_forbids_turning_context_into_filters():
    # The model once answered "how many measurements in the Southern Indian
    # Ocean" with project = 'INCOIS' AND platform = 'APEX' bolted on, because
    # those words appear in every retrieved summary. Describing the database
    # is not the same as describing the question.
    prompt = build_prompt("how many measurements?", context="- A note.")

    assert "WHERE condition" in prompt
    assert "does not mean the question is about" in prompt


def test_prompt_forbids_inventing_a_date_range_from_context():
    prompt = build_prompt("profiles per year", context="- A note.")

    assert "If the question names no period, do not bound one." in " ".join(
        prompt.split()
    )


def test_question_lands_after_the_context():
    prompt = build_prompt("how many floats?", context="- Something.")

    assert prompt.index("Something.") < prompt.index("how many floats?")


def test_different_context_cannot_share_a_cache_entry():
    assert cache_version("- A.") != cache_version("- B.")


def test_version_is_stable_for_an_unchanged_prompt():
    from src.sqlgen.generator import PROMPT_VERSION

    assert cache_version("").startswith(PROMPT_VERSION)
    assert cache_version("") == cache_version("   ")


def test_a_changed_schema_catalog_invalidates_the_cache(monkeypatch):
    # The catalog is part of the prompt. Leaving it out of the key once meant
    # adding the BGC columns changed what the model was told while every
    # cached entry still looked valid.
    import src.sqlgen.generator as generator

    before = cache_version("")

    monkeypatch.setattr(
        generator,
        "build_context",
        lambda include_examples=True: "a different schema",
    )

    assert cache_version("") != before


def test_float_identifiers_are_picked_out_of_a_question():
    assert IDENTIFIER.findall("what did float 2900007 measure") == ["2900007"]
    assert IDENTIFIER.findall("compare 2900007 and 2900012") == [
        "2900007",
        "2900012",
    ]


def test_years_and_depths_are_not_mistaken_for_identifiers():
    # Four digits is a year or a pressure, not a WMO number. Treating one as a
    # float id would pull an unrelated summary into the prompt.
    assert IDENTIFIER.findall("mean temperature in 2020 below 1000 dbar") == []


def test_refresh_removes_summaries_of_subjects_that_are_gone(monkeypatch):
    """The upsert on its own never deleted anything. When the synthetic floats
    were replaced by real ones, forty summaries of floats that no longer existed
    stayed in the index and were retrieved as context."""
    import contextlib

    import src.semantic.index as index

    executed = []

    class FakeCursor:
        rowcount = 40

        def executemany(self, sql, rows):
            executed.append(("upsert", sql, list(rows)))

        def execute(self, sql, params):
            executed.append(("prune", sql, params))

    @contextlib.contextmanager
    def fake_cursor(*args, **kwargs):
        yield FakeCursor()

    class FakeEmbedder:
        def embed_passages(self, texts):
            return [[0.0] * 3 for _ in texts]

    monkeypatch.setattr(
        index,
        "collect",
        lambda: [("float", "1901393", "a real float"), ("region", "Arabian Sea", "a region")],
    )
    monkeypatch.setattr(index, "cursor", fake_cursor)
    monkeypatch.setattr(index, "get_embedding_model", lambda: FakeEmbedder())
    monkeypatch.setattr(index, "get_config", lambda: {"semantic": {}})

    result = index.SemanticIndex(table="data_summaries").refresh()

    prune = [step for step in executed if step[0] == "prune"]

    assert len(prune) == 1
    assert "DELETE FROM data_summaries" in prune[0][1]
    # Everything just rebuilt is kept; anything else is removed.
    assert prune[0][2] == (["float", "region"], ["1901393", "Arabian Sea"])
    assert result["removed"] == 40
