from pathlib import Path

import pytest

from src.sqlgen.validator import SQLRejected, validate

ALLOWED = ["floats", "profiles", "measurements"]

QUERY_DIR = Path(__file__).resolve().parents[1] / "db" / "queries"


def v(sql):
    return validate(sql, ALLOWED)


def test_simple_select_passes_and_gets_a_limit():
    out = v("SELECT count(*) FROM measurements")

    assert out.lower().startswith("select")
    assert "limit 500" in out.lower()


def test_join_across_allowed_tables_is_fine():
    out = v(
        """SELECT p.region, avg(m.temperature_c)
FROM measurements m
JOIN profiles p
    ON p.profile_id = m.profile_id
GROUP BY p.region"""
    )

    assert "limit" in out.lower()


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE measurements",
        "DELETE FROM profiles WHERE 1=1",
        "UPDATE floats SET project = 'x'",
        "INSERT INTO floats (float_id) VALUES (1)",
        "TRUNCATE measurements",
    ],
)
def test_write_statements_are_rejected(sql):
    with pytest.raises(SQLRejected):
        v(sql)


def test_statement_chaining_is_rejected():
    with pytest.raises(SQLRejected):
        v("SELECT 1 FROM floats; DROP TABLE floats")


def test_unknown_table_is_rejected():
    with pytest.raises(SQLRejected):
        v("SELECT * FROM pg_shadow")


def test_dangerous_function_is_rejected():
    with pytest.raises(SQLRejected):
        v("SELECT pg_sleep(60) FROM floats")


def test_unanswerable_is_rejected_cleanly():
    with pytest.raises(SQLRejected):
        v("UNANSWERABLE")


def test_oversized_limit_is_capped():
    out = v("SELECT * FROM measurements LIMIT 999999")

    assert "limit 5000" in out.lower()

CATALOG = {
    "floats": frozenset({"float_id", "platform", "project", "first_seen", "last_seen"}),
    "profiles": frozenset(
        {"profile_id", "float_id", "cycle_number", "obs_time", "latitude", "longitude", "region"}
    ),
    "measurements": frozenset(
        {"measurement_id", "profile_id", "pressure_dbar", "temperature_c", "salinity_psu", "qc_flag"}
    ),
}


def vc(sql):
    return validate(sql, ALLOWED, column_catalog=CATALOG)


def test_column_from_the_wrong_table_is_rejected():
    # qc_flag is on measurements, not floats. This is what the model actually
    # produced for "how many floats reported in the Bay of Bengal".
    with pytest.raises(SQLRejected, match="qc_flag"):
        vc(
            """SELECT count(DISTINCT f.float_id)
FROM profiles p
JOIN floats f ON p.float_id = f.float_id
WHERE p.region = 'Bay of Bengal' AND f.qc_flag = 1"""
        )


def test_valid_qualified_columns_pass():
    out = vc(
        """SELECT date_trunc('year', p.obs_time) AS year, avg(m.temperature_c)
FROM measurements m
JOIN profiles p ON p.profile_id = m.profile_id
WHERE p.region = 'Arabian Sea' AND m.qc_flag = 1
GROUP BY year
ORDER BY year"""
    )

    assert "limit 500" in out.lower()


def test_table_name_used_as_its_own_qualifier_is_checked():
    with pytest.raises(SQLRejected, match="region"):
        vc("SELECT measurements.region FROM measurements")


def test_alias_free_join_does_not_confuse_the_alias_map():
    out = vc(
        """SELECT profiles.region, count(*)
FROM profiles
JOIN floats ON floats.float_id = profiles.float_id
GROUP BY profiles.region"""
    )

    assert "limit" in out.lower()


def test_unknown_qualifier_is_left_alone():
    # A CTE name resolves to nothing in the catalog, so it must not be judged.
    out = vc(
        """SELECT recent.region
FROM profiles
WHERE profiles.region = recent.region"""
    )

    assert "limit" in out.lower()


def test_without_a_catalog_columns_are_not_checked():
    out = v("SELECT f.qc_flag FROM floats f")

    assert "limit" in out.lower()


def test_cte_name_is_not_treated_as_an_unknown_table():
    out = v(
        """WITH yearly AS (
    SELECT date_trunc('year', p.obs_time) AS yr, avg(m.temperature_c) AS t
    FROM measurements m
    JOIN profiles p ON p.profile_id = m.profile_id
    GROUP BY yr
)
SELECT yr, t - lag(t) OVER (ORDER BY yr) AS change
FROM yearly"""
    )

    assert "limit" in out.lower()


def test_a_cte_over_no_real_table_is_still_rejected():
    # Dropping CTE names must not become a way to reference nothing at all.
    with pytest.raises(SQLRejected, match="no known table"):
        v("WITH x AS (SELECT 1 AS n) SELECT n FROM x")


def test_from_inside_extract_is_not_a_table():
    out = v(
        """SELECT EXTRACT(YEAR FROM p.obs_time) AS year, count(*)
FROM profiles p
GROUP BY year"""
    )

    assert "limit" in out.lower()


def test_alias_still_resolves_when_extract_uses_it():
    # EXTRACT(... FROM p...) used to bind `p` to a table called `p`, which
    # silently disabled column checking for every other p.<column> reference.
    with pytest.raises(SQLRejected, match="qc_flag"):
        vc(
            """SELECT EXTRACT(YEAR FROM p.obs_time) AS year, count(*)
FROM profiles p
WHERE p.qc_flag = 1
GROUP BY year"""
        )


def test_column_inside_extract_is_still_checked():
    with pytest.raises(SQLRejected, match="bogus"):
        vc("SELECT EXTRACT(YEAR FROM p.bogus) FROM profiles p")


def test_order_by_on_a_bare_aggregate_is_dropped():
    # PostgreSQL rejects this outright, and ordering one row means nothing.
    out = v("SELECT max(temperature_c) FROM measurements ORDER BY temperature_c")

    assert "order by" not in out.lower()
    assert "max(temperature_c)" in out.lower()


def test_order_by_survives_a_grouped_aggregate():
    out = v(
        "SELECT region, count(*) FROM profiles GROUP BY region ORDER BY region"
    )

    assert "order by region" in out.lower()


def test_order_by_survives_a_window_function():
    out = v(
        "SELECT float_id, rank() OVER (ORDER BY cycle_number) FROM profiles "
        "ORDER BY float_id"
    )

    assert out.lower().count("order by") == 2


def test_order_by_survives_when_nothing_is_aggregated():
    out = v("SELECT float_id FROM profiles ORDER BY float_id")

    assert "order by float_id" in out.lower()


def test_a_subquery_order_by_is_not_stripped():
    out = v(
        "SELECT max(t) FROM (SELECT temperature_c AS t FROM measurements "
        "ORDER BY temperature_c) s"
    )

    assert "order by temperature_c" in out.lower()


def _reference_sql(path):
    """The SQL a reference file contributes, minus its leading question line.

    Mirrors ``schema_context.load_examples``, which is what the model is
    actually shown.
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    return "\n".join(lines[1:]).strip().rstrip(";")


@pytest.mark.parametrize(
    "path",
    sorted(QUERY_DIR.glob("*.sql")),
    ids=lambda path: path.name,
)
def test_reference_queries_survive_the_validator(path):
    """Every hand-written query in db/queries is a gold-set case.

    The first few are fed to the model as few-shot examples, so a pattern the
    validator rejects is one the model is being taught and then punished for.
    """
    validate(_reference_sql(path), ALLOWED)


# --- the platform boundary -------------------------------------------------
#
# A float profiles the water column; a buoy rides the surface. They share no
# key. The catalog says so and the model joined them anyway, twice, so the
# rule lives here now.

ALL_TABLES = [
    "floats",
    "profiles",
    "measurements",
    "drifters",
    "drifter_observations",
]


def va(sql):
    return validate(sql, ALL_TABLES)


def test_joining_the_two_platforms_on_region_is_rejected():
    """Pairs every measurement in a basin with every buoy fix in it. This one
    ran to the statement timeout in evaluation rather than returning a wrong
    answer, which is a worse failure than it looks."""
    with pytest.raises(SQLRejected, match="share no key"):
        va(
            """SELECT p.region, avg(m.temperature_c), avg(o.sst_c)
FROM measurements m
JOIN profiles p ON p.profile_id = m.profile_id
JOIN drifter_observations o ON o.region = p.region
GROUP BY p.region"""
        )


def test_equating_a_buoy_fix_with_a_profile_is_rejected():
    """Asked how deep the buoys dived, the model joined observation_id to
    profile_id and averaged pressure. Buoys have no depth at all."""
    with pytest.raises(SQLRejected, match="share no key"):
        va(
            """SELECT d.buoy_id, avg(p.pressure_dbar) AS mean_depth
FROM drifter_observations o
JOIN drifters d ON d.buoy_id = o.buoy_id
JOIN profiles p ON p.profile_id = o.observation_id
GROUP BY d.buoy_id"""
        )


def test_aggregating_each_platform_separately_is_the_right_answer():
    """The rule exists to push the model here, so this must keep passing."""
    out = va(
        """WITH argo AS (
    SELECT p.region, avg(m.temperature_c) AS t
    FROM measurements m
    JOIN profiles p ON p.profile_id = m.profile_id
    WHERE m.qc_flag = 1 AND m.temperature_c IS NOT NULL
    GROUP BY p.region
), buoys AS (
    SELECT o.region, avg(o.sst_c) AS t
    FROM drifter_observations o
    WHERE o.sst_c IS NOT NULL
    GROUP BY o.region
)
SELECT argo.region, argo.t, buoys.t
FROM argo
JOIN buoys ON buoys.region = argo.region
ORDER BY argo.region"""
    )

    assert "limit" in out.lower()


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) FROM measurements m JOIN profiles p ON p.profile_id = m.profile_id",
        "SELECT count(*) FROM drifter_observations o JOIN drifters d ON d.buoy_id = o.buoy_id",
        "SELECT count(*) FROM floats",
        "SELECT count(*) FROM drifters",
    ],
)
def test_one_platform_at_a_time_is_untouched(sql):
    assert va(sql).lower().startswith("select")


def test_a_subquery_keeps_its_own_scope():
    """A buoy table inside a subquery is not joined to the outer float tables
    merely by appearing in the same statement."""
    out = va(
        """SELECT count(*)
FROM profiles p
WHERE p.region IN (SELECT o.region FROM drifter_observations o)"""
    )

    assert "limit" in out.lower()
