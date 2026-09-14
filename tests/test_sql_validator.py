import pytest

from src.sqlgen.validator import SQLRejected, validate

ALLOWED = ["floats", "profiles", "measurements"]


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