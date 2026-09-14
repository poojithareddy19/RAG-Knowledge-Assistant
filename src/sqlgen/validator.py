import re

import sqlparse

FORBIDDEN = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "grant",
    "revoke",
    "copy",
    "vacuum",
    "call",
    "do",
    "merge",
}

# pg_read_file and friends are readable but should never be reachable
FORBIDDEN_IDENTIFIERS = {
    "pg_read_file",
    "pg_ls_dir",
    "lo_import",
    "lo_export",
    "dblink",
    "pg_sleep",
}


class SQLRejected(Exception):
    pass


def validate(sql, allowed_tables, max_limit=5000, default_limit=500):
    """Return a safe, LIMIT-capped query, or raise SQLRejected."""
    if not sql or sql.strip().upper() == "UNANSWERABLE":
        raise SQLRejected("model declined to answer from this schema")

    statements = [s for s in sqlparse.parse(sql) if str(s).strip()]

    if len(statements) != 1:
        raise SQLRejected(f"expected 1 statement, got {len(statements)}")

    stmt = statements[0]

    if stmt.get_type() != "SELECT":
        raise SQLRejected(
            f"statement type is {stmt.get_type()}, not SELECT"
        )

    lowered = str(stmt).lower()

    for word in FORBIDDEN:
        if re.search(rf"\b{word}\b", lowered):
            raise SQLRejected(f"forbidden keyword: {word}")

    for ident in FORBIDDEN_IDENTIFIERS:
        if ident in lowered:
            raise SQLRejected(f"forbidden identifier: {ident}")

    if ";" in str(stmt).strip().rstrip(";"):
        raise SQLRejected("statement chaining is not allowed")

    used = _tables(lowered)
    unknown = used - {t.lower() for t in allowed_tables}

    if unknown:
        raise SQLRejected(f"unknown table(s): {sorted(unknown)}")

    if not used:
        raise SQLRejected("no known table referenced")

    return _cap_limit(
        str(stmt).strip(),
        max_limit,
        default_limit,
    )


def _tables(lowered):
    """Table names following FROM or JOIN. Deliberately conservative."""
    found = set()

    for m in re.finditer(
        r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)",
        lowered,
    ):
        found.add(m.group(1))

    return found - {"select"}


def _cap_limit(sql, max_limit, default_limit):
    m = re.search(r"\blimit\s+(\d+)\s*$", sql, re.I)

    if m:
        if int(m.group(1)) > max_limit:
            return re.sub(
                r"\blimit\s+\d+\s*$",
                f"LIMIT {max_limit}",
                sql,
                flags=re.I,
            )
        return sql

    return f"{sql} LIMIT {default_limit}"