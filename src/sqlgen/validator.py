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


def validate(
    sql,
    allowed_tables,
    max_limit=5000,
    default_limit=500,
    column_catalog=None,
):
    """Return a safe, LIMIT-capped query, or raise SQLRejected.

    ``column_catalog`` is an optional ``{table: {column, ...}}`` mapping. When
    given, qualified references such as ``f.qc_flag`` are checked against the
    table the alias resolves to, so a column the model invented is rejected
    here instead of failing in the database.
    """

    if not sql:
        raise SQLRejected("model declined to answer from this schema")

    # A refusal may carry the reason after a colon, which is what the scope
    # gate uses to say which quantity is missing. Matching on the prefix
    # rather than the whole string also catches a model that ends the word
    # with a full stop or adds a line of explanation after it: those are
    # refusals too, and before this they fell through to the SQL parser and
    # were reported as a syntax error rather than as a decline.
    head = sql.strip().upper()

    if head.startswith("UNANSWERABLE"):
        _, _, reason = sql.strip().partition(":")

        raise SQLRejected(
            reason.strip()
            or "model declined to answer from this schema"
        )

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

    scan = _blank_function_args(lowered)

    used = _tables(scan)
    unknown = used - {t.lower() for t in allowed_tables}

    if unknown:
        raise SQLRejected(f"unknown table(s): {sorted(unknown)}")

    if not used:
        raise SQLRejected("no known table referenced")

    if column_catalog:
        _check_columns(lowered, scan, column_catalog)

    return _cap_limit(
        _drop_pointless_order_by(str(stmt).strip()),
        max_limit,
        default_limit,
    )


# Functions that take FROM as a keyword inside their argument list. Table
# detection is regex-based, so EXTRACT(YEAR FROM p.obs_time) would otherwise
# register `p` as a table.
_FROM_IN_ARGS = (
    "extract",
    "substring",
    "trim",
    "overlay",
    "position",
)


def _blank_function_args(lowered):
    """Blank the argument list of every function in ``_FROM_IN_ARGS``.

    Blanking rather than deleting keeps every other offset intact, so the
    result can still be scanned with the same expressions as the original.
    An unbalanced query blanks to the end and then fails the
    "no known table referenced" check, which is the safe direction.
    """
    out = list(lowered)

    for name in _FROM_IN_ARGS:
        for m in re.finditer(rf"\b{name}\s*\(", lowered):
            depth = 0

            for i in range(m.end() - 1, len(lowered)):
                char = lowered[i]
                out[i] = " "

                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1

                    if depth == 0:
                        break

    return "".join(out)


# `x AS (` binds a CTE, or a named window. Neither is a table, and a column
# alias never takes a parenthesis, so this form is unambiguous.
_BOUND_NAME = re.compile(
    r"\b([a-z_][a-z0-9_]*)\s+as\s*\(",
)


def _tables(scan):
    """Table names following FROM or JOIN. Deliberately conservative.

    Names bound by the query itself (CTEs) are dropped: they look like tables
    here but are defined above, and the real tables their bodies read from are
    picked up by the same scan.
    """
    found = set()

    for m in re.finditer(
        r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)",
        scan,
    ):
        found.add(m.group(1))

    bound = {m.group(1) for m in _BOUND_NAME.finditer(scan)}

    return found - bound - {"select"}


# Words that can follow a table name without being an alias.
_NOT_AN_ALIAS = {
    "as",
    "on",
    "using",
    "where",
    "group",
    "order",
    "having",
    "limit",
    "offset",
    "fetch",
    "window",
    "union",
    "intersect",
    "except",
    "join",
    "inner",
    "left",
    "right",
    "full",
    "cross",
    "natural",
    "lateral",
    "and",
    "or",
}


def _alias_map(scan):
    """Map every alias and bare table name to the table it refers to."""
    mapping = {}

    for m in re.finditer(
        r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)"
        r"(?:\s+(?:as\s+)?([a-z_][a-z0-9_]*))?",
        scan,
    ):
        table, alias = m.group(1), m.group(2)
        mapping[table] = table

        if alias and alias not in _NOT_AN_ALIAS:
            mapping[alias] = table

    return mapping


def _check_columns(lowered, scan, column_catalog):
    """Reject qualified references to columns the table does not have.

    Aliases are resolved from ``scan`` so a FROM inside a function call cannot
    bind one, but columns are read from the original text so references inside
    those calls are still checked.

    Only qualifiers that resolve to a known table are checked. Anything else
    (a schema prefix, a CTE, a function result) is left alone, so a query is
    never rejected on a guess.
    """
    aliases = _alias_map(scan)

    for m in re.finditer(
        r"\b([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\b",
        lowered,
    ):
        qualifier, column = m.group(1), m.group(2)
        table = aliases.get(qualifier)

        if table is None:
            continue

        known = column_catalog.get(table)

        if known and column not in known:
            raise SQLRejected(
                f"{table} has no column {column!r} "
                f"(referenced as {qualifier}.{column})"
            )


_AGGREGATE = re.compile(r"\b(?:count|sum|avg|min|max)\s*\(", re.I)
_WINDOW = re.compile(r"\bover\s*\(", re.I)

# Only a trailing ORDER BY, and only one with no parenthesis in it, so an
# ORDER BY belonging to a subquery or a window frame is never touched.
_TRAILING_ORDER_BY = re.compile(r"\border\s+by\b[^()]*$", re.I)


def _drop_pointless_order_by(sql):
    """Remove an ORDER BY that PostgreSQL will reject anyway.

    ``SELECT max(temperature_c) FROM measurements ORDER BY temperature_c``
    aggregates to one row, so ordering it is meaningless, and ordering by a
    column that is not grouped is an error rather than a no-op. The model
    emits this often enough that repairing it turns a guaranteed failure into
    the answer it was already trying to give.
    """
    if not _AGGREGATE.search(sql):
        return sql

    lowered = sql.lower()

    if "group by" in lowered or _WINDOW.search(sql):
        return sql

    return _TRAILING_ORDER_BY.sub("", sql).strip()


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