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
    platforms=None,
):
    """Return a safe, LIMIT-capped query, or raise SQLRejected.

    ``column_catalog`` is an optional ``{table: {column, ...}}`` mapping. When
    given, qualified references such as ``f.qc_flag`` are checked against the
    table the alias resolves to, so a column the model invented is rejected
    here instead of failing in the database.

    ``platforms`` maps a platform name to the tables belonging to it, and no
    single FROM/JOIN chain may span two of them. It defaults to ``PLATFORMS``
    rather than to off, because a caller that forgets it gets the rule rather
    than silence, and because the evaluation harness calls this with positional
    arguments only.
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

    _check_platforms(
        scan,
        PLATFORMS if platforms is None else platforms,
    )

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


# The two in-situ platforms. A float profiles the water column on a cycle; a
# buoy rides the surface and reports a fix every six hours. They share no key,
# and the catalog says so, so a question about both is answered by aggregating
# each separately and joining the aggregates.
#
# Prose did not hold. Asked to compare float and buoy temperatures the model
# wrote `JOIN drifter_observations o ON o.region = p.region`, which pairs every
# measurement in a basin with every buoy fix in it and ran until the statement
# timeout. Asked how deep the buoys dived it wrote
# `JOIN profiles p ON p.profile_id = o.observation_id`, equating a buoy fix id
# with a profile id, and returned an average pressure as the depth a buoy
# reached. Both are fluent, and the second answers a question about a platform
# that has no depth at all.
PLATFORMS = {
    "argo": frozenset({"floats", "profiles", "measurements"}),
    "drifter": frozenset({"drifters", "drifter_observations"}),
}


# A chain may combine the platforms only through aggregates. These are how an
# aggregate is recognised: a GROUP BY, or an aggregate call in a query with no
# grouping, which collapses to one row.
# The call's own parentheses have already been replaced by a placeholder when
# this runs, so `avg(x)` reads as `avg__span3__`. Both forms are matched: the
# flattened one is what is actually seen, and the literal one keeps the pattern
# readable and correct if it is ever used on unflattened text.
_AGGREGATED = re.compile(
    r"\bgroup\s+by\b"
    r"|\b(?:avg|sum|count|min|max|percentile_cont|percentile_disc)"
    r"\s*(?:\(|__span)",
)

_SPAN = re.compile(r"\(([^()]*)\)")

_SPAN_REF = re.compile(r"__span(\d+)__")

# `name AS __spanN__` binds a CTE to a span. `__spanN__ alias` or
# `__spanN__ AS alias` binds a derived table to one.
_CTE_BIND = re.compile(r"\b([a-z_][a-z0-9_]*)\s+as\s+__span(\d+)__")


def _flatten(scan):
    """Replace every parenthesised span with a ``__spanN__`` placeholder.

    Returns the top-level text and ``{n: body}``. A body may itself contain
    placeholders, so a span's full contents are recovered by following them.

    This replaces an earlier version that blanked spans to spaces. Blanking
    lost which span sat where, and a query could then put each platform in its
    own trivial subquery and join the two: every nesting level held one
    platform, and the level holding the join held no tables at all. A model
    found that on its first attempt at working around the rule.
    """
    spans = {}
    text = scan

    while True:
        match = _SPAN.search(text)

        if not match:
            return text, spans

        index = len(spans)
        spans[index] = match.group(1)
        text = text[: match.start()] + f"__span{index}__" + text[match.end() :]

        if len(spans) > 200:
            # Pathological nesting. Stop rather than loop, and let the rest of
            # the validator judge what is left.
            return text, spans


def _span_text(index, spans, seen=None):
    """A span's text with every nested placeholder expanded."""

    seen = seen or set()

    if index in seen or index not in spans:
        return ""

    seen = seen | {index}
    body = spans[index]

    return body + " " + " ".join(
        _span_text(int(n), spans, seen) for n in _SPAN_REF.findall(body)
    )


def _platforms_of(text, platforms):
    """Which platforms the tables in this text belong to."""

    names = {
        m.group(1)
        for m in re.finditer(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", text)
    }

    return {
        family
        for family, tables in platforms.items()
        if names & tables
    }


def _check_platforms(scan, platforms):
    """Reject a FROM/JOIN chain that spans both platforms unaggregated.

    The correct answer to a question about both platforms has the same shape
    as the way round the rule: two subqueries, one per platform, joined. What
    separates them is aggregation. Two per-region aggregates joined on region
    is one row per region against one row per region, and means something.
    Two raw row sets joined on region pairs every measurement in a basin with
    every buoy fix in it, which is the query that ran to the statement timeout.

    So a chain may span the platforms only when every side carrying one is an
    aggregate. A bare table never qualifies.
    """
    text, spans = _flatten(scan)
    bodies = {i: _span_text(i, spans) for i in spans}

    # Every scope: the statement itself and each span, since a join can live
    # at any depth.
    scopes = [text] + [spans[i] for i in spans]

    for scope in scopes:
        cte = {m.group(1): int(m.group(2)) for m in _CTE_BIND.finditer(scope)}

        # What each FROM/JOIN target contributes, as (platforms, aggregated).
        participants = []

        for m in re.finditer(
            r"\b(?:from|join)\s+(__span(\d+)__|[a-z_][a-z0-9_]*)",
            scope,
        ):
            target, span_no = m.group(1), m.group(2)

            if span_no is not None:
                body = bodies.get(int(span_no), "")
                participants.append((_platforms_of(body, platforms), bool(_AGGREGATED.search(body))))
            elif target in cte:
                body = bodies.get(cte[target], "")
                participants.append((_platforms_of(body, platforms), bool(_AGGREGATED.search(body))))
            else:
                own = {f for f, t in platforms.items() if target in t}
                participants.append((own, False))

        carried = sorted({f for fams, _ in participants for f in fams})

        if len(carried) < 2:
            continue

        if all(agg for fams, agg in participants if fams):
            continue

        raise SQLRejected(
            "a query cannot join the "
            + " tables to the ".join(carried)
            + " tables: the two platforms share no key, so a question about "
            "both is answered by aggregating each separately and joining the "
            "results"
        )


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