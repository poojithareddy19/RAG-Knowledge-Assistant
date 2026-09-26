import re
from functools import lru_cache
from pathlib import Path

CATALOG = Path("db/schema_catalog.md")
QUERY_DIR = Path("db/queries")


@lru_cache(maxsize=1)
def load_column_catalog():
    """Return ``{table: frozenset(columns)}`` read from the live database.

    Used by the validator to reject columns the model invented. Read from
    information_schema rather than the markdown catalog so it cannot drift
    away from the real tables.
    """
    from src.utils.db import fetch_all

    _, rows = fetch_all(
        """
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        """,
        readonly=True,
    )

    catalog = {}

    for table, column in rows:
        catalog.setdefault(table, set()).add(column)

    return {
        table: frozenset(columns)
        for table, columns in catalog.items()
    }


# The buoy tables and their conventions are marked in the catalog so an Argo
# question can be shown the catalog without them. They were a sixth of the SQL
# prompt, paid by every question, and the prompt had outgrown the model's
# context window: Ollama cut it to its first and last tokens and the model
# never saw the rules. The markers themselves never reach the model.
_DRIFTER_BLOCK = re.compile(r"<!-- drifters -->\n.*?<!-- /drifters -->\n", re.S)
_MARKER = re.compile(r"<!-- /?drifters -->\n")


def load_catalog(include_drifters=True):
    if not CATALOG.exists():
        raise FileNotFoundError(f"missing {CATALOG}")

    text = CATALOG.read_text(encoding="utf-8")

    if not include_drifters:
        text = _DRIFTER_BLOCK.sub("", text)

    return _MARKER.sub("", text)


def load_examples(limit=4):
    if not QUERY_DIR.exists():
        return []

    files = sorted(QUERY_DIR.glob("*.sql"))[:limit]
    out = []

    for f in files:
        text = f.read_text(encoding="utf-8").strip()
        lines = text.splitlines()

        question = (
            lines[0].lstrip("-\t").lstrip()
            if lines
            else f.stem
        )
        sql = "\n".join(lines[1:]).strip()

        out.append({
            "question": question,
            "sql": sql,
        })

    return out


def build_context(include_examples=True, include_drifters=True):
    parts = [load_catalog(include_drifters)]

    if include_examples:
        for ex in load_examples():
            parts.append(
                f"Question:\t{ex['question']}\n"
                f"SQL:\n{ex['sql']}"
            )

    return "\n\n".join(parts)

# ---------------------------------------------------------------------------
# What period the archive covers
# ---------------------------------------------------------------------------
#
# The problem statement's second example is "compare BGC parameters in the
# Arabian Sea for the last 6 months". Nothing told the model what "now" was or
# where the data ended, so it wrote `now() - interval '6 months'`: a window
# that starts after the archive's last profile, which returns nothing and
# reads as "there is no data" rather than "you asked about a period the
# snapshot does not reach".
#
# This is read from the tables rather than written into the catalog, because
# the catalog cannot know when the next load lands. It uses its own connection
# with a two second timeout and not the shared pool, which waits thirty seconds
# for a connection that is not coming: the prompt is built in CI, in the tests
# and on a laptop whose database is off, and each of those must fail fast.

_COVERAGE = {"text": None, "at": 0.0}

COVERAGE_MAX_AGE = 600.0

_COVERAGE_SQL = (
    ("Argo profiles", "SELECT min(obs_time)::date, max(obs_time)::date FROM profiles"),
    (
        "Drifting buoy fixes",
        "SELECT min(obs_time)::date, max(obs_time)::date FROM drifter_observations",
    ),
)


def _read_coverage() -> list[tuple[str, object, object]]:
    import os

    import psycopg

    url = os.environ.get("DATABASE__READONLY_URL") or os.environ.get("DATABASE__URL")

    if not url:
        return []

    spans = []

    with psycopg.connect(url, connect_timeout=2) as conn:
        for label, sql in _COVERAGE_SQL:
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    first, last = cur.fetchone()
            except Exception:
                conn.rollback()
                continue

            if first and last:
                spans.append((label, first, last))

    return spans


def coverage_note(spans) -> str:
    """The prompt text for a set of (label, first, last) spans, or empty."""
    if not spans:
        return ""

    from dateutil.relativedelta import relativedelta

    lines = [
        f"- {label} run from {first.isoformat()} to {last.isoformat()}."
        for label, first, last in spans
    ]

    latest_label, _, latest = max(spans, key=lambda span: span[2])
    window_start = latest - relativedelta(months=6)

    return (
        "=== WHAT PERIOD THE DATABASE COVERS ===\n"
        + "\n".join(lines)
        + "\n\nThis is a snapshot, not a live feed: nothing is newer than the "
        "dates above. A relative period in the question, such as \"the last "
        "6 months\", \"recent\", \"this year\" or \"lately\", means relative to "
        "the latest date of the platform being asked about, not relative to "
        "today, and never now() or current_date. Write it with explicit date "
        "literals so the window is visible in the query. For example, for "
        f"{latest_label}, \"the last 6 months\" means obs_time >= "
        f"'{window_start.isoformat()}' AND obs_time <= '{latest.isoformat()}'."
    )


def data_coverage() -> str:
    """The coverage section of the SQL prompt, cached, or empty if unreadable.

    A failure is cached as well, for the same ten minutes, so a process with no
    database pays the two second timeout once rather than on every prompt.
    """
    import time

    now = time.monotonic()

    if _COVERAGE["text"] is not None and now - _COVERAGE["at"] < COVERAGE_MAX_AGE:
        return _COVERAGE["text"]

    try:
        text = coverage_note(_read_coverage())
    except Exception:
        text = ""

    _COVERAGE.update(text=text, at=now)

    return text
