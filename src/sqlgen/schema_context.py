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


def load_catalog():
    if not CATALOG.exists():
        raise FileNotFoundError(f"missing {CATALOG}")
    return CATALOG.read_text(encoding="utf-8")


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


def build_context(include_examples=True):
    parts = [load_catalog()]

    if include_examples:
        for ex in load_examples():
            parts.append(
                f"Question:\t{ex['question']}\n"
                f"SQL:\n{ex['sql']}"
            )

    return "\n\n".join(parts)