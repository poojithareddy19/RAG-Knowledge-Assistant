import time

from src.utils.db import fetch_all


def run_query(sql, timeout_ms=5000):
    """Execute validated SQL as the read-only user."""
    started = time.perf_counter()

    cols, rows = fetch_all(
        sql,
        readonly=True,
        timeout_ms=timeout_ms,
    )

    elapsed_ms = round(
        (time.perf_counter() - started) * 1000,
        1,
    )

    return {
        "columns": cols,
        "rows": [list(r) for r in rows],
        "row_count": len(rows),
        "elapsed_ms": elapsed_ms,
        "sql": sql,
    }