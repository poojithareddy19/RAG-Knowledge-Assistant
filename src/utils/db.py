# src/utils/db.py

import os
from contextlib import contextmanager

from dotenv import load_dotenv
from psycopg_pool import ConnectionPool

load_dotenv()

_pools = {}


def _url(readonly):
    key = "DATABASE__READONLY_URL" if readonly else "DATABASE__URL"
    url = os.environ.get(key)

    if not url:
        raise RuntimeError(
            f"{key} is not set. Copy .env.example to .env."
        )

    return url


def get_pool(readonly=False):
    """One pool per role, created on first use."""
    name = "ro" if readonly else "rw"

    if name not in _pools:
        _pools[name] = ConnectionPool(
            _url(readonly),
            min_size=1,
            max_size=5,
            open=True,
        )

    return _pools[name]


@contextmanager
def cursor(readonly=False, timeout_ms=None):
    """Borrow a connection, hand back a cursor, always return the connection."""
    pool = get_pool(readonly)

    with pool.connection() as conn:
        if readonly:
            conn.read_only = True

        with conn.cursor() as cur:
            if timeout_ms:
                cur.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (str(timeout_ms),),
            )

            yield cur


def fetch_all(sql, params=None, readonly=True, timeout_ms=5000):
    with cursor(
        readonly=readonly,
        timeout_ms=timeout_ms,
    ) as cur:
        cur.execute(sql, params or ())
        cols = [d.name for d in cur.description]
        return cols, cur.fetchall()
def close_pools():
    """Close all database connection pools."""
    for pool in _pools.values():
        pool.close()
    _pools.clear()