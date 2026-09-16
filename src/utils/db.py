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


def _configure(conn):
    """Teach a new connection the pgvector types.

    This has to happen before any cursor is made: a cursor captures the
    connection's adapter map when it is created, so registering afterwards has
    no effect on it and numpy arrays fail to adapt. Running it as the pool's
    configure hook means it happens once per physical connection, ahead of
    every cursor that connection will ever produce.
    """
    try:
        from pgvector.psycopg import register_vector

        register_vector(conn)
    except Exception:
        # A database without the vector extension still serves the SQL path.
        pass


def get_pool(readonly=False):
    """One pool per role, created on first use."""
    name = "ro" if readonly else "rw"

    if name not in _pools:
        _pools[name] = ConnectionPool(
            _url(readonly),
            min_size=1,
            max_size=5,
            open=True,
            configure=_configure,
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