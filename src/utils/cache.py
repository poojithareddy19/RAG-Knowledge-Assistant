"""A tiny on-disk cache for generated SQL.

The model call is the slow, expensive part of the data route. Identical
questions are common, especially from a demo page, so remembering the SQL we
produced for a question turns a two-second answer into a much faster one.

The TTL prevents stale SQL from surviving indefinitely when the schema
catalog changes.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from src.utils.config import get_config


def _settings() -> tuple[Path, int, bool]:
    cfg = get_config().get("cache", {})

    return (
        Path(
            cfg.get(
                "file",
                "data/cache/sql_cache.json",
            )
        ),
        int(
            cfg.get(
                "ttl_seconds",
                7 * 24 * 3600,
            )
        ),
        bool(
            cfg.get(
                "enabled",
                True,
            )
        ),
    )


def _key(
    question: str,
    model: str,
    include_examples: bool,
) -> str:
    raw = (
        f"{question.strip().lower()}"
        f"|{model}"
        f"|{include_examples}"
    )

    return hashlib.sha256(
        raw.encode()
    ).hexdigest()[:32]


def _load(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError:
        # A corrupt cache should not crash the application.
        return {}


def get(
    question: str,
    model: str,
    include_examples: bool = True,
) -> str | None:
    path, ttl, enabled = _settings()

    if not enabled:
        return None

    entry = _load(path).get(
        _key(
            question,
            model,
            include_examples,
        )
    )

    if not entry:
        return None

    if time.time() - entry["stored_at"] > ttl:
        return None

    return entry["sql"]


def put(
    question: str,
    model: str,
    sql: str,
    include_examples: bool = True,
) -> None:
    path, _, enabled = _settings()

    if not enabled:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = _load(path)

    data[
        _key(
            question,
            model,
            include_examples,
        )
    ] = {
        "sql": sql,
        "stored_at": time.time(),
        "question": question,
    }

    path.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )


def clear() -> None:
    path, _, _ = _settings()

    if path.exists():
        path.unlink()