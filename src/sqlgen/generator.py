"""Generate safe PostgreSQL SELECT statements with optional caching."""

from __future__ import annotations

import os
import re

import httpx

from src.sqlgen.schema_context import build_context
from src.utils import cache


SYSTEM = """You write PostgreSQL SELECT queries.

Rules:
- Output exactly one SQL statement and nothing else. No prose, no markdown.
- SELECT only. Never INSERT, UPDATE, DELETE, DROP, ALTER or CREATE.
- Use only the tables and columns in the schema below.
- Always exclude rows where qc_flag <> 1 and where the measured value is NULL.
- If the question cannot be answered from this schema, output exactly:
UNANSWERABLE
"""

FENCE = re.compile(
    r"```(?:sql)?(.*?)```",
    re.S | re.I,
)


def _clean(text: str) -> str:
    """Remove markdown fences and trailing semicolons."""

    match = FENCE.search(text)

    if match:
        text = match.group(1)

    return text.strip().rstrip(";").strip()


def generate_sql(
    question: str,
    model: str | None = None,
    include_examples: bool = True,
    timeout: int = 90,
    use_cache: bool = True,
    return_cache_flag: bool = False,
) -> str | tuple[str, bool]:
    """Generate one SQL statement for a question.

    ``use_cache=False`` is useful for evaluation because cached responses
    would make latency measurements misleading.
    """

    base = os.environ.get(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )

    model = model or os.environ.get(
        "GENERATION__MODEL",
        "llama3.1:latest",
    )

    if use_cache:
        hit = cache.get(
            question,
            model,
            include_examples,
        )

        if hit is not None:
            return (
                (hit, True)
                if return_cache_flag
                else hit
            )

    prompt = (
        f"{SYSTEM}\n\n"
        f"=== SCHEMA ===\n"
        f"{build_context(include_examples)}\n\n"
        f"=== QUESTION ===\n"
        f"{question}\n\n"
        f"SQL:"
    )

    response = httpx.post(
        f"{base}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 400,
            },
        },
        timeout=timeout,
    )

    response.raise_for_status()

    sql = _clean(
        response.json()["response"]
    )

    # Cache successful SQL, including UNANSWERABLE decisions.
    if use_cache and sql:
        cache.put(
            question,
            model,
            sql,
            include_examples,
        )

    return (
        (sql, False)
        if return_cache_flag
        else sql
    )