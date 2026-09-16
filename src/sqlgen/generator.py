"""Generate safe PostgreSQL SELECT statements with optional caching."""

from __future__ import annotations

import hashlib
import os
import re

import httpx

from src.sqlgen.schema_context import build_context
from src.utils import cache


SYSTEM = """You write PostgreSQL SELECT queries.

Rules:
- Output exactly one SQL statement and nothing else. No prose, no markdown.
- SELECT only. Never INSERT, UPDATE, DELETE, DROP, ALTER or CREATE.
- Use only the tables and columns in the schema below. Never assume a column
  exists on a table just because it exists on another one. qc_flag lives on
  measurements only.
- Always exclude rows where qc_flag <> 1 and where the measured value is NULL.
- When the question names a period (a year, a month, a range), bound it at
  BOTH ends. "in 2020" means obs_time >= '2020-01-01' AND obs_time <
  '2021-01-01', never an open-ended >= alone.
- Whenever the result is a series or a grouped breakdown, end with ORDER BY on
  the grouping column so the rows come back in a meaningful order. Do not add
  ORDER BY to a query that returns one aggregate row and has no GROUP BY:
  ordering by a column you did not group by is rejected by PostgreSQL.
- If the question cannot be answered from this schema, output exactly:
UNANSWERABLE
"""

CONTEXT_RULE = """The notes below describe what this database actually holds:
which floats exist, where they worked and when. Use them only to work out what
the question refers to, such as which float id or which region name.

They are background, not query terms:

- Do not turn a value from a note into a WHERE condition unless the question
  itself asks for it. A note saying a float is an APEX platform on project
  INCOIS does not mean the question is about APEX platforms or that project.
  Those notes describe every float in the database, not a filter.
- Do not narrow a query to the dates or ranges a note happens to mention. If
  the question names no period, do not bound one.
- Never copy a number out of a note as an answer. Compute every number with
  SQL, even when a note appears to state it already.
"""

# Bump when SYSTEM or CONTEXT_RULE changes so cached SQL is not reused.
PROMPT_VERSION = "5"

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


def build_prompt(
    question: str,
    context: str = "",
    include_examples: bool = True,
) -> str:
    """Assemble the full prompt, with the semantic block only when there is one."""

    sections = [
        SYSTEM,
        f"=== SCHEMA ===\n{build_context(include_examples)}",
    ]

    if context.strip():
        sections.append(
            f"{CONTEXT_RULE}\n=== WHAT IS IN THE DATABASE ===\n{context.strip()}"
        )

    sections.append(f"=== QUESTION ===\n{question}\n\nSQL:")

    return "\n\n".join(sections)


def cache_version(
    context: str = "",
    include_examples: bool = True,
) -> str:
    """Prompt version, extended by a digest of everything but the question.

    One rule: if any part of the prompt other than the question changes, a
    cached answer is not reused. That covers the retrieved context, the schema
    catalog, the worked examples and the instructions themselves.

    Keying only on the hand-maintained version constant was not enough. Adding
    the biogeochemical columns to the catalog changed what the model was told
    and left every cached entry looking valid, so the cache would have served
    SQL written by a model that had never heard of those columns.
    """
    digest = hashlib.sha256(
        build_prompt("", context, include_examples).encode()
    ).hexdigest()[:12]

    return f"{PROMPT_VERSION}:{digest}"


def generate_sql(
    question: str,
    model: str | None = None,
    include_examples: bool = True,
    timeout: int = 90,
    use_cache: bool = True,
    return_cache_flag: bool = False,
    context: str = "",
) -> str | tuple[str, bool]:
    """Generate one SQL statement for a question.

    ``context`` is the semantic layer's description of what the database holds,
    retrieved for this question. Empty means the model sees the schema alone.

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

    version = cache_version(context, include_examples)

    if use_cache:
        hit = cache.get(
            question,
            model,
            include_examples,
            version=version,
        )

        if hit is not None:
            return (
                (hit, True)
                if return_cache_flag
                else hit
            )

    prompt = build_prompt(
        question,
        context,
        include_examples,
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 400,
        },
    }

    try:
        response = httpx.post(
            f"{base}/api/generate",
            json=payload,
            timeout=timeout,
        )
    except httpx.TimeoutException:
        # The first call after a restart pays for loading the model into
        # memory, which on its own can outlast the timeout. By now that load
        # has finished, so one retry is usually enough.
        response = httpx.post(
            f"{base}/api/generate",
            json=payload,
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
            version=version,
        )

    return (
        (sql, False)
        if return_cache_flag
        else sql
    )