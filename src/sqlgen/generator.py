import os
import re

import httpx

from src.sqlgen.schema_context import build_context

SYSTEM = """You write PostgreSQL SELECT queries.
Rules:
- Output exactly one SQL statement and nothing else. No prose, no markdown.
- SELECT only. Never INSERT, UPDATE, DELETE, DROP, ALTER or CREATE.
- Use only the tables and columns in the schema below.
- Always exclude rows where qc_flag <> 1 and where the measured value is NULL.
- If the question cannot be answered from this schema, output exactly:
UNANSWERABLE
"""

FENCE = re.compile(r"```(?:sql)?(.*?)```", re.S | re.I)


def _clean(text):
    m = FENCE.search(text)
    if m:
        text = m.group(1)
    return text.strip().rstrip(";").strip()


def generate_sql(
    question,
    model=None,
    include_examples=True,
    timeout=90,
):
    base = os.environ.get(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )
    model = model or os.environ.get(
        "GENERATION__MODEL",
        "llama3.1:8b",
    )

    prompt = (
        f"{SYSTEM}\n\n=== SCHEMA ===\n"
        f"{build_context(include_examples)}\n\n"
        f"=== QUESTION ===\n{question}\n\nSQL:"
    )

    r = httpx.post(
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
    r.raise_for_status()

    return _clean(r.json()["response"])