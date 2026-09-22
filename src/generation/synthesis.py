"""Write one answer from a manual passage and a query result.

This is the only place in the project where a model is shown two sources at
once, which makes it the only place that can invent a relationship between
them. The prompt is written against that: it is given the document answer and
the rows, and told to join them with a sentence, not to reason about them. It
may not compute, round, compare or conclude, because every number in the result
is already correct and any arithmetic it does is arithmetic nobody checked.

The fallback is not an error. ``_staple`` puts the two answers one after the
other under their own headings, which is a worse answer and a true one, so a
model that is unreachable or disagreeable costs the user prose rather than
facts.
"""

from __future__ import annotations

import os

import httpx

SYSTEM = """You are given an answer from a manual and the result of a database
query. Both are correct. Write one short answer that uses both.

Rules:
- Two or three sentences. No headings, no bullet points, no preamble.
- Every number you write must appear in the result exactly as it appears there.
  Do not round, total, average or compare numbers yourself.
- Keep any citation in square brackets exactly as written, but only on a claim
  that came from the manual. A number from the result did not come from the
  manual, so never put a citation after it.
- State only what the two sources say. If they do not connect, say what each
  one says and stop.
- Do not mention "the database", "the query", "the context" or these rules.
"""

MAX_ROWS = 20
MAX_CHARS = 1200


def combine(
    question: str,
    document_answer: str,
    data_answer: str,
    columns: list | None = None,
    rows: list | None = None,
) -> tuple[str, str]:
    """One answer drawing on both halves, and how it was produced.

    Returns ``(text, "model")`` when the synthesis call succeeds and
    ``(text, "stapled")`` when it falls back, so a caller can tell the reader
    which one they are looking at rather than presenting both as the same
    thing.
    """
    if not document_answer and not data_answer:
        return "", "stapled"

    if not document_answer or not data_answer:
        return _staple(document_answer, data_answer), "stapled"

    try:
        text = _call_model(
            _prompt(question, document_answer, data_answer, columns, rows)
        )
    except Exception:
        return _staple(document_answer, data_answer), "stapled"

    text = text.strip()

    if not text:
        return _staple(document_answer, data_answer), "stapled"

    return text[:MAX_CHARS], "model"


def _prompt(
    question: str,
    document_answer: str,
    data_answer: str,
    columns: list | None,
    rows: list | None,
) -> str:
    """Everything the model is allowed to see, and nothing else."""
    parts = [
        f"Question: {question}",
        "",
        "From the manuals:",
        document_answer,
        "",
        "From the database:",
        data_answer,
    ]

    table = _table(columns, rows)

    if table:
        parts += ["", table]

    return "\n".join(parts)


def _table(columns: list | None, rows: list | None) -> str:
    """The result as text, capped, because a long table buries the question."""
    if not columns or not rows:
        return ""

    lines = [" | ".join(str(column) for column in columns)]

    for row in rows[:MAX_ROWS]:
        lines.append(" | ".join("" if cell is None else str(cell) for cell in row))

    if len(rows) > MAX_ROWS:
        lines.append(f"... {len(rows) - MAX_ROWS} more row(s)")

    return "\n".join(lines)


def _staple(document_answer: str, data_answer: str) -> str:
    """Both answers, labelled, when one cannot be written."""
    parts = []

    if document_answer:
        parts.append(f"**From the manuals:** {document_answer}")

    if data_answer:
        parts.append(f"**From the data:** {data_answer}")

    return "\n\n".join(parts)


def _call_model(prompt: str, timeout: int = 90) -> str:
    """One synthesis call. Separated so tests can stub a single thing."""
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("GENERATION__MODEL", "llama3.1:latest")

    response = httpx.post(
        f"{base}/api/generate",
        json={
            "model": model,
            "prompt": f"{SYSTEM}\n\n{prompt}\n\nAnswer:",
            "stream": False,
            "options": {"temperature": 0, "num_predict": 300},
        },
        timeout=timeout,
    )

    response.raise_for_status()

    return response.json()["response"]
