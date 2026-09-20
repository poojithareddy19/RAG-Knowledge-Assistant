"""Split a compound question into the half each backend can answer.

A question like "what does the manual say about a QC flag of 4, and how many
of my measurements carry one" is two questions wearing one coat. Sent whole to
the retriever, the counting clause is noise in the embedding. Sent whole to the
SQL generator, the manual clause invites a join against a table that does not
exist. Each backend does better on its own half.

The split is one model call, deliberately small: it rewrites, it does not
answer. Every failure path returns the original question twice, which is the
behaviour the system had before this module existed and is never worse than a
wrong split, because a backend given the whole question still tends to answer
its own part of it.
"""

from __future__ import annotations

import json
import os

import httpx

SYSTEM = """Split a question into two standalone questions.

documents - the part answered by reading a manual: what a term means, how a
            procedure works, what a flag or a code stands for
data      - the part answered by querying a measurements database: a count, an
            average, a maximum, a list of rows

Rules:
- Each part must stand on its own, with no pronoun pointing at the other part.
- Use the words of the original question. Do not answer either part.
- If one half is not really asked, repeat the original question there.

Reply with JSON only: {"documents": "...", "data": "..."}
"""

MAX_CHARS = 400


def split(question: str) -> tuple[str, str]:
    """The documents question and the data question, in that order.

    Returns ``(question, question)`` unchanged when the model cannot be
    reached or answers with something unusable.
    """
    try:
        payload = _call_model(question)
    except Exception:
        return question, question

    documents = _clean(payload.get("documents"), question)
    data = _clean(payload.get("data"), question)

    return documents, data


def _clean(value: object, fallback: str) -> str:
    """One half of the split, or the whole question if the half is unusable.

    A model that returns an empty string, or a fragment too short to be a
    question, has not split anything. Falling back to the original costs a
    slightly worse retrieval, where accepting the fragment costs the answer.
    """
    text = str(value or "").strip().strip('"').strip()

    if len(text) < 8:
        return fallback

    return text[:MAX_CHARS]


def _call_model(question: str, timeout: int = 60) -> dict:
    """One split call. Separated so tests can stub a single thing."""
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("ROUTER__MODEL", "llama3.1:latest")

    response = httpx.post(
        f"{base}/api/generate",
        json={
            "model": model,
            "prompt": f"{SYSTEM}\n\nQuestion: {question}\nJSON:",
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 200},
        },
        timeout=timeout,
    )

    response.raise_for_status()

    return json.loads(response.json()["response"])
