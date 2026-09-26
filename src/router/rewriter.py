"""Turn a follow up question into one that stands on its own.

The alternative was threading conversation history through the router, the SQL
generator and the cache key, which means three components that each have to
decide what history means to them, and a cache that can no longer be keyed on
the question alone. This is one step in front of routing instead: the follow up
becomes a standalone question, and everything downstream stays exactly as
stateless as it was.

The cache stays correct for the same reason. What gets hashed is the rewritten
question, so "and in 2022?" asked after two different conversations produces two
different keys rather than one wrong hit.

A rewrite that goes wrong is the most likely new failure mode here, so both the
original and the rewritten question travel in the result and into the log. A
silent rewrite nobody can see is a system answering a question the user did not
ask, with no way to tell.
"""

from __future__ import annotations

import re

import httpx

from src.generation.llm import keep_alive, model_for, num_ctx, ollama_base_url
from src.monitoring.tracing import llm_span, record_ollama
from src.utils.config import get_config

SYSTEM = """You rewrite a follow up question so that it can be understood on \
its own, without the conversation around it.

Rules:
- Resolve pronouns and elisions using the conversation. "there" and "that
  float" and a bare "and in 2022?" all refer to something said earlier.
- Change nothing else. Keep the user's own wording wherever it already stands
  on its own.
- Add no new constraints. Carrying over a region the user did not change is
  resolving an elision. Adding a year, a depth or a filter nobody mentioned is
  inventing one.
- If the question already names its own subject, repeat it back unchanged.
- Output only the rewritten question. No prose, no quotation marks, no
  explanation.
"""

# A rewrite is one sentence. Anything much longer is the model explaining
# itself, and the original question is a better answer than an explanation.
MAX_CHARS = 400


def rewrite(question: str, history: list[tuple[str, str]]) -> str:
    """Return a standalone version of a follow up question.

    ``history`` is the most recent (question, answer_summary) pairs, oldest
    first. Returns the question unchanged when history is empty, when the
    question already names its own subject, or when the model call fails,
    because a failed rewrite must degrade to today's behaviour rather than
    to an error.

    The middle case is the model's judgement rather than a rule here: the
    prompt tells it to repeat a self-contained question back unchanged, and
    anything this module could test for instead would be a guess about English
    rather than about the conversation.
    """
    settings = _settings()

    if not settings.get("enabled", True):
        return question

    trimmed = _recent(history, settings.get("max_turns", 4))

    if not trimmed:
        return question

    prompt = build_prompt(question, trimmed)

    try:
        raw = _call_model(prompt)
    except Exception:
        return question

    rewritten = _clean(raw) or question

    if not keeps_identifiers(question, rewritten):
        # The model did the opposite of its job: asked "tell me about float
        # 1902373" with two earlier turns in view, it answered "Tell me about
        # that float." A question that names a float, a year or a depth has
        # already resolved its own subject, and a rewrite that drops the
        # number the user typed is answering a different question.
        return question

    return rewritten


# A float id, a year or a pressure: any number long enough to be a subject.
_NUMBER = re.compile(r"\b\d{4,}\b")


def keeps_identifiers(question: str, rewritten: str) -> bool:
    """Whether every number the user typed survives in the rewrite."""
    return set(_NUMBER.findall(question)) <= set(_NUMBER.findall(rewritten))


def build_prompt(question: str, history: list[tuple[str, str]]) -> str:
    """Assemble the rewrite prompt from the history and the follow up."""
    turns = []

    for asked, answered in history:
        turns.append(f"Q: {asked}")
        turns.append(f"A: {answered}")

    return "\n\n".join(
        [
            SYSTEM,
            "=== CONVERSATION ===\n" + "\n".join(turns),
            f"=== FOLLOW UP ===\n{question}",
            "Rewritten:",
        ]
    )


def _recent(history, max_turns: int) -> list[tuple[str, str]]:
    """The last ``max_turns`` pairs, oldest first, ignoring empty ones.

    Capped because the point of the history is to resolve a reference, and a
    reference reaches back a turn or two. A longer window costs prompt tokens
    on every question and gives the model more chances to carry over a filter
    the user has moved on from.
    """
    if not history:
        return []

    pairs = [
        (str(asked).strip(), str(answered or "").strip())
        for asked, answered in history
        if str(asked).strip()
    ]

    return pairs[-max_turns:] if max_turns > 0 else []


def _clean(text: str) -> str:
    """The first line of the reply, without the quotes a model likes to add."""
    line = str(text).strip().splitlines()[0].strip() if str(text).strip() else ""

    line = line.strip('"').strip("'").strip()

    if line.lower().startswith("rewritten:"):
        line = line[len("rewritten:") :].strip()

    return line[:MAX_CHARS]


def _settings() -> dict:
    try:
        return dict(get_config().get("conversation", {}) or {})
    except Exception:
        return {}


def _call_model(prompt: str, timeout: int = 30) -> str:
    """Ask the model for the rewrite. Separated so tests can stub one thing."""
    base = ollama_base_url()
    model = model_for("generation")

    with llm_span("rewrite", model, temperature=0, max_tokens=120) as span:
        response = httpx.post(
            f"{base}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": keep_alive(),
                "options": {
                    "temperature": 0,
                    "num_predict": 120,
                    "num_ctx": num_ctx(),
                },
            },
            timeout=timeout,
        )

        response.raise_for_status()

        data = response.json()
        record_ollama(span, data, prompt=prompt)

    return data["response"]
