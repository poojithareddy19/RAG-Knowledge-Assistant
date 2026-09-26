"""Prompt construction for answering from the data summaries.

The design goals:

  * The model answers ONLY from the summaries it is handed.
  * Every claim is traceable, so the summaries are numbered and the model is
    asked to reference them as [1], [2], which the UI maps back to the float or
    region each one describes.
  * If the summaries do not cover the question the model must say so rather
    than guess. That is the model-side half of the fallback mechanism; the
    retrieval-side half is the confidence threshold.
  * Every number in a summary is a statistic of the tables, computed at index
    time. The model is told this so it does not present a mean as a reading.

Prompt text is versioned via PROMPT_VERSION so evaluation runs can be attributed
to a specific prompt.
"""
from __future__ import annotations

from src.utils.schemas import Summary

PROMPT_VERSION = "v2"

SYSTEM_PROMPT = (
    "You are FloatChat, an assistant for an archive of ARGO ocean float data. "
    "You answer strictly from the numbered summaries provided. Each summary "
    "describes one float or one ocean region held in the database. Follow "
    "these rules without exception:\n"
    "1. Use ONLY information in the summaries. Do not use prior knowledge "
    "about ARGO, the ocean, or any float.\n"
    "2. Cite the summaries you used with bracketed numbers, e.g. [1], [2].\n"
    "3. If the summaries do not contain enough information to answer, reply "
    "exactly: INSUFFICIENT_CONTEXT\n"
    "4. Be concise and factual. Do not speculate, and do not invent floats, "
    "regions, dates or numbers that are not in the summaries."
)


def build_context_block(sources: list[Summary]) -> str:
    parts = []

    for hit in sources:
        header = f"[{hit.rank}] ({hit.kind} {hit.subject})"
        parts.append(f"{header}\n{hit.text}")

    return "\n\n".join(parts)


def build_messages(question: str, sources: list[Summary]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt)."""
    context = build_context_block(sources)

    user_prompt = (
        f"Summaries:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the summaries above and cite their numbers."
    )

    return SYSTEM_PROMPT, user_prompt
