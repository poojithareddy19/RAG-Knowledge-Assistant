"""Prompt construction.

Builds the grounded-answering prompt. The design goals:

  * The model must answer ONLY from the provided context.
  * Every claim should be traceable, so we number the context blocks and ask
    the model to reference them as [1], [2] ... which we map back to real
    citations in the UI.
  * If the context is insufficient the model must say so rather than guess —
    this is the model-side half of the fallback mechanism (the retrieval-side
    half is the confidence threshold).

Prompt text is versioned via PROMPT_VERSION so evaluation runs can be attributed
to a specific prompt (a lightweight form of prompt versioning).
"""
from __future__ import annotations

from typing import List, Tuple

from src.utils.schemas import RetrievedChunk

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are a knowledge assistant that answers strictly from the provided "
    "context passages. Follow these rules without exception:\n"
    "1. Use ONLY information in the context. Do not use prior knowledge.\n"
    "2. Cite the passages you used with bracketed numbers, e.g. [1], [2].\n"
    "3. If the context does not contain enough information to answer, reply "
    "exactly: INSUFFICIENT_CONTEXT\n"
    "4. Be concise and factual. Do not speculate."
)


def build_context_block(sources: List[RetrievedChunk]) -> str:
    parts = []
    for rc in sources:
        c = rc.chunk
        header = f"[{rc.rank}] ({c.doc_name}, page {c.page}, {c.chunk_id})"
        parts.append(f"{header}\n{c.text}")
    return "\n\n".join(parts)


def build_messages(question: str, sources: List[RetrievedChunk]) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt)."""
    context = build_context_block(sources)
    user_prompt = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above and cite passage numbers."
    )
    return SYSTEM_PROMPT, user_prompt
