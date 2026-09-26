"""Answer generation over the data summaries.

This is the top of the summary path. Given a question it:

    1. retrieves the top-k summaries from the semantic index (+ timing)
    2. estimates confidence from the retrieval scores
    3. if confidence < threshold -> DECLINE, returning the closest summaries
       and an explanation instead of calling the model
    4. otherwise calls the model with a grounded, citation-forcing prompt
    5. if the model itself signals INSUFFICIENT_CONTEXT -> DECLINE
    6. packages everything into one Answer for UI, logging and evaluation

There are therefore two independent guards against hallucination: a
retrieval-side confidence gate and a generation-side self-abstention signal.
"""
from __future__ import annotations

import time

from src.generation.llm import BaseLLM, get_llm
from src.generation.prompt import build_messages
from src.semantic.confidence import estimate_confidence
from src.semantic.index import SemanticIndex
from src.utils.config import get_config
from src.utils.schemas import Answer, Summary

DECLINE_MESSAGE = (
    "I couldn't find enough in the data summaries to answer this question."
)
INSUFFICIENT_SENTINEL = "INSUFFICIENT_CONTEXT"


def _build_retrieval_only_answer(sources: list[Summary]) -> str:
    """A useful fallback when the model is unreachable: the summaries themselves."""
    if not sources:
        return DECLINE_MESSAGE

    lines = [
        "I could not reach the language model, but these summaries are the "
        "closest to your question:",
    ]

    for hit in sources[:2]:
        lines.append(f"- {hit.kind} {hit.subject}: {hit.text}")

    return "\n".join(lines)


class AnswerGenerator:
    def __init__(self, index: SemanticIndex, llm: BaseLLM | None = None) -> None:
        self.index = index
        self.cfg = get_config()
        self._llm = llm  # lazy: only construct on first answered query

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def retrieve(self, question: str) -> tuple[list[Summary], dict[str, float]]:
        """The ranked summaries for a question, and how long they took."""
        started = time.perf_counter()

        sources = self.index.search(question)

        for rank, hit in enumerate(sources, start=1):
            hit.rank = rank

        return sources, {"search_ms": (time.perf_counter() - started) * 1000}

    def answer(self, question: str) -> Answer:
        gen_cfg = self.cfg.generation
        threshold = float(self.cfg.semantic.answer_threshold)
        embedding_model = self.cfg.embeddings.model_name
        latency: dict = {}

        wall_start = time.perf_counter()
        sources, retrieval_timing = self.retrieve(question)
        latency.update(retrieval_timing)
        confidence = estimate_confidence(sources)

        base = dict(
            question=question,
            confidence=confidence,
            sources=sources,
            provider=gen_cfg.provider,
            model=gen_cfg.model,
            embedding_model=embedding_model,
        )

        # --- Guard 1: retrieval-side confidence gate -----------------------
        if not sources or confidence.score < threshold:
            latency["total_ms"] = (time.perf_counter() - wall_start) * 1000
            reason = (
                "No summary is close enough to the question. The index "
                "describes the floats and regions in the database, so this "
                "is probably not something it holds."
                if not sources
                else (
                    f"Confidence {confidence.percent}% is below the "
                    f"{int(threshold * 100)}% threshold; the retrieved "
                    "summaries were not similar enough to the question."
                )
            )
            return Answer(
                answer=DECLINE_MESSAGE,
                answered=False,
                latency_ms=latency,
                reason=reason,
                **base,
            )

        # --- Model generation ----------------------------------------------
        system_prompt, user_prompt = build_messages(question, sources)
        try:
            t0 = time.perf_counter()
            llm_resp = self.llm.generate(system_prompt, user_prompt)
            latency["llm_ms"] = (time.perf_counter() - t0) * 1000
        except Exception as exc:  # provider/network failure -> graceful error
            latency["total_ms"] = (time.perf_counter() - wall_start) * 1000
            return Answer(
                answer=_build_retrieval_only_answer(sources),
                answered=False,
                latency_ms=latency,
                reason=(
                    "The language model could not be reached; the closest "
                    "summaries are shown instead of a generated answer."
                ),
                error=str(exc).strip().splitlines()[0] if str(exc).strip() else "LLM call failed",
                **base,
            )

        text = (llm_resp.text or "").strip()
        latency["total_ms"] = (time.perf_counter() - wall_start) * 1000
        prompt_record = (
            f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
            if gen_cfg.get("expose_prompt", False)
            else None
        )

        # --- Guard 2: generation-side self-abstention ----------------------
        if INSUFFICIENT_SENTINEL in text:
            return Answer(
                answer=DECLINE_MESSAGE,
                answered=False,
                latency_ms=latency,
                tokens=llm_resp.tokens,
                prompt=prompt_record,
                reason="The model judged the retrieved summaries insufficient.",
                **base,
            )

        return Answer(
            answer=text,
            answered=True,
            latency_ms=latency,
            tokens=llm_resp.tokens,
            prompt=prompt_record,
            **base,
        )
