"""Answer generation orchestrator.

This is the top of the query pipeline. Given a question it:

    1. retrieves top-k chunks (+ timing)
    2. estimates confidence from retrieval signals
    3. if confidence < threshold -> DECLINE (graceful fallback), returning the
       closest passages and an explanation instead of calling the LLM
    4. otherwise calls the LLM with a grounded, citation-forcing prompt
    5. if the LLM itself signals INSUFFICIENT_CONTEXT -> DECLINE
    6. packages everything into a single Answer object for UI + logging + eval

There are therefore two independent guards against hallucination: a
retrieval-side confidence gate and a generation-side self-abstention signal.
"""
from __future__ import annotations

import re
import time

from src.generation.llm import BaseLLM, get_llm
from src.generation.prompt import build_messages
from src.retrieval.confidence import estimate_confidence
from src.retrieval.retriever import Retriever
from src.utils.config import get_config
from src.utils.schemas import Answer, RetrievedChunk

DECLINE_MESSAGE = (
    "I couldn't find sufficient evidence in the uploaded documents to answer "
    "this question."
)
INSUFFICIENT_SENTINEL = "INSUFFICIENT_CONTEXT"


def _sanitize_error_message(message: str) -> str:
    """Mask API key-like fragments before surfacing provider errors."""
    # Mask common OpenAI key formats so logs/UI never expose real fragments.
    return re.sub(r"\bsk-[A-Za-z0-9_-]+\b", "sk-***", message)


def _is_auth_error(message: str) -> bool:
    low = message.lower()
    markers = (
        "invalid_api_key",
        "incorrect api key",
        "no api key provided",
        "authentication",
        "unauthorized",
        "status': 401",
        "status\": 401",
        "openai_api_key",
    )
    return any(m in low for m in markers)


def _is_provider_capacity_error(message: str) -> bool:
    low = message.lower()
    markers = (
        "insufficient_quota",
        "rate_limit",
        "rate limit",
        "status': 429",
        "status\": 429",
        "too many requests",
    )
    return any(m in low for m in markers)


def _build_retrieval_only_answer(sources: list[RetrievedChunk]) -> str:
    """Return a useful fallback response from top retrieved passages."""
    if not sources:
        return DECLINE_MESSAGE
    top = sources[:2]
    lines = [
        "I could not use the configured LLM right now, but these passages are most relevant:",
    ]
    for s in top:
        excerpt = " ".join(s.chunk.text.split())[:320]
        lines.append(f"- {s.chunk.doc_name} (page {s.chunk.page}): {excerpt}")
    return "\n".join(lines)


class AnswerGenerator:
    def __init__(self, retriever: Retriever, llm: BaseLLM | None = None) -> None:
        self.retriever = retriever
        self.cfg = get_config()
        self._llm = llm  # lazy: only construct on first answered query

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def answer(self, question: str) -> Answer:
        gen_cfg = self.cfg.generation
        threshold = float(self.cfg.confidence.answer_threshold)
        embedding_model = self.cfg.embeddings.model_name
        latency: dict = {}

        wall_start = time.perf_counter()
        sources, retrieval_timing = self.retriever.retrieve(question)
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
                "No documents indexed."
                if not sources
                else (
                    f"Confidence {confidence.percent}% is below the "
                    f"{int(threshold * 100)}% threshold; the retrieved passages "
                    "were not similar enough to the question."
                )
            )
            return Answer(
                answer=DECLINE_MESSAGE,
                answered=False,
                latency_ms=latency,
                reason=reason,
                **base,
            )

        # --- LLM generation ------------------------------------------------
        system_prompt, user_prompt = build_messages(question, sources)
        try:
            t0 = time.perf_counter()
            llm_resp = self.llm.generate(system_prompt, user_prompt)
            latency["llm_ms"] = (time.perf_counter() - t0) * 1000
        except Exception as exc:  # provider/network failure -> graceful error
            latency["total_ms"] = (time.perf_counter() - wall_start) * 1000
            err_text = _sanitize_error_message(str(exc))
            if _is_auth_error(err_text) or _is_provider_capacity_error(err_text):
                return Answer(
                    answer=_build_retrieval_only_answer(sources),
                    answered=False,
                    latency_ms=latency,
                    reason=(
                        "LLM provider unavailable; returned retrieval-only fallback. "
                        "Check API credentials/quota in .env for full generated answers."
                    ),
                    error=None,
                    **base,
                )

            return Answer(
                answer="The language model could not be reached.",
                answered=False,
                latency_ms=latency,
                reason="LLM call failed.",
                error=err_text,
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
                reason="The model judged the retrieved context insufficient.",
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
