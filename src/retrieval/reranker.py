"""Optional cross-encoder reranker (bonus feature).

Bi-encoder retrieval (what FAISS does) scores query and passage independently,
which is fast but coarse. A cross-encoder scores the (query, passage) pair
jointly and is much more accurate — but too slow to run over the whole corpus.
The standard pattern is: retrieve a wide pool with the bi-encoder, then rerank
that small pool with the cross-encoder. Enable via ``retrieval.use_reranker``.
"""
from __future__ import annotations

from functools import lru_cache

from src.utils.config import get_config
from src.utils.schemas import RetrievedChunk


@lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_config().retrieval.reranker_model)


def rerank(question: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Reorder candidates by cross-encoder relevance without changing cosine scores."""
    if not candidates:
        return candidates

    model = _get_cross_encoder()
    pairs = [(question, rc.chunk.text) for rc in candidates]
    scores = model.predict(pairs)

    # Keep the original FAISS cosine similarity in rc.score.
    # Use the cross-encoder score only to determine the new ordering.
    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )

    return [rc for rc, _ in ranked]
