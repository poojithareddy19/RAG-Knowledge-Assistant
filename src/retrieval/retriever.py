"""Retrieval.

Embeds the query, pulls a candidate pool from the vector store, optionally
reranks with a cross-encoder, and returns the top-k results. Timing for the
embedding and search steps is captured for the monitoring layer.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

from src.embeddings.embedding_model import get_embedding_model
from src.utils.config import get_config
from src.utils.schemas import RetrievedChunk
from src.vectorstore.vectordb import FaissVectorStore


class Retriever:
    def __init__(self, store: FaissVectorStore) -> None:
        self.store = store
        self.cfg = get_config().retrieval
        self.embedder = get_embedding_model()

    def retrieve(self, question: str) -> Tuple[List[RetrievedChunk], Dict[str, float]]:
        """Return (top_k chunks, timing_ms)."""
        timing: Dict[str, float] = {}

        t0 = time.perf_counter()
        query_vec = self.embedder.embed_query(question)
        timing["embed_ms"] = (time.perf_counter() - t0) * 1000

        fetch_k = max(self.cfg.fetch_k, self.cfg.top_k)
        t1 = time.perf_counter()
        candidates = self.store.search(query_vec, k=fetch_k)
        timing["search_ms"] = (time.perf_counter() - t1) * 1000

        if self.cfg.get("use_reranker", False) and candidates:
            from src.retrieval.reranker import rerank

            t2 = time.perf_counter()
            candidates = rerank(question, candidates)
            timing["rerank_ms"] = (time.perf_counter() - t2) * 1000

        top = candidates[: self.cfg.top_k]
        # Re-number ranks after any trimming/reranking.
        for i, rc in enumerate(top, start=1):
            rc.rank = i
        return top, timing
