"""Typed data contracts shared across the pipeline.

Keeping these in one place means every module — ingestion, retrieval,
generation, evaluation, monitoring — agrees on the shape of the data it passes
around. This is the backbone that keeps the architecture modular.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A single indexable unit of text plus provenance metadata."""

    chunk_id: str            # stable id: "<doc>::p<page>::c<n>"
    doc_name: str
    page: int                # 1-based page number (0 if not paginated)
    text: str
    char_len: int = 0
    token_estimate: int = 0

    def __post_init__(self) -> None:
        self.char_len = len(self.text)
        # Rough heuristic: ~4 chars/token for English. Good enough for display
        # and cost estimation without importing a tokenizer per document type.
        self.token_estimate = max(1, self.char_len // 4)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever with its relevance score."""

    chunk: Chunk
    score: float             # cosine similarity in [-1, 1], typically [0, 1]
    rank: int                # 1-based position in the result list

    def to_dict(self) -> dict[str, Any]:
        d = self.chunk.to_dict()
        d.update({"score": self.score, "rank": self.rank})
        return d


@dataclass
class Confidence:
    """Structured confidence breakdown for explainability."""

    score: float                       # normalized 0-1
    percent: int                       # 0-100 for display
    components: dict[str, float] = field(default_factory=dict)
    n_supporting: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Answer:
    """The full result of a query, ready for UI, logging and evaluation."""

    question: str
    answer: str
    answered: bool                     # False when the system declined
    confidence: Confidence
    sources: list[RetrievedChunk] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    provider: str = ""
    model: str = ""
    embedding_model: str = ""
    prompt: str | None = None       # populated when expose_prompt is on
    reason: str = ""                   # why an answer was declined, if so
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "answered": self.answered,
            "confidence": self.confidence.to_dict(),
            "sources": [s.to_dict() for s in self.sources],
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
            "provider": self.provider,
            "model": self.model,
            "embedding_model": self.embedding_model,
            "prompt": self.prompt,
            "reason": self.reason,
            "error": self.error,
        }
