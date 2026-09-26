"""Typed data contracts shared across the pipeline.

Keeping these in one place means every module, retrieval, generation,
evaluation and monitoring, agrees on the shape of the data it passes around.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Summary:
    """One retrieved data summary and how close it was to the question.

    A summary describes one subject: a float or a region. The subject is the
    citation. An answer that rests on "float 1901393" names the float, which a
    reader can then check against the tables the summary was built from.
    """

    kind: str                # "float" or "region"
    subject: str             # the WMO number or the region name
    text: str
    score: float             # cosine similarity, or 1.0 when looked up by name
    rank: int = 0            # 1-based position in the result list

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    """The full result of a summary query, ready for UI, logging and evaluation."""

    question: str
    answer: str
    answered: bool                     # False when the system declined
    confidence: Confidence
    sources: list[Summary] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    provider: str = ""
    model: str = ""
    embedding_model: str = ""
    prompt: str | None = None       # populated when expose_prompt is on
    reason: str = ""                   # why an answer was declined, if so
    error: str | None = None
    trace_id: str | None = None        # joins this answer to its spans

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
            "trace_id": self.trace_id,
        }
