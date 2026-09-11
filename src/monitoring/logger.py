"""Structured interaction logging.

Every query produces one JSON object appended to ``logs/interactions.jsonl``
(one JSON document per line — easy to stream, grep and load into pandas). We log
the fields required for both monitoring and offline evaluation: question,
retrieved chunk ids + scores, confidence, answer, latency, provider, embedding
model, token usage, and any error.

A second standard Python logger emits human-readable operational lines to
``logs/app.log`` for debugging.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.utils.config import get_config
from src.utils.schemas import Answer

_configured = False


def _ensure_logging() -> logging.Logger:
    global _configured
    cfg = get_config()
    log_dir = Path(cfg.paths.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rag")
    if not _configured:
        logger.setLevel(getattr(logging, cfg.monitoring.level, logging.INFO))
        handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)
        _configured = True
    return logger


def log_interaction(answer: Answer) -> None:
    """Append one structured record describing a completed query."""
    cfg = get_config()
    logger = _ensure_logging()

    record: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": answer.question,
        "answer": answer.answer,
        "answered": answer.answered,
        "confidence": answer.confidence.percent,
        "confidence_components": answer.confidence.components,
        "reason": answer.reason,
        "retrieved": [
            {
                "chunk_id": s.chunk.chunk_id,
                "doc_name": s.chunk.doc_name,
                "page": s.chunk.page,
                "score": round(s.score, 4),
                "rank": s.rank,
                "char_len": s.chunk.char_len,
            }
            for s in answer.sources
        ],
        "latency_ms": {k: round(v, 2) for k, v in answer.latency_ms.items()},
        "provider": answer.provider,
        "model": answer.model,
        "embedding_model": answer.embedding_model,
        "tokens": answer.tokens,
        "error": answer.error,
    }

    log_file = Path(cfg.monitoring.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    if answer.error:
        logger.error("Query failed: %s | %s", answer.question, answer.error)
    else:
        logger.info(
            "Query ok answered=%s conf=%s latency=%.0fms",
            answer.answered,
            answer.confidence.percent,
            answer.latency_ms.get("total_ms", 0.0),
        )
