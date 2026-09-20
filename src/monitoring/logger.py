"""Structured interaction logging.

Every query produces one JSON object appended to
``logs/interactions.jsonl``.
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
    """Configure and return the application logger."""

    global _configured

    cfg = get_config()
    log_dir = Path(cfg.paths.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("rag")

    if not _configured:
        logger.setLevel(
            getattr(
                logging,
                cfg.monitoring.level,
                logging.INFO,
            )
        )

        log_path = str(log_dir / "app.log")

        handler = logging.FileHandler(
            log_path,
            encoding="utf-8",
        )

        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"
            )
        )

        logger.addHandler(handler)
        _configured = True

    return logger


def log_interaction(answer: Answer) -> None:
    """Append one structured record for a completed document query."""

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
                "chunk_id": source.chunk.chunk_id,
                "doc_name": source.chunk.doc_name,
                "page": source.chunk.page,
                "score": round(source.score, 4),
                "rank": source.rank,
                "char_len": source.chunk.char_len,
            }
            for source in answer.sources
        ],
        "latency_ms": {
            key: round(value, 2)
            for key, value in answer.latency_ms.items()
        },
        "provider": answer.provider,
        "model": answer.model,
        "embedding_model": answer.embedding_model,
        "tokens": answer.tokens,
        "error": answer.error,
    }

    log_file = Path(cfg.monitoring.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

    if answer.error:
        logger.error(
            "Query failed: %s | %s",
            answer.question,
            answer.error,
        )
    else:
        logger.info(
            "Query ok answered=%s conf=%s latency=%.0fms",
            answer.answered,
            answer.confidence.percent,
            answer.latency_ms.get("total_ms", 0.0),
        )


def log_result(
    question: str,
    result: dict[str, Any],
) -> None:
    """Append one structured record for a routed query."""

    cfg = get_config()
    logger = _ensure_logging()

    record: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": question,
        "answer": result.get("answer", ""),
        "answered": bool(
            result.get(
                "answered",
                not result.get("refused", False),
            )
        ),
        "confidence": result.get("confidence"),
        "reason": result.get("reason", ""),
        "route": result.get("route"),
        "question_rewritten": result.get("question_rewritten"),
        "route_decided_by": result.get("route_decided_by"),
        "generated_sql": result.get("generated_sql"),
        "sql_rejected": (
            bool(result.get("refused"))
            and result.get("generated_sql") is not None
        ),
        "sql_cached": result.get("sql_cached"),
        "row_count": result.get("row_count"),
        "elapsed_ms": result.get("elapsed_ms"),
        "db_elapsed_ms": result.get("db_elapsed_ms"),
        "latency_ms": result.get("latency_ms", {}),
        "retrieved": result.get("retrieved", []),
        "provider": result.get("provider", ""),
        "model": result.get("model", ""),
        "embedding_model": result.get(
            "embedding_model",
            "",
        ),
        "error": result.get("error"),
    }

    log_file = Path(cfg.monitoring.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )

    logger.info(
        "Query ok route=%s decided_by=%s answered=%s",
        record["route"],
        record["route_decided_by"],
        record["answered"],
    )