"""Analytics over the interaction log.

Reads ``logs/interactions.jsonl`` and computes the aggregates the monitoring
dashboard displays: latency, confidence distribution, fallback rate, most-queried
documents, error counts. Returns plain dicts / a DataFrame so the UI layer stays
free of business logic.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.config import get_config


def load_records() -> list[dict[str, Any]]:
    path = Path(get_config().monitoring.log_file)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def summarize(records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    records = records if records is not None else load_records()
    n = len(records)
    if n == 0:
        return {"total_queries": 0}

    latencies = [r.get("latency_ms", {}).get("total_ms", 0.0) for r in records]
    llm_latencies = [
        r["latency_ms"]["llm_ms"] for r in records if "llm_ms" in r.get("latency_ms", {})
    ]
    embed_latencies = [
        r["latency_ms"]["embed_ms"]
        for r in records
        if "embed_ms" in r.get("latency_ms", {})
    ]
    confidences = [r.get("confidence", 0) for r in records]
    declined = sum(1 for r in records if not r.get("answered", False))
    errors = sum(1 for r in records if r.get("error"))

    doc_counter: Counter = Counter()
    for r in records:
        for s in r.get("retrieved", []):
            doc_counter[s.get("doc_name", "?")] += 1

    avg_retrieval_score = _mean(
        [s["score"] for r in records for s in r.get("retrieved", [])]
    )
    avg_answer_len = _mean([len(r.get("answer", "")) for r in records])

    return {
        "total_queries": n,
        "avg_latency_ms": round(_mean(latencies), 1),
        "avg_llm_latency_ms": round(_mean(llm_latencies), 1),
        "avg_embed_latency_ms": round(_mean(embed_latencies), 1),
        "avg_confidence": round(_mean(confidences), 1),
        "fallback_rate": round(declined / n, 3),
        "fallback_count": declined,
        "error_count": errors,
        "avg_retrieval_score": round(avg_retrieval_score, 4),
        "avg_answer_len_chars": round(avg_answer_len, 1),
        "most_queried_documents": doc_counter.most_common(10),
        "confidence_values": confidences,
        "latency_values": latencies,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
