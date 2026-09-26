"""Retrieval evaluation for the summaries route.

Runs a gold dataset through the live semantic index and reports retrieval
quality (Recall@K, Precision@K, MRR, Hit@K, nDCG@K). Relevance is matched at
the subject level: a retrieved summary counts as relevant if its
``"<kind>:<subject>"`` key is in the question's gold set, so the wording of a
summary can change without invalidating the labels.

Generation metrics (citation accuracy, lexical support, and the judged
faithfulness, relevance and correctness) live in ``generation_metrics.py``.
This module covers the retrieval half, which is deterministic and needs no
model call.

Gold dataset format (data/evaluation/summary_questions.csv):
    question, expected_answer, relevant_subjects
where relevant_subjects is a ';'-separated list of ``kind:subject`` keys such
as ``float:1901393;region:Arabian Sea``. A row with no relevant subjects is a
question the summaries cannot answer. It is kept in the set so the generation
metrics can check that it is declined, and left out of the retrieval scores,
where there is nothing for retrieval to find.
"""
from __future__ import annotations

import csv
from pathlib import Path

from src.evaluation.metrics import (
    aggregate,
    hit_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

GOLD = "data/evaluation/summary_questions.csv"


def gold_keys(subjects: str) -> set[str]:
    """The ``kind:subject`` keys a gold row names, whitespace forgiven."""
    keys = set()

    for item in (subjects or "").split(";"):
        item = item.strip()

        if not item:
            continue

        kind, _, subject = item.partition(":")
        keys.add(f"{kind.strip()}:{subject.strip()}")

    return keys


def subject_key(kind: str, subject: str) -> str:
    return f"{kind}:{subject}"


def load_gold(path: str | Path = GOLD) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def evaluate_retrieval(index, gold_path: str | Path = GOLD, k: int = 5) -> dict:
    """Score the semantic index against the gold set.

    ``index`` is anything with ``search(question, k)`` returning ranked
    summaries, which is the SemanticIndex in production and a fake in tests.
    """
    gold = load_gold(gold_path)
    per_question = []
    skipped = 0

    for row in gold:
        question = row["question"]
        relevant = gold_keys(row.get("relevant_subjects", ""))

        if not relevant:
            skipped += 1
            continue

        retrieved = index.search(question, k=k)
        retrieved_keys = [subject_key(hit.kind, hit.subject) for hit in retrieved]

        per_question.append(
            {
                "question": question,
                "recall@k": recall_at_k(retrieved_keys, relevant, k),
                "precision@k": precision_at_k(retrieved_keys, relevant, k),
                "hit@k": hit_at_k(retrieved_keys, relevant, k),
                "mrr": mrr(retrieved_keys, relevant),
                "ndcg@k": ndcg_at_k(retrieved_keys, relevant, k),
            }
        )

    metrics = [
        {key: value for key, value in row.items() if key != "question"}
        for row in per_question
    ]

    return {
        "k": k,
        "n_questions": len(per_question),
        "n_unanswerable": skipped,
        "aggregate": aggregate(metrics),
        "per_question": per_question,
    }


def main(argv=None) -> int:
    import argparse

    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        description="Score summary retrieval against the gold set."
    )
    parser.add_argument("--questions", default=GOLD)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)

    load_dotenv()

    from src.semantic.index import SemanticIndex

    report = evaluate_retrieval(SemanticIndex(), args.questions, k=args.k)

    for row in report["per_question"]:
        mark = "hit " if row["hit@k"] else "MISS"
        print(f"{mark}  mrr={row['mrr']:.2f}  {row['question'][:70]}")

    print()
    print(
        f"n={report['n_questions']} k={report['k']} "
        f"(plus {report['n_unanswerable']} unanswerable, not scored here)"
    )

    for key, value in report["aggregate"].items():
        print(f"{key:14s} {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
