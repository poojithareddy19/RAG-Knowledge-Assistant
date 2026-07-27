"""Evaluation harness.

Runs a gold dataset through the live retriever and reports retrieval quality
(Recall@K, Precision@K, MRR, Hit@K, nDCG@K). Relevance is matched at the
document+page level: a retrieved chunk counts as relevant if its
``"<doc>|<page>"`` key is in the question's gold set. This is robust to
chunk-boundary changes (re-chunking doesn't invalidate labels).

Generation metrics (faithfulness, groundedness, answer relevance) require an LLM
judge or Ragas and are implemented separately; this module covers the
retrieval half, which is fully deterministic and needs no API calls.

Gold dataset format (data/evaluation/questions.csv):
    question, expected_answer, relevant_documents, relevant_pages, ground_truth_passage
where relevant_documents and relevant_pages are ';'-separated and positionally
aligned (doc[i] is on page[i]).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Set

from src.evaluation.metrics import (
    aggregate,
    hit_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.retrieval.retriever import Retriever


def _gold_keys(docs: str, pages: str) -> Set[str]:
    doc_list = [d.strip() for d in docs.split(";") if d.strip()]
    page_list = [p.strip() for p in pages.split(";") if p.strip()]
    keys = set()
    for i, doc in enumerate(doc_list):
        page = page_list[i] if i < len(page_list) else "0"
        keys.add(f"{doc}|{page}")
    return keys


def load_gold(path: str | Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def evaluate_retrieval(
    retriever: Retriever, gold_path: str | Path, k: int = 5
) -> Dict:
    gold = load_gold(gold_path)
    per_question = []

    for row in gold:
        question = row["question"]
        relevant = _gold_keys(
            row.get("relevant_documents", ""), row.get("relevant_pages", "")
        )
        retrieved, _ = retriever.retrieve(question)
        retrieved_keys = [f"{rc.chunk.doc_name}|{rc.chunk.page}" for rc in retrieved]

        per_question.append(
            {
                "recall@k": recall_at_k(retrieved_keys, relevant, k),
                "precision@k": precision_at_k(retrieved_keys, relevant, k),
                "hit@k": hit_at_k(retrieved_keys, relevant, k),
                "mrr": mrr(retrieved_keys, relevant),
                "ndcg@k": ndcg_at_k(retrieved_keys, relevant, k),
            }
        )

    return {
        "k": k,
        "n_questions": len(gold),
        "aggregate": aggregate(per_question),
        "per_question": per_question,
    }
