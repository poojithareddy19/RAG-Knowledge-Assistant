"""Generation evaluation for the summaries route.

Retrieval is scored with hit@5, recall and nDCG in ``evaluator.py``.
Text-to-SQL is scored by executing the generated query against a reference in
``sql_metrics.py``. This module scores the **prose**: whether the sentence
handed to a user follows from the summaries it cites.

That is the half the project's own claim rests on. "Being confidently wrong is
worse than saying nothing" is a statement about generated text, and without
this the only evidence for it would be that retrieval found the right float.

Four metrics, in two groups.

**Deterministic, no model call.** These cost nothing and cannot drift:

- ``answered_rate`` - how often the system answers rather than declining.
  Refusing everything scores perfectly on faithfulness, so no faithfulness
  number means anything without this beside it.
- ``citation_accuracy`` - of the answers given, how many were written from the
  float or region the gold set names. An answer about the Arabian Sea built
  from the Bay of Bengal's summary is not grounded, and this is checkable
  without asking a model anything.
- ``lexical_support`` - the share of the answer's content words that appear in
  the summaries it was given. A **proxy** rather than a measure: paraphrase
  scores low and fluent copying scores high. Useful as a floor and reported as
  one, because an answer scoring near zero here is not reading its sources.

**Judged, one model call per question, opt-in.** Local and slow, which is why
it is behind a flag:

- ``faithfulness`` - is every claim in the answer supported by the summaries?
- ``answer_relevance`` - does the answer address the question asked?
- ``correctness`` - does it agree with the gold answer?

The judge is the same local model that wrote the answer, which is the honest
limit of this file: a model is a poor judge of its own output and shares its
own blind spots. Treat a judged score as a smoke alarm rather than a
measurement, and read ``citation_accuracy`` first, which nothing can talk its
way out of.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from src.evaluation.evaluator import gold_keys, subject_key

GOLD = "data/evaluation/summary_questions.csv"

# Words carrying no evidence either way. Kept short on purpose: this is a
# floor, and a longer list would flatter the score.
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "then", "there", "these", "this", "to", "was", "were", "which", "with",
}

_WORD = re.compile(r"[a-z0-9_]+")


def _content_words(text: str) -> set[str]:
    return {
        w
        for w in _WORD.findall((text or "").lower())
        if w not in _STOP and len(w) > 2
    }


def load_gold(path: str | Path = GOLD) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _gold_subjects(row: dict) -> set[str]:
    """The ``kind:subject`` keys a gold row says the answer must rest on."""
    return gold_keys(row.get("relevant_subjects") or "")


def _cited_subjects(sources) -> set[str]:
    """The subjects of the summaries an answer was written from."""
    out = set()

    for source in sources or []:
        kind = getattr(source, "kind", None)
        subject = getattr(source, "subject", None)

        if kind is None or subject is None:
            continue

        out.add(subject_key(str(kind), str(subject)))

    return out


def _lexical_support(answer: str, sources) -> float:
    """Share of the answer's content words that occur in its summaries."""

    words = _content_words(answer)

    if not words:
        return 0.0

    passage = " ".join(getattr(s, "text", "") or "" for s in sources or [])

    return len(words & _content_words(passage)) / len(words)


JUDGE_SYSTEM = (
    "You grade answers against the summaries they were written from. You "
    "answer with one word and nothing else."
)

_JUDGE = {
    "faithfulness": (
        "Is every factual claim in the ANSWER supported by the PASSAGES?\n"
        "Answer YES or NO."
    ),
    "answer_relevance": (
        "Does the ANSWER address the QUESTION that was asked?\n"
        "Answer YES or NO."
    ),
    "correctness": (
        "Does the ANSWER agree with the REFERENCE ANSWER?\n"
        "Answer YES or NO."
    ),
}


def _judge(llm, aspect: str, question: str, answer: str, passages: str, reference: str) -> bool | None:
    prompt = (
        f"{_JUDGE[aspect]}\n\n"
        f"=== QUESTION ===\n{question}\n\n"
        f"=== PASSAGES ===\n{passages[:4000]}\n\n"
        f"=== ANSWER ===\n{answer}\n\n"
        f"=== REFERENCE ANSWER ===\n{reference}\n"
    )

    try:
        text = llm.generate(JUDGE_SYSTEM, prompt).text.strip().upper()
    except Exception:
        # A judge that cannot be reached is not a failing grade. Recorded as
        # unknown and excluded from the mean, so an outage cannot look like a
        # quality drop.
        return None

    if text.startswith("YES"):
        return True

    if text.startswith("NO"):
        return False

    return None


def evaluate_generation(
    service=None,
    gold_path: str | Path = GOLD,
    judge: bool = False,
    progress=None,
    limit: int | None = None,
) -> dict:
    """Score the answers the summaries route produces for the gold questions."""

    from src.utils.pipeline import RAGService

    service = service or RAGService()
    llm = None

    if judge:
        from src.generation.llm import get_llm

        llm = get_llm()

    gold = load_gold(gold_path)

    if limit:
        gold = gold[:limit]

    rows = []

    for i, row in enumerate(gold, 1):
        question = row["question"]
        answer_obj = service.ask(question)

        sources = getattr(answer_obj, "sources", []) or []
        answered = bool(getattr(answer_obj, "answered", False))
        answer = getattr(answer_obj, "answer", "") or ""

        wanted = _gold_subjects(row)
        cited = _cited_subjects(sources)

        record = {
            "question": question,
            "answered": answered,
            # Only meaningful where an answer was given: an abstention cites
            # nothing and should not be scored as a wrong citation.
            "citation_correct": bool(wanted & cited) if answered else None,
            "lexical_support": round(_lexical_support(answer, sources), 4) if answered else None,
            "confidence": float(getattr(getattr(answer_obj, "confidence", None), "score", 0.0) or 0.0),
            "answer": answer,
        }

        if judge and answered:
            passages = "\n\n".join(getattr(s, "text", "") or "" for s in sources)

            for aspect in _JUDGE:
                record[aspect] = _judge(
                    llm,
                    aspect,
                    question,
                    answer,
                    passages,
                    row.get("expected_answer", ""),
                )

        rows.append(record)

        if progress:
            progress(i, len(gold), record)

    return {"n": len(rows), "aggregate": summarise(rows), "per_question": rows}


def summarise(rows: list[dict]) -> dict:
    """Aggregate, skipping rows where a metric does not apply.

    A metric is averaged over the questions it applies to rather than over all
    of them, and the denominator is reported beside it. Scoring an abstention
    as an unfaithful answer would reward answering badly over declining, which
    is the opposite of what this system is built to do.
    """
    n = len(rows) or 1
    answered = [r for r in rows if r["answered"]]

    out = {
        "n": len(rows),
        "answered_rate": round(len(answered) / n, 4),
        "n_answered": len(answered),
    }

    for key in ("citation_correct", "faithfulness", "answer_relevance", "correctness"):
        judged = [r[key] for r in rows if r.get(key) is not None]

        if judged:
            out[key] = round(sum(1 for v in judged if v) / len(judged), 4)
            out[f"n_{key}"] = len(judged)

    support = [r["lexical_support"] for r in rows if r.get("lexical_support") is not None]

    if support:
        out["lexical_support"] = round(sum(support) / len(support), 4)

    return out


def _print_progress(i, total, record):
    mark = "ANSWERED" if record["answered"] else "DECLINED"
    cite = record.get("citation_correct")
    cite_mark = "-" if cite is None else ("cited" if cite else "MISCITED")
    print(f"[{i:2d}/{total}] {mark:9s} {cite_mark:9s} {record['question'][:52]}", flush=True)


def main(argv=None) -> int:
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        description="Score the generated answers on the summaries route."
    )
    parser.add_argument("--questions", default=GOLD)
    parser.add_argument("--out", default="data/evaluation/generation_results.csv")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="score only the first N questions, for a quick check",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help=(
            "also ask the local model to grade faithfulness, relevance and "
            "correctness, at one call per question"
        ),
    )
    args = parser.parse_args(argv)

    load_dotenv()

    result = evaluate_generation(
        gold_path=args.questions,
        judge=args.judge,
        progress=_print_progress,
        limit=args.limit,
    )

    print()

    for key, value in result["aggregate"].items():
        print(f"{key:24s} {value}")

    import pandas as pd

    pd.DataFrame(result["per_question"]).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
