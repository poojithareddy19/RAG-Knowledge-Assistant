"""Text-to-SQL evaluation.

The metric that matters is **execution accuracy**: run the generated query and
the hand-written reference query, and check they return the same rows. Whether
a query parses, or returns something, says nothing about whether it answered
the question. A query can validate, execute, return 21 tidy rows and still be
wrong.

Comparing results rather than SQL text is deliberate. There are many correct
ways to write the same query, and string comparison would fail all but one of
them. Comparing against a reference dataset also keeps the gold set portable:
the expected answers are recomputed from whatever data is loaded, so nothing
has to be re-labelled when the database changes.

Rows are compared as a multiset with floats rounded, so column order and row
order do not matter but values and shape do.

Writing a question for the gold set
-----------------------------------

Three rules, each learned from a question that scored a model wrong for the
question's fault rather than the model's:

1. **A superlative needs a unique winner in the data.** "Which float recorded
   the most profiles?" is unanswerable when every float has exactly 730. The
   reference picks one arbitrarily, the model picks another, and both are
   right. Check for ties before adding a "most", "top" or "first" question.

2. **The question must imply one result shape.** Comparison is shape
   sensitive, so "which float is warmest" is a bad question when the answer
   could reasonably be an id or an id and a temperature. Ask for the id.

3. **One reading only.** "Average surface temperature for each calendar
   month" means twelve seasonal averages to one reader and a monthly time
   series to another. Say which.

Fixing a question because the model got it wrong is tuning the benchmark.
Fixing one because it has no single right answer is maintenance. Only the
second is allowed.
"""

from __future__ import annotations

import argparse
import statistics
import time
from datetime import date, datetime
from decimal import Decimal

import pandas as pd

from src.sqlgen.executor import run_query
from src.sqlgen.generator import generate_sql
from src.sqlgen.validator import SQLRejected, validate

ALLOWED = ["floats", "profiles", "measurements"]

PLACES = 6


def _cell(value, places=PLACES):
    """Normalise one value so equal results compare equal."""
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float, Decimal)):
        return round(float(value), places)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    return str(value)


def _normalise(rows, places=PLACES):
    """Rows as a sorted list of tuples, so ordering does not affect equality."""
    normalised = [
        tuple(_cell(value, places) for value in row) for row in rows
    ]

    return sorted(normalised, key=repr)


def results_match(actual, expected, places=PLACES) -> bool:
    """Do two result sets carry the same values?

    Order-insensitive, because a question rarely specifies one and the
    reference query's ORDER BY is a presentation choice. Shape-sensitive: a
    query returning an extra column has not answered the same question.
    """
    if actual is None or expected is None:
        return False

    if len(actual) != len(expected):
        return False

    if actual and expected and len(actual[0]) != len(expected[0]):
        return False

    return _normalise(actual, places) == _normalise(expected, places)


def _expected_rows(expected_sql):
    """Run the reference query through the same validation as a generated one.

    The validator caps a query with a LIMIT, so skipping it here would make
    long results differ for a reason that has nothing to do with correctness.
    """
    return run_query(
        validate(expected_sql, ALLOWED),
        timeout_ms=60_000,
    )["rows"]


def evaluate_one(
    question,
    expected_sql=None,
    include_examples=True,
    use_cache=False,
    use_semantic=True,
):
    """Generate, validate, execute and compare one question."""
    started = time.perf_counter()

    record = {
        "question": question,
        "validated": False,
        "executed": False,
        "matched": False,
        "rows": 0,
        "error": None,
        "sql": None,
        "context_used": 0,
    }

    try:
        context = _context(question) if use_semantic else ""

        record["context_used"] = len(context.splitlines()) if context else 0

        raw = generate_sql(
            question,
            include_examples=include_examples,
            use_cache=use_cache,
            context=context,
        )

        record["sql"] = raw

        safe = validate(raw, ALLOWED)

        record["validated"] = True

        out = run_query(safe)

        record["executed"] = True
        record["rows"] = out["row_count"]

        if isinstance(expected_sql, str) and expected_sql.strip():
            record["matched"] = results_match(
                out["rows"],
                _expected_rows(expected_sql),
            )

    except SQLRejected as exc:
        record["error"] = f"rejected: {exc}"

    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"

    record["latency_ms"] = round(
        (time.perf_counter() - started) * 1000,
        1,
    )

    return record


def _context(question: str) -> str:
    """Semantic context for a question, or empty if the layer is unavailable.

    Evaluating without it would measure a system nobody runs.
    """
    try:
        from src.semantic.index import SemanticIndex, as_context

        return as_context(SemanticIndex().search(question))
    except Exception:
        return ""


def evaluate_file(
    path="data/evaluation/ocean_questions.csv",
    include_examples=True,
    use_cache=False,
    use_semantic=True,
    progress=None,
):
    """Score every question in the gold set.

    ``progress`` is called with ``(index, total, record)`` after each question.
    A full run is dominated by model latency and takes tens of minutes, so a
    caller that reports nothing until the end is unusable.
    """
    gold = pd.read_csv(path)

    records = []

    for position, (_, row) in enumerate(gold.iterrows(), start=1):
        record = evaluate_one(
            row["question"],
            expected_sql=row.get("expected_sql"),
            include_examples=include_examples,
            use_cache=use_cache,
            use_semantic=use_semantic,
        )

        record["bucket"] = row.get("bucket", "unknown")

        records.append(record)

        if progress:
            progress(position, len(gold), record)

    df = pd.DataFrame(records)

    answerable = df[df["bucket"] != "unanswerable"]
    unanswerable = df[df["bucket"] == "unanswerable"]

    summary = {
        "n": len(df),
        "n_answerable": len(answerable),
        # The headline number: right answer, not merely a query that ran.
        "execution_accuracy": _rate(answerable["matched"]),
        "validation_pass_rate": _rate(answerable["validated"]),
        "execution_rate": _rate(answerable["executed"]),
        "false_refusal_rate": _rate(~answerable["validated"]),
        "correct_refusal_rate": _rate(~unanswerable["validated"]),
        "median_latency_ms": round(
            statistics.median(df["latency_ms"]),
            1,
        ),
        "p95_latency_ms": round(
            df["latency_ms"].quantile(0.95),
            1,
        ),
    }

    by_bucket = (
        df.groupby("bucket")[["validated", "executed", "matched"]]
        .mean()
        .round(3)
        .to_dict("index")
    )

    return summary, by_bucket, df


def _rate(series):
    return round(float(series.mean()), 3) if len(series) else None


def evaluate_runs(
    runs=1,
    path="data/evaluation/ocean_questions.csv",
    include_examples=True,
    use_cache=False,
    use_semantic=True,
    progress=None,
):
    """Score the gold set ``runs`` times and keep every run's numbers.

    One run cannot separate a real change from sampling noise. The model is
    sampled rather than greedy, so the same question can be answered correctly
    at one o'clock and wrongly at two, and a single figure quoted to three
    decimal places reads as more certain than it is.

    The SQL cache is off by default here for the same reason: a cached run
    replays the first run's generated SQL, so every repeat would agree and the
    spread would be zero by construction rather than by measurement.
    """
    summaries = []
    buckets = []
    frames = []

    for run in range(1, runs + 1):
        summary, by_bucket, detail = evaluate_file(
            path=path,
            include_examples=include_examples,
            use_cache=use_cache,
            use_semantic=use_semantic,
            progress=progress,
        )

        summary["run"] = run
        detail["run"] = run

        summaries.append(summary)
        buckets.append(by_bucket)
        frames.append(detail)

    return summaries, buckets, pd.concat(frames, ignore_index=True)


def mean_sd(values) -> tuple[float | None, float | None]:
    """Mean and sample standard deviation, ignoring metrics that were None.

    A single run has no spread to report rather than a spread of zero, so the
    deviation is None until there are two runs to compare.
    """
    clean = [float(value) for value in values if value is not None]

    if not clean:
        return None, None

    mean = round(statistics.fmean(clean), 3)

    if len(clean) < 2:
        return mean, None

    return mean, round(statistics.stdev(clean), 3)


def summarise_runs(summaries) -> dict[str, tuple[float | None, float | None]]:
    """Each headline metric as ``(mean, standard deviation)`` over the runs."""
    if not summaries:
        return {}

    keys = [key for key in summaries[0] if key != "run"]

    return {
        key: mean_sd([summary.get(key) for summary in summaries])
        for key in keys
    }


def summarise_buckets(buckets) -> dict[str, dict[str, tuple]]:
    """Per bucket metrics as ``(mean, standard deviation)`` over the runs."""
    names = sorted({name for run in buckets for name in run})

    out = {}

    for name in names:
        scores = [run[name] for run in buckets if name in run]
        metrics = sorted({key for score in scores for key in score})

        out[name] = {
            metric: mean_sd([score.get(metric) for score in scores])
            for metric in metrics
        }

    return out


def _format(mean, sd, places=3) -> str:
    """One metric as ``mean +/- sd``, or just the mean when there is one run."""
    if mean is None:
        return "n/a"

    if sd is None:
        return f"{mean:.{places}f}"

    return f"{mean:.{places}f} +/- {sd:.{places}f}"


def _print_progress(position, total, record):
    if record["bucket"] == "unanswerable":
        verdict = "REFUSED  " if not record["validated"] else "ANSWERED!"
    else:
        verdict = "MATCH    " if record["matched"] else "WRONG    "

    print(
        f"[{position:2d}/{total}] {verdict} "
        f"{record['latency_ms'] / 1000:5.1f}s  "
        f"{record['question'][:58]}",
        flush=True,
    )


def main(argv=None) -> int:
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="Score the text-to-SQL gold set.")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="run the whole set this many times and report mean and spread",
    )
    parser.add_argument(
        "--questions",
        default="data/evaluation/ocean_questions.csv",
        help="gold set to score",
    )
    parser.add_argument(
        "--out",
        default="data/evaluation/sql_eval_results.csv",
        help="where to write the per question detail",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="reuse cached SQL, which makes an interrupted single run resumable",
    )
    args = parser.parse_args(argv)

    load_dotenv()

    if args.cache and args.runs > 1:
        # Every repeat would replay the first run's SQL and agree with it, so
        # the reported deviation would describe the cache, not the model.
        print("ignoring --cache: repeated runs must generate SQL each time")

    use_cache = args.cache and args.runs == 1

    summaries, buckets, detail = evaluate_runs(
        runs=args.runs,
        path=args.questions,
        use_cache=use_cache,
        progress=_print_progress,
    )

    print()

    for key, (mean, sd) in summarise_runs(summaries).items():
        places = 1 if key.endswith("_ms") else 3
        print(f"{key:24s} {_format(mean, sd, places)}")

    print()

    for bucket, scores in sorted(summarise_buckets(buckets).items()):
        line = "  ".join(
            f"{metric} {_format(mean, sd)}"
            for metric, (mean, sd) in scores.items()
        )

        print(f"{bucket:16s} {line}")

    detail.to_csv(args.out, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
