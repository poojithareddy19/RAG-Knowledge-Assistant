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

from src.sqlgen.counting import ProfileOvercount, profiles_overcounted
from src.sqlgen.executor import run_query
from src.sqlgen.generator import generate_sql, repair_sql
from src.sqlgen.period import PeriodMismatch, period_problem
from src.sqlgen.schema_context import load_column_catalog
from src.sqlgen.scope import error_is_the_answer
from src.sqlgen.validator import PLATFORMS, SQLRejected, validate
from src.utils.config import get_config

ALLOWED = ["floats", "profiles", "measurements", "drifters", "drifter_observations"]

PLACES = 6


def _cell(value, places=PLACES):
    """Normalise one value so equal results compare equal."""
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, int | float | Decimal):
        return round(float(value), places)

    if isinstance(value, datetime | date):
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


def _catalog():
    """Real columns, read from the live database, or empty if unreachable.

    Empty means the repair guard cannot judge and lets the repair proceed,
    which is the behaviour that existed before the guard.
    """
    try:
        return load_column_catalog()
    except Exception:
        return {}


def _gen_timeout():
    """The wall clock the app allows a generation, read from the same config.

    The harness used the function default of 90 seconds while config.yaml gave
    the app 120. On hardware this slow that gap is not cosmetic: it turned
    generations that would have finished into ReadTimeout, which the harness
    scores as a refusal. A benchmark must not be stricter than the system it
    measures.
    """
    try:
        return int(get_config().sql.get("gen_timeout_seconds", 90))
    except Exception:
        return 90


def _is_refusal(sql):
    """Whether this "failure" is the system correctly declining.

    A refusal must never be repaired. The scope gate returns UNANSWERABLE for a
    quantity the schema does not hold, and a model may return it for a question
    it cannot answer; feeding either back as an error to be fixed asks for the
    fabrication the refusal exists to prevent.
    """
    return not sql or sql.strip().upper().startswith("UNANSWERABLE")


def evaluate_one(
    question,
    expected_sql=None,
    include_examples=True,
    use_cache=False,
    use_semantic=True,
    model=None,
    repair=True,
):
    """Generate, validate, execute and compare one question.

    ``model`` names the model to generate with, defaulting to whatever the
    environment selects. It is recorded on the row rather than only in a run
    header, because the point of the flag is to put two models side by side in
    one file and a header cannot survive a concatenation.
    """
    started = time.perf_counter()

    record = {
        "question": question,
        "model": model or _default_model(),
        "validated": False,
        "executed": False,
        "matched": False,
        "rows": 0,
        "error": None,
        "sql": None,
        "context_used": 0,
        # A repair that was tried and failed is not the same as one that was
        # never needed, and both look like a failed row without these.
        "repair_attempted": False,
        "repaired": False,
        "first_error": None,
        # A refusal is the safety path declining, whichever layer declined.
        # Kept separate from `validated` because the two came apart: the repair
        # guard declines after the validator has already passed the query, so a
        # question can be correctly refused with validated still true.
        "refused": False,
    }

    try:
        context = _context(question) if use_semantic else ""

        record["context_used"] = len(context.splitlines()) if context else 0

        raw = generate_sql(
            question,
            model=model,
            include_examples=include_examples,
            use_cache=use_cache,
            context=context,
            timeout=_gen_timeout(),
        )

        record["sql"] = raw

        try:
            safe = validate(raw, ALLOWED)
            record["validated"] = True

            # The same period check as the app, so the benchmark keeps
            # measuring the system people use.
            if problem := period_problem(question, safe):
                raise PeriodMismatch(problem)

            if overcount := profiles_overcounted(question, safe):
                raise ProfileOvercount(overcount)

            out = run_query(safe)
            record["executed"] = True
        except Exception as first:
            # One repair attempt, given the error the query failed with. A
            # refusal is never repaired: the model declining, or the scope
            # gate refusing a quantity the schema does not hold, is the
            # correct answer rather than a failure to work around.
            if not repair or _is_refusal(raw):
                raise

            # A failure can be the answer. If the column the query wanted does
            # not exist on the platform it reads, the schema has said the data
            # is not there, and handing that to a repair asks it to find
            # something else to put under the same alias. It did exactly that
            # once: asked how deep the buoys dived, the repair swapped sea
            # surface temperature in and kept the aliases min_pressure and
            # max_pressure.
            unrepairable = error_is_the_answer(
                raw,
                first,
                _catalog(),
                PLATFORMS,
            )

            if unrepairable:
                raise SQLRejected(unrepairable) from first

            record["repair_attempted"] = True
            record["first_error"] = f"{type(first).__name__}: {first}"

            raw = repair_sql(
                question,
                raw,
                str(first),
                model=model,
                context=context,
                include_examples=include_examples,
                timeout=_gen_timeout(),
            )

            record["sql"] = raw

            safe = validate(raw, ALLOWED)
            record["validated"] = True

            if problem := period_problem(question, safe):
                raise PeriodMismatch(problem) from first

            if overcount := profiles_overcounted(question, safe):
                raise ProfileOvercount(overcount) from first

            out = run_query(safe)
            record["executed"] = True
            record["repaired"] = True

        record["rows"] = out["row_count"]

        if isinstance(expected_sql, str) and expected_sql.strip():
            record["matched"] = results_match(
                out["rows"],
                _expected_rows(expected_sql),
            )

    except SQLRejected as exc:
        record["error"] = f"rejected: {exc}"
        record["refused"] = True

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
    model=None,
    repair=True,
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
            repair=repair,
            model=model,
        )

        record["bucket"] = row.get("bucket", "unknown")

        records.append(record)

        if progress:
            progress(position, len(gold), record)

    df = pd.DataFrame(records)

    answerable = df[df["bucket"] != "unanswerable"]
    unanswerable = df[df["bucket"] == "unanswerable"]

    summary = {
        "model": model or _default_model(),
        "n": len(df),
        "n_answerable": len(answerable),
        # The headline number: right answer, not merely a query that ran.
        "execution_accuracy": _rate(answerable["matched"]),
        "validation_pass_rate": _rate(answerable["validated"]),
        "execution_rate": _rate(answerable["executed"]),
        "false_refusal_rate": _rate(_refused(answerable)),
        "correct_refusal_rate": _rate(_refused(unanswerable)),
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


def _default_model() -> str:
    """Whatever the generator would pick if nobody named a model."""
    import os

    return os.environ.get("GENERATION__MODEL", "llama3.1:latest")


def _refused(frame):
    """Whether each row was declined by the safety path.

    This was `~validated`, which read the validator's verdict as the whole
    answer. That stopped being true when the repair guard landed: it declines
    after the validator has already passed a query, because the refusal comes
    from the execution error rather than from the SQL's shape. A question
    refused that way was being scored as answered.

    Falls back to the old reading for result files written before the flag
    existed, so an old CSV still summarises rather than raising.
    """
    if "refused" in frame.columns:
        return frame["refused"].astype(bool)

    return ~frame["validated"].astype(bool)


def _rate(series):
    return round(float(series.mean()), 3) if len(series) else None


def evaluate_runs(
    runs=1,
    path="data/evaluation/ocean_questions.csv",
    include_examples=True,
    use_cache=False,
    use_semantic=True,
    progress=None,
    model=None,
    repair=True,
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
            repair=repair,
            progress=progress,
            model=model,
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

    keys = [key for key in summaries[0] if key not in ("run", "model")]

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
        "--model",
        default=None,
        help="generate with this model instead of the configured one",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="reuse cached SQL, which makes an interrupted single run resumable",
    )
    parser.add_argument(
        "--no-repair",
        action="store_true",
        help=(
            "do not give a failed query one more attempt with the error it "
            "failed with"
        ),
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
        model=args.model,
        repair=not args.no_repair,
    )

    print()
    print(f"model: {summaries[0]['model']}")

    if args.runs == 1:
        # Two back-to-back runs of an identical configuration disagreed on
        # four of 46 questions on this hardware, a swing of 0.087 available to
        # any single run. Printing a bare number invites a comparison it
        # cannot support, so the caveat travels with it rather than living in
        # a README paragraph nobody reads next to the figure.
        print(
            "\nONE RUN. Temperature 0 is not reproducible here: identical "
            "configurations have disagreed on four of 46 questions. Treat a "
            "difference smaller than about a tenth as noise, and use --runs 3 "
            "for anything you intend to publish."
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
