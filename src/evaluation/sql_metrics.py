# src/evaluation/sql_metrics.py

import statistics
import time

import pandas as pd

from src.sqlgen.executor import run_query
from src.sqlgen.generator import generate_sql
from src.sqlgen.validator import SQLRejected, validate

ALLOWED = ["floats", "profiles", "measurements"]


def evaluate_one(question, include_examples=True):
    started = time.perf_counter()

    record = {
        "question": question,
        "validated": False,
        "executed": False,
        "rows": 0,
        "error": None,
        "sql": None,
    }

    try:
        raw = generate_sql(
            question,
            include_examples=include_examples,
        )

        record["sql"] = raw

        safe = validate(raw, ALLOWED)

        record["validated"] = True

        out = run_query(safe)

        record["executed"] = True
        record["rows"] = out["row_count"]

    except SQLRejected as exc:
        record["error"] = f"rejected: {exc}"

    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"

    record["latency_ms"] = round(
        (time.perf_counter() - started) * 1000,
        1,
    )

    return record


def evaluate_file(
    path="data/evaluation/ocean_questions.csv",
    include_examples=True,
):
    gold = pd.read_csv(path)

    records = []

    for _, row in gold.iterrows():
        rec = evaluate_one(
            row["question"],
            include_examples,
        )

        rec["bucket"] = row.get(
            "bucket",
            "unknown",
        )

        records.append(rec)

    df = pd.DataFrame(records)

    answerable = df[df["bucket"] != "unanswerable"]
    unanswerable = df[df["bucket"] == "unanswerable"]

    summary = {
        "n": len(df),
        "validation_pass_rate": round(
            answerable["validated"].mean(),
            3,
        ),
        "execution_accuracy": round(
            answerable["executed"].mean(),
            3,
        ),
        "non_empty_rate": round(
            (answerable["rows"] > 0).mean(),
            3,
        ),
        "correct_refusals": (
            round(
                (~unanswerable["validated"]).mean(),
                3,
            )
            if len(unanswerable)
            else None
        ),
        "false_refusals": round(
            (~answerable["validated"]).mean(),
            3,
        ),
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
        df.groupby("bucket")[["validated", "executed"]]
        .mean()
        .round(3)
        .to_dict("index")
    )

    return summary, by_bucket, df


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    summary, buckets, detail = evaluate_file()

    print(summary)
    print(buckets)

    detail.to_csv(
        "data/evaluation/sql_eval_results.csv",
        index=False,
    )