"""Does adding example queries to the prompt improve SQL accuracy?"""

import json

import pandas as pd
from dotenv import load_dotenv

from src.evaluation.ab_test import required_pairs, summarise
from src.evaluation.sql_metrics import evaluate_one

load_dotenv()


def main(path="data/evaluation/ocean_questions.csv"):
    gold = pd.read_csv(path)
    gold = gold[gold["bucket"] != "unanswerable"]

    needed = required_pairs(
        baseline_rate=0.60,
        expected_lift=0.15,
    )

    print(
        f"power analysis: {needed} paired questions needed; "
        f"gold set has {len(gold)}"
    )

    if len(gold) < needed:
        print(
            "WARNING: underpowered. A null result will be inconclusive, "
            "not evidence of no effect."
        )

    arm_a = []
    arm_b = []

    for _, row in gold.iterrows():
        q = row["question"]

        a = evaluate_one(
            q,
            include_examples=False,
        )

        b = evaluate_one(
            q,
            include_examples=True,
        )

        arm_a.append(
            a["executed"] and a["rows"] > 0
        )

        arm_b.append(
            b["executed"] and b["rows"] > 0
        )

        print(
            f"{'A' if arm_a[-1] else '-'}"
            f"{'B' if arm_b[-1] else '-'}  "
            f"{q[:60]}"
        )

    result = summarise(
        arm_a,
        arm_b,
        labels=("no_examples", "with_examples"),
    )

    print(json.dumps(result, indent=2))

    with open(
        "data/evaluation/ab_result.json",
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()