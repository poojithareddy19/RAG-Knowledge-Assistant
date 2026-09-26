"""Smoke-test the routed query path end to end.

Runs a handful of questions through RAGService, printing the route taken, the
SQL that was generated and whether it executed. Chart questions write their PNG
to data/processed/ so the plot can be inspected without starting the app.

Usage:
    python -m scripts.verify_routes
"""

import sys
from pathlib import Path

# Run from a plain checkout. `python scripts/x.py` puts scripts/ on sys.path,
# not the repository, so `src` is not importable without this line.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pathlib
import time

from src.utils.pipeline import RAGService

QUESTIONS = [
    # (question, expected_route)
    ("What is the average surface temperature per year in the Arabian Sea?", "data"),
    ("How many floats reported in the Bay of Bengal?", "data"),
    ("Plot the monthly mean temperature in the Southern Indian Ocean for 2020", "chart"),
    ("Delete everything from the measurements table", None),
]

OUT_DIR = pathlib.Path("data/processed")


def main() -> int:
    service = RAGService()
    failures = 0

    for question, expected in QUESTIONS:
        print("=" * 70)
        print(f"Q: {question}")

        started = time.perf_counter()

        try:
            result = service.answer(question)
        except Exception as exc:  # noqa: BLE001 - smoke test reports, never raises
            print(f"  EXCEPTION: {type(exc).__name__}: {exc}")
            failures += 1
            continue

        wall_ms = round((time.perf_counter() - started) * 1000, 1)

        route = result.get("route")
        print(f"  route            : {route} (by {result.get('route_decided_by')})")
        print(f"  sql_cached       : {result.get('sql_cached')}")

        if expected and route != expected:
            print(f"  ROUTE MISMATCH   : expected {expected}")
            failures += 1

        if result.get("generated_sql"):
            print("  sql              :")
            for line in result["generated_sql"].splitlines():
                print(f"      {line}")

        print(f"  answered         : {result.get('answered')}")
        print(f"  answer           : {result.get('answer')}")

        if result.get("row_count") is not None:
            print(f"  rows             : {result['row_count']}")
            print(f"  columns          : {result.get('columns')}")
            for row in (result.get("rows") or [])[:5]:
                print(f"      {row}")

        if result.get("error"):
            print(f"  error            : {result['error']}")

        png = result.get("chart_png")
        if png:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            path = OUT_DIR / "verify_chart.png"
            path.write_bytes(png)
            print(f"  chart            : {result.get('chart_kind')}, {len(png)} bytes -> {path}")
        elif route == "chart":
            print(f"  chart            : none produced (kind={result.get('chart_kind')})")

        # result["elapsed_ms"] is the database time on the success path, so
        # report the measured wall clock alongside it.
        print(f"  elapsed_ms       : {result.get('elapsed_ms')} (wall {wall_ms})")

    print("=" * 70)
    print(f"{'FAILURES: ' + str(failures) if failures else 'all checks passed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
