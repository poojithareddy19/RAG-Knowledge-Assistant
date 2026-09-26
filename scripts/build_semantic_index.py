"""Rebuild the semantic layer from the current contents of the database.

    python scripts/build_semantic_index.py

Run after loading data. The summaries describe what is in the tables, so they
are stale the moment new profiles land, and nothing rebuilds them automatically.
"""

import sys
from pathlib import Path

# Run from a plain checkout. `python scripts/x.py` puts scripts/ on sys.path,
# not the repository, so `src` is not importable without this line.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.semantic.index import SemanticIndex

load_dotenv()


if __name__ == "__main__":
    index = SemanticIndex()

    counts = index.refresh()

    if not counts.get("subjects"):
        raise SystemExit(
            "no floats or regions found; load measurements first"
        )

    print(
        f"indexed {counts['subjects']} summaries "
        f"({counts.get('float', 0)} floats, "
        f"{counts.get('region', 0)} regions)"
    )
