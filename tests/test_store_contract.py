"""The two vector store backends must be interchangeable.

config.yaml has a vectorstore.backend switch, and the retriever, pipeline
and UI all assume they cannot tell which backend is behind it. This test
fails if PgVectorStore drifts away from the FaissVectorStore contract.

Parsed with ast rather than imported, so it runs in CI without faiss,
psycopg or a live database.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FAISS = ROOT / "src" / "vectorstore" / "vectordb.py"
PGVEC = ROOT / "src" / "vectorstore" / "pgvector_store.py"

# load() is a FAISS-specific constructor; __init__ signatures legitimately differ.
EXEMPT = {"__init__", "load"}


def _methods(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}

    for cls in (
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    ):
        for fn in (
            node for node in cls.body if isinstance(node, ast.FunctionDef)
        ):
            out[fn.name] = [
                arg.arg for arg in fn.args.args if arg.arg != "self"
            ]

    return out


def test_pgvector_implements_every_faiss_method():
    missing = set(_methods(FAISS)) - EXEMPT - set(_methods(PGVEC))

    assert not missing, (
        f"PgVectorStore is missing {sorted(missing)}"
    )


def test_shared_methods_take_the_same_arguments():
    faiss = _methods(FAISS)
    pg = _methods(PGVEC)

    for name in set(faiss) & set(pg) - EXEMPT:
        # The caller must be able to use positional args identically.
        shared = min(len(faiss[name]), len(pg[name]))

        assert faiss[name][:shared] == pg[name][:shared], (
            f"{name}() args differ: "
            f"faiss={faiss[name]} pgvector={pg[name]}"
        )