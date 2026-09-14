"""Pipeline service facade.

One object that wires the whole system together so callers (Streamlit, tests,
the FastAPI layer) never touch individual modules. Responsibilities:

  * build the vector store chosen in config (FAISS or pgvector)
  * ingest files: load -> chunk -> embed -> add -> persist
  * route a question to documents, data or chart, then delegate
  * answer from documents through the AnswerGenerator
  * answer from data through generate -> validate -> execute
  * expose knowledge-base stats

Kept deliberately thin: it orchestrates, it doesn't implement.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from src.charts.builder import render
from src.embeddings.embedding_model import get_embedding_model
from src.generation.answer_generator import AnswerGenerator
from src.ingestion.chunker import chunk_segments
from src.ingestion.loader import SUPPORTED_EXTENSIONS, load_document
from src.retrieval.retriever import Retriever
from src.router.classifier import route as pick_route
from src.sqlgen.executor import run_query
from src.sqlgen.generator import generate_sql
from src.sqlgen.validator import SQLRejected, validate
from src.utils.config import get_config
from src.utils.schemas import Chunk


def _to_dict(result):
    """Answer objects, pydantic models and dataclasses all become plain dicts."""
    if isinstance(result, dict):
        return dict(result)
    for attr in ("model_dump", "dict", "_asdict"):
        fn = getattr(result, attr, None)
        if callable(fn):
            return dict(fn())
    return dict(vars(result))


class RAGService:
    def __init__(self) -> None:
        self.cfg = get_config()
        self.embedder = get_embedding_model()

        backend = self.cfg.vectorstore.backend

        if backend == "pgvector":
            from src.vectorstore.pgvector_store import PgVectorStore

            self.store = PgVectorStore(
                table=self.cfg.vectorstore.table,
                dimension=self.cfg.vectorstore.dimension,
            )
        else:
            from src.vectorstore.vectordb import FaissVectorStore

            self.store = FaissVectorStore.load(
                dimension=self.embedder.dimension
            )

        self.retriever = Retriever(self.store)
        self.generator = AnswerGenerator(self.retriever)

    # -- ingestion -----------------------------------------------------------
    def ingest_file(self, file_path: str | Path, skip_duplicates: bool = True) -> dict:
        file_path = Path(file_path)
        doc_name = file_path.name

        if skip_duplicates and self.store.contains(doc_name):
            return {"doc_name": doc_name, "status": "skipped_duplicate", "chunks": 0}

        segments = load_document(file_path)
        chunks: list[Chunk] = chunk_segments(doc_name, segments)
        if not chunks:
            return {"doc_name": doc_name, "status": "no_text", "chunks": 0}

        vectors = self.embedder.embed_passages([c.text for c in chunks])
        self.store.add(chunks, vectors, embedding_model=self.embedder.model_name)
        self.store.save()

        # Keep a copy of the raw file for provenance.
        raw_dir = Path(self.cfg.paths.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        try:
            if Path(file_path).resolve() != (raw_dir / doc_name).resolve():
                shutil.copy2(file_path, raw_dir / doc_name)
        except Exception:
            pass

        return {
            "doc_name": doc_name,
            "status": "indexed",
            "pages": max((c.page for c in chunks), default=0),
            "chunks": len(chunks),
        }

    # -- querying ------------------------------------------------------------
    def answer(self, question: str) -> dict:
        """Single entry point. Routes, then delegates."""
        chosen, how = pick_route(
            question,
            fallback=self.cfg.router.fallback,
            use_model=self.cfg.router.enabled,
        )

        if chosen == "documents":
            result = self.answer_from_documents(question)
        else:
            result = self.answer_from_data(question, want_chart=(chosen == "chart"))

        result["route"] = chosen
        result["route_decided_by"] = how

        logger = getattr(self, "logger", None)
        if logger is not None:
            logger.log_interaction(question, result)

        return result

    def answer_from_documents(self, question: str) -> dict:
        """Documents only. No routing here: answer() decides the route."""
        result = _to_dict(self.generator.answer(question))

        # Answer uses `sources`; the API contract calls them citations.
        if "citations" not in result and "sources" in result:
            result["citations"] = [
                s if isinstance(s, dict) else _to_dict(s)
                for s in (result["sources"] or [])
            ]

        result.setdefault("answer", "")
        result.setdefault("confidence", 0.0)
        result.setdefault("refused", not result.get("answered", True))
        result.setdefault("citations", [])
        return result

    def answer_from_data(self, question: str, want_chart: bool = False) -> dict:
        """Text to SQL, validated, executed. Numbers come from the database."""
        raw_sql = generate_sql(question)

        try:
            safe_sql = validate(
                raw_sql,
                allowed_tables=self.cfg.sql.allowed_tables,
                max_limit=self.cfg.sql.max_rows,
                default_limit=self.cfg.sql.default_limit,
            )
        except SQLRejected as exc:
            return {
                "answer": ("I could not answer that from the measurements "
                           f"database. Reason: {exc}"),
                "refused": True,
                "generated_sql": raw_sql,
                "confidence": 0.0,
            }

        result = run_query(safe_sql, timeout_ms=self.cfg.sql.timeout_ms)

        png, kind = (None, None)
        if want_chart:
            png, kind = render(result, title=question)

        return {
            "answer": self._summarise_rows(question, result),
            "refused": False,
            "generated_sql": safe_sql,
            "columns": result["columns"],
            "rows": result["rows"][:100],
            "row_count": result["row_count"],
            "elapsed_ms": result["elapsed_ms"],
            "chart_png": png,
            "chart_kind": kind,
            # confidence is 1.0 when a validated query returned rows: the answer
            # comes from the database, not from the model's memory
            "confidence": 1.0 if result["row_count"] else 0.0,
        }

    def _summarise_rows(self, question: str, result: dict) -> str:
        """One short sentence describing the result set, no invented numbers."""
        n = result["row_count"]
        if n == 0:
            return "The query ran successfully but returned no rows."
        if n == 1 and len(result["columns"]) == 1:
            return f"{result['columns'][0]}: {result['rows'][0][0]}"
        return (f"{n} row(s) returned with columns "
                f"{', '.join(result['columns'])}. See the table below.")

    # -- knowledge base ------------------------------------------------------
    def stats(self) -> dict:
        return self.store.stats()

    def reset(self) -> None:
        self.store.reset()

    @staticmethod
    def supported_extensions() -> tuple:
        return SUPPORTED_EXTENSIONS