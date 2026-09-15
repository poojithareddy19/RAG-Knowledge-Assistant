"""Pipeline service facade.

One object that wires the whole system together so callers
(Streamlit, FastAPI, tests) never touch individual modules.

Responsibilities:
- build the configured vector store
- ingest files: load -> chunk -> embed -> add -> persist
- route a question to documents, data or chart
- log every interaction exactly once

The service exposes two public query methods:

    ask(question) -> Answer
        Documents-only path returning the typed Answer dataclass.
        Used by the existing "Ask Questions" page.

    answer(question) -> dict
        Routed path returning a uniform dictionary.
        Used by the Ocean Data page and FastAPI.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from src.charts.builder import render
from src.embeddings.embedding_model import get_embedding_model
from src.generation.answer_generator import AnswerGenerator
from src.ingestion.chunker import chunk_segments
from src.ingestion.loader import SUPPORTED_EXTENSIONS, load_document
from src.monitoring.logger import log_interaction
from src.retrieval.retriever import Retriever
from src.router.classifier import route as pick_route
from src.sqlgen.executor import run_query
from src.sqlgen.generator import generate_sql
from src.sqlgen.validator import SQLRejected, validate
from src.utils.config import get_config
from src.utils.schemas import Answer, Chunk


class RAGService:
    """Facade that orchestrates ingestion, retrieval, routing and queries."""

    def __init__(self) -> None:
        self.cfg = get_config()
        self.embedder = get_embedding_model()
        self.store = self._build_store()

        # The retriever and generator keep a reference to the store,
        # so the store must be created before them.
        self.retriever = Retriever(self.store)
        self.generator = AnswerGenerator(self.retriever)

    def _build_store(self):
        """Build the configured vector-store backend."""
        backend = self.cfg.vectorstore.get(
            "backend",
            "faiss",
        )

        if backend == "pgvector":
            from src.vectorstore.pgvector_store import PgVectorStore

            return PgVectorStore(
                dimension=self.cfg.vectorstore.dimension,
                table=self.cfg.vectorstore.table,
            )

        if backend == "faiss":
            from src.vectorstore.vectordb import FaissVectorStore

            return FaissVectorStore.load(
                dimension=self.embedder.dimension
            )

        raise ValueError(
            f"unknown vectorstore.backend {backend!r}; "
            "expected 'faiss' or 'pgvector'"
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_file(
        self,
        file_path: str | Path,
        skip_duplicates: bool = True,
    ) -> dict:
        """Load, chunk, embed and persist one document."""
        file_path = Path(file_path)
        doc_name = file_path.name

        if (
            skip_duplicates
            and self.store.contains(doc_name)
        ):
            return {
                "doc_name": doc_name,
                "status": "skipped_duplicate",
                "chunks": 0,
            }

        segments = load_document(file_path)

        chunks: list[Chunk] = chunk_segments(
            doc_name,
            segments,
        )

        if not chunks:
            return {
                "doc_name": doc_name,
                "status": "no_text",
                "chunks": 0,
            }

        vectors = self.embedder.embed_passages(
            [chunk.text for chunk in chunks]
        )

        self.store.add(
            chunks,
            vectors,
            embedding_model=self.embedder.model_name,
        )

        self.store.save()

        # Keep a copy of the raw file for provenance.
        raw_dir = Path(self.cfg.paths.raw_dir)
        raw_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            destination = raw_dir / doc_name

            if file_path.resolve() != destination.resolve():
                shutil.copy2(
                    file_path,
                    destination,
                )
        except Exception:
            # Provenance copy should never break ingestion.
            pass

        return {
            "doc_name": doc_name,
            "status": "indexed",
            "pages": max(
                (
                    chunk.page
                    for chunk in chunks
                ),
                default=0,
            ),
            "chunks": len(chunks),
        }

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
    ) -> dict:
        """Route, delegate and log one query exactly once."""
        started = time.perf_counter()

        chosen, how = pick_route(
            question,
            fallback=self.cfg.router.fallback,
            use_model=self.cfg.router.enabled,
        )

        if chosen == "documents":
            result = self.answer_from_documents(
                question
            )
        else:
            result = self.answer_from_data(
                question,
                want_chart=(chosen == "chart"),
            )

        result["route"] = chosen
        result["route_decided_by"] = how

        result.setdefault(
            "elapsed_ms",
            round(
                (
                    time.perf_counter()
                    - started
                )
                * 1000,
                1,
            ),
        )

        try:
            log_interaction(
                result,
                question=question,
            )
        except Exception:
            # Logging must never break the user path.
            pass

        return result

    def answer_from_documents(
        self,
        question: str,
    ) -> dict:
        """Answer a question using the document retrieval pipeline."""
        answer = self.generator.answer(question)

        return {
            "answer": answer.answer,
            "answered": answer.answered,
            "refused": not answer.answered,
            "confidence": answer.confidence.score,
            "confidence_percent": answer.confidence.percent,
            "reason": answer.reason,
            "sources": [
                source.to_dict()
                for source in answer.sources
            ],
            "latency_ms": answer.latency_ms,
            "provider": answer.provider,
            "model": answer.model,
            "embedding_model": answer.embedding_model,
            "error": answer.error,
        }

    def answer_from_data(
        self,
        question: str,
        want_chart: bool = False,
    ) -> dict:
        """Generate, validate and execute a SQL query."""
        raw_sql, cached = generate_sql(
            question,
            model=self.cfg.sql.get("model"),
            use_cache=self.cfg.cache.get(
                "enabled",
                True,
            ),
            return_cache_flag=True,
        )

        try:
            safe_sql = validate(
                raw_sql,
                allowed_tables=self.cfg.sql.allowed_tables,
                max_limit=self.cfg.sql.max_rows,
                default_limit=self.cfg.sql.default_limit,
            )
        except SQLRejected as exc:
            return {
                "answer": (
                    "I could not answer that from the "
                    "measurements database. "
                    f"Reason: {exc}"
                ),
                "answered": False,
                "refused": True,
                "generated_sql": raw_sql,
                "sql_cached": cached,
                "confidence": 0.0,
                "reason": str(exc),
            }

        try:
            result = run_query(
                safe_sql,
                timeout_ms=self.cfg.sql.timeout_ms,
            )
        except Exception as exc:
            return {
                "answer": (
                    "The query was valid but failed to run: "
                    f"{exc}"
                ),
                "answered": False,
                "refused": True,
                "generated_sql": safe_sql,
                "sql_cached": cached,
                "confidence": 0.0,
                "error": str(exc),
            }

        png = None
        kind = None

        if want_chart:
            png, kind = render(
                result,
                title=question,
            )

        return {
            "answer": self._summarise_rows(result),
            "answered": True,
            "refused": False,
            "generated_sql": safe_sql,
            "sql_cached": cached,
            "columns": result["columns"],
            "rows": result["rows"][:100],
            "row_count": result["row_count"],
            "elapsed_ms": result["elapsed_ms"],
            "chart_png": png,
            "chart_kind": kind,
            # For documents, confidence measures retrieval quality.
            # For SQL, it indicates whether a validated query returned rows.
            "confidence": (
                1.0
                if result["row_count"]
                else 0.0
            ),
        }

    @staticmethod
    def _summarise_rows(
        result: dict,
    ) -> str:
        """Create a concise summary without inventing numbers."""
        n = result["row_count"]

        if n == 0:
            return (
                "The query ran successfully "
                "but returned no rows."
            )

        if (
            n == 1
            and len(result["columns"]) == 1
        ):
            return (
                f"{result['columns'][0]}: "
                f"{result['rows'][0][0]}"
            )

        return (
            f"{n} row(s) returned with columns "
            f"{', '.join(result['columns'])}. "
            "See the table below."
        )

    def ask(
        self,
        question: str,
    ) -> Answer:
        """Documents-only path returning the typed Answer."""
        answer = self.generator.answer(question)

        try:
            log_interaction(
                answer,
                question=question,
            )
        except Exception:
            # Logging must never break the user path.
            pass

        return answer

    # ------------------------------------------------------------------
    # Knowledge base
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return vector-store statistics."""
        return self.store.stats()

    def reset(self) -> None:
        """Reset the configured vector store."""
        self.store.reset()

    @staticmethod
    def supported_extensions() -> tuple:
        """Return supported document extensions."""
        return SUPPORTED_EXTENSIONS