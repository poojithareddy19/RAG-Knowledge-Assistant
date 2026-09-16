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

import httpx

from src.charts.builder import render
from src.embeddings.embedding_model import get_embedding_model
from src.generation.answer_generator import AnswerGenerator
from src.ingestion.chunker import chunk_segments
from src.ingestion.loader import SUPPORTED_EXTENSIONS, load_document
from src.monitoring.logger import log_interaction, log_result
from src.retrieval.retriever import Retriever
from src.router.classifier import route as pick_route
from src.semantic.index import SemanticIndex, Summary, as_context
from src.sqlgen.executor import run_query
from src.sqlgen.generator import generate_sql
from src.sqlgen.schema_context import load_column_catalog
from src.sqlgen.validator import SQLRejected, validate
from src.utils.config import get_config
from src.utils.schemas import Answer, Chunk


def _first_line(exc: Exception) -> str:
    """The readable part of an exception, without trailing detail blocks."""
    return str(exc).strip().splitlines()[0].strip()


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
        self.semantic = SemanticIndex()

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

        # Always the end-to-end wall clock. The database time, when there
        # is one, is reported separately as db_elapsed_ms.
        result["elapsed_ms"] = round(
            (
                time.perf_counter()
                - started
            )
            * 1000,
            1,
        )

        try:
            log_result(
                question,
                result,
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
        """Retrieve context, then generate, validate and execute a SQL query.

        The summaries that shaped the query come back with it, so a reader can
        see what the model was told before it wrote any SQL.
        """
        summaries = self._summaries(question)

        result = self._query_data(
            question,
            want_chart,
            as_context(summaries),
        )

        result["context_used"] = [
            summary.text for summary in summaries
        ]

        return result

    def _query_data(
        self,
        question: str,
        want_chart: bool,
        context: str,
    ) -> dict:
        try:
            raw_sql, cached = generate_sql(
                question,
                model=self.cfg.sql.get("model"),
                timeout=self.cfg.sql.get(
                    "gen_timeout_seconds",
                    90,
                ),
                use_cache=self.cfg.cache.get(
                    "enabled",
                    True,
                ),
                return_cache_flag=True,
                context=context,
            )
        except httpx.TimeoutException:
            return self._sql_failure(
                "The language model did not answer in time, so no SQL was "
                "produced. It may still be loading into memory; try again.",
                reason="sql generation timed out",
            )
        except Exception as exc:
            return self._sql_failure(
                "The language model could not be reached, so no SQL was "
                f"produced. Reason: {_first_line(exc)}",
                reason=f"sql generation failed: {_first_line(exc)}",
            )

        try:
            safe_sql = validate(
                raw_sql,
                allowed_tables=self.cfg.sql.allowed_tables,
                max_limit=self.cfg.sql.max_rows,
                default_limit=self.cfg.sql.default_limit,
                column_catalog=self._column_catalog(),
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
            # Postgres errors carry a multi-line caret diagram that means
            # nothing to a reader of the web page. Keep the first line.
            detail = _first_line(exc)

            return {
                "answer": (
                    "The query was accepted but the database rejected it: "
                    f"{detail}"
                ),
                "answered": False,
                "refused": True,
                "generated_sql": safe_sql,
                "sql_cached": cached,
                "confidence": 0.0,
                "reason": detail,
                "error": detail,
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
            "db_elapsed_ms": result["elapsed_ms"],
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

    def _summaries(self, question: str) -> list[Summary]:
        """What the semantic layer knows about this question, or nothing.

        A database without the summaries table, or one nobody has indexed yet,
        should still answer questions from the schema alone, so a failure here
        degrades to the previous behaviour rather than an error.
        """
        if not self.cfg.get("semantic", {}).get("enabled", True):
            return []

        try:
            return self.semantic.search(question)
        except Exception:
            return []

    @staticmethod
    def _column_catalog() -> dict | None:
        """Live table/column map for the validator, or None if unavailable.

        A database that cannot be introspected should not stop a query from
        being attempted; the validator simply skips the column check.
        """
        try:
            return load_column_catalog()
        except Exception:
            return None

    @staticmethod
    def _sql_failure(
        message: str,
        reason: str,
    ) -> dict:
        """Uniform refusal for a query that never reached the database."""
        return {
            "answer": message,
            "answered": False,
            "refused": True,
            "generated_sql": None,
            "sql_cached": False,
            "confidence": 0.0,
            "reason": reason,
            "error": reason,
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
            log_interaction(answer)
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