"""Pipeline service facade.

One object that wires the whole system together so callers
(FastAPI, the evaluation, tests) never touch individual modules.

Responsibilities:
- route a question to summaries, data or chart
- answer from the data summaries, with citations and a confidence gate
- answer from the measurements database, with generated and validated SQL
- log every interaction exactly once

The service exposes two public query methods:

    ask(question) -> Answer
        Summaries-only path returning the typed Answer dataclass.
        Used by the API's route_override and by the generation evaluation.

    answer(question) -> dict
        Routed path returning a uniform dictionary. Used by FastAPI, and so by
        the React front end. This is the normal way in: the router decides which
        source answers, so a caller does not have to know before asking.
"""

from __future__ import annotations

import json
import re
import time

import httpx

from src.charts.builder import render, to_frame
from src.charts.ocean import pick_ocean_chart, render_ocean
from src.generation.answer_generator import AnswerGenerator
from src.monitoring.logger import log_interaction, log_result
from src.monitoring.tracing import set_attributes, span, trace_id
from src.router.classifier import route as pick_route
from src.router.rewriter import rewrite
from src.semantic.index import SemanticIndex, as_context
from src.sqlgen.checks import check as check_answers_question
from src.sqlgen.executor import run_query
from src.sqlgen.generator import generate_sql, repair_sql
from src.sqlgen.schema_context import load_column_catalog
from src.sqlgen.validator import SQLRejected, validate
from src.utils.config import get_config
from src.utils.schemas import Answer, Summary


def _first_line(exc: Exception) -> str:
    """The readable part of an exception, without trailing detail blocks."""
    return str(exc).strip().splitlines()[0].strip()


_LIMIT = re.compile(r"\blimit\s+(\d+)\s*$", re.I)


def _hit_limit(sql: str | None, row_count: int) -> bool:
    """Whether the result filled the LIMIT and so may have been cut short."""
    match = _LIMIT.search((sql or "").strip().rstrip(";"))

    return bool(match) and row_count >= int(match.group(1))


def _repair_refusal(sql, error, column_catalog):
    """Why this failure must not be repaired, or None."""
    from src.sqlgen.scope import error_is_the_answer
    from src.sqlgen.validator import PLATFORMS

    try:
        return error_is_the_answer(sql, error, column_catalog, PLATFORMS)
    except Exception:
        return None


class RAGService:
    """Facade that orchestrates retrieval, routing and queries."""

    def __init__(self) -> None:
        self.cfg = get_config()

        # One index, two jobs. The summaries route answers from it directly;
        # the data route reads it for context before writing SQL.
        self.semantic = SemanticIndex()
        self.generator = AnswerGenerator(self.semantic)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
    ) -> dict:
        """Route, delegate and log one query exactly once.

        ``history`` is the recent (question, answer) pairs of this
        conversation, oldest first.

        A follow up is rewritten into a standalone question here and nowhere
        else. Everything after that works on one self-contained question, so
        the router, the SQL generator and the cache never learn that
        conversations exist.
        """
        with span("floatchat.answer", **{"floatchat.turn": len(history or []) + 1}) as root:
            result = self._answer(question, history, root)

        try:
            # The question as typed, so the log reads back as the conversation
            # happened. What was actually run is question_rewritten beside it.
            log_result(
                question,
                result,
            )
        except Exception:
            # Logging must never break the user path.
            pass

        return result

    def _answer(self, question, history, root) -> dict:
        started = time.perf_counter()

        asked = question

        if history:
            with span("rewrite", **{"floatchat.history_turns": len(history)}) as step:
                question = rewrite(question, history)
                set_attributes(step, **{"floatchat.rewritten": question != asked})

        with span("route") as step:
            chosen, how = pick_route(
                question,
                fallback=self.cfg.router.fallback,
                use_model=self.cfg.router.enabled,
            )
            set_attributes(
                step,
                **{"floatchat.route": chosen, "floatchat.route_decided_by": how},
            )

        if chosen == "summaries":
            result = self.answer_from_summaries(
                question
            )
        else:
            result = self.answer_from_data(
                question,
                want_chart=(chosen == "chart"),
            )

        result["route"] = chosen
        result["route_decided_by"] = how

        # Both, always. A bad rewrite is the most likely failure of a follow up
        # and it is invisible unless the pair travels together.
        result["question"] = asked
        result["question_rewritten"] = question

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

        # The join key between interactions.jsonl and traces.jsonl, and what a
        # reader pastes into Jaeger or Phoenix to see where the time went.
        result["trace_id"] = trace_id(root)

        set_attributes(
            root,
            **{
                "floatchat.route": chosen,
                "floatchat.answered": bool(result.get("answered")),
                "floatchat.refused": bool(result.get("refused")),
                "floatchat.reason": result.get("reason") or None,
            },
        )

        return result

    def answer_from_summaries(
        self,
        question: str,
    ) -> dict:
        """Answer a question from the data summaries, with citations."""
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
            with span("sql.generate") as step:
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
                set_attributes(step, **{"floatchat.sql_cached": bool(cached)})
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

        def _attempt(sql):
            """Validate and run one candidate query."""
            with span("sql.validate"):
                safe = validate(
                    sql,
                    allowed_tables=self.cfg.sql.allowed_tables,
                    max_limit=self.cfg.sql.max_rows,
                    default_limit=self.cfg.sql.default_limit,
                    column_catalog=self._column_catalog(),
                )

                # A query can be safe and still answer a different question.
                # Every problem found goes to the repair in one message, before
                # the database is touched (src/sqlgen/checks.py).
                check_answers_question(question, safe, context)

            with span(
                "db.query",
                **{"db.system.name": "postgresql", "db.query.text": safe},
            ) as step:
                rows = run_query(
                    safe,
                    timeout_ms=self.cfg.sql.timeout_ms,
                )
                set_attributes(step, **{"db.response.returned_rows": rows["row_count"]})

            return safe, rows

        repaired = False

        try:
            safe_sql, result = _attempt(raw_sql)
        except Exception as first:
            # One more attempt, given the error the query failed with. The
            # benchmark has had this since it was written, and the app had
            # not, which made the published numbers describe a system nobody
            # was using. They match now.
            declined = isinstance(first, SQLRejected)

            reason = _repair_refusal(raw_sql, first, self._column_catalog())

            if declined or reason or not self.cfg.sql.get("repair", True):
                # A refusal is not repaired, and neither is a failure that is
                # itself the answer: a column missing from the platform the
                # query reads means the data is not there, and a repair told
                # to fix that will find something else to put under the same
                # alias. It did exactly that once, answering how deep the
                # buoys dived with sea surface temperature aliased to
                # min_pressure.
                detail = reason or _first_line(first)

                return {
                    "answer": (
                        "I could not answer that from the measurements "
                        f"database. Reason: {detail}"
                    ),
                    "answered": False,
                    "refused": True,
                    "generated_sql": raw_sql,
                    "sql_cached": cached,
                    "confidence": 0.0,
                    "reason": detail,
                }

            try:
                raw_sql = repair_sql(
                    question,
                    raw_sql,
                    str(first),
                    model=self.cfg.sql.get("model"),
                    timeout=self.cfg.sql.get("gen_timeout_seconds", 90),
                    context=context,
                )

                safe_sql, result = _attempt(raw_sql)
                repaired = True
            except Exception as second:
                detail = _first_line(second)

                return {
                    "answer": (
                        "The first query failed and a second attempt "
                        f"failed too: {detail}"
                    ),
                    "answered": False,
                    "refused": True,
                    "generated_sql": raw_sql,
                    "sql_cached": cached,
                    "confidence": 0.0,
                    "reason": detail,
                    "error": detail,
                }

        png = None
        kind = None

        # The ocean plot follows the shape of the rows, not the verb in the
        # question. It used to be drawn only on the chart route, which the
        # router picks for "plot", "map" or "track", so "show me salinity
        # profiles near the equator" came back as 500 rows of casts with no
        # profile plot at all. An ordinary aggregate matches no ocean shape and
        # stays a table, so this changes nothing for questions that were
        # never pictures.
        with span("chart.ocean") as step:
            spec, ocean_kind = self._ocean_chart(
                result,
                question,
                truncated=_hit_limit(safe_sql, result["row_count"]),
            )
            set_attributes(step, **{"floatchat.chart_kind": ocean_kind})

        if spec is not None:
            kind = ocean_kind

        if want_chart:
            # The matplotlib render runs whatever the shape, because the API
            # contract promises a PNG and an image client has to keep working.
            # Guarded like the ocean chart: a chart that cannot be drawn should
            # cost the user the chart, not the answer. It was unguarded, so a
            # bug in the renderer surfaced as an HTTP 500 with the rows lost.
            png, generic_kind = None, None

            try:
                with span("chart.render") as step:
                    png, generic_kind = render(
                        result,
                        title=question,
                    )
                    set_attributes(step, **{"floatchat.chart_kind": generic_kind})
            except Exception:
                pass

            kind = kind or generic_kind

        return {
            "answer": self._summarise_rows(result),
            "answered": True,
            "refused": False,
            "generated_sql": safe_sql,
            "sql_cached": cached,
            # The first query failed and this is the second. Surfaced rather
            # than hidden: the shown SQL is not what the model first wrote,
            # and a reader comparing it against the question deserves to know
            # an error was fed back in between.
            "sql_repaired": repaired,
            "columns": result["columns"],
            "rows": result["rows"][:100],
            "row_count": result["row_count"],
            "db_elapsed_ms": result["elapsed_ms"],
            "chart_png": png,
            "chart_kind": kind,
            "chart_spec": spec,
            # For summaries, confidence measures retrieval quality.
            # For SQL, it indicates whether a validated query returned rows.
            "confidence": (
                1.0
                if result["row_count"]
                else 0.0
            ),
        }

    def _ocean_chart(self, result, question, truncated=False):
        """A Plotly spec for this result when it is a known ocean shape.

        Returns ``(None, None)`` for anything unrecognised, so the generic
        matplotlib chart stays the answer for an ordinary aggregate. A failure
        to build the figure is swallowed for the same reason: a chart that
        cannot be drawn should cost the user a chart, not the answer.
        """
        try:
            df = to_frame(result)

            if df.empty:
                return None, None

            kind = pick_ocean_chart(df, question)

            if kind is None:
                return None, None

            figure = render_ocean(df, kind, title=question, truncated=truncated)

            # Through the JSON encoder rather than to_dict, because the figure
            # holds numpy arrays and pandas timestamps that neither FastAPI nor
            # the JSONL log can serialise.
            return json.loads(figure.to_json()), kind

        except Exception:
            return None, None

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
        """Summaries-only path returning the typed Answer."""
        with span("floatchat.ask") as root:
            answer = self.generator.answer(question)
            set_attributes(root, **{"floatchat.answered": answer.answered})

        answer.trace_id = trace_id(root)

        try:
            log_interaction(answer)
        except Exception:
            # Logging must never break the user path.
            pass

        return answer

    def warm_up(self) -> dict:
        """Load the embedding model and the language model before a question does.

        Meant for a background thread at startup. Without it the first
        question paid about 40 seconds to load the embedding model and 18 to
        load the language model, on top of answering. A failure is swallowed:
        a model that could not be loaded now is loaded by the first question,
        exactly as before.
        """
        from src.embeddings.embedding_model import get_embedding_model
        from src.generation.llm import warm_up as load_models

        took: dict = {}

        with span("floatchat.warm_up") as step:
            try:
                started = time.perf_counter()
                get_embedding_model().embed_query("warm up")
                took["embedding_s"] = round(time.perf_counter() - started, 1)
            except Exception as exc:
                took["embedding_error"] = _first_line(exc)

            models = sorted(
                {
                    self.cfg.generation.model,
                    self.cfg.router.get("model") or self.cfg.generation.model,
                    self.cfg.sql.get("model") or self.cfg.generation.model,
                }
            )

            try:
                took.update(load_models(models))
            except Exception as exc:
                took["llm_error"] = _first_line(exc)

            set_attributes(step, **{f"floatchat.{k}": v for k, v in took.items()})

        return took

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """How much the semantic index currently describes."""
        try:
            return {"summaries": self.semantic.count()}
        except Exception:
            return {"summaries": 0}
