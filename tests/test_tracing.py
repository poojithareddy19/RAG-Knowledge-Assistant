"""Tracing: what a span records, where it is written, and how one question nests.

The contract pinned down here: a model call carries the GenAI attributes and
the Ollama timings, prompt text stays out unless asked for, a failing step is
marked as an error, the file exporter writes one linked record per span, and
every step of a routed question sits under one root whose id the result and
the interaction log both carry.
"""

from __future__ import annotations

import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from src.monitoring import tracing
from src.utils.config import AttrDict
from src.utils.schemas import Confidence, Summary

OLLAMA_BODY = {
    "model": "llama3.1:latest",
    "response": "SELECT 1",
    "prompt_eval_count": 812,
    "eval_count": 34,
    "total_duration": 4_500_000_000,
    "load_duration": 3_000_000_000,
    "prompt_eval_duration": 900_000_000,
    "eval_duration": 600_000_000,
}


@pytest.fixture
def spans(monkeypatch):
    """Finished spans, captured in memory for the duration of one test."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    previous = tracing._provider
    tracing.use_provider(provider)
    monkeypatch.setattr(tracing, "_capture_content", False)

    yield exporter

    tracing.use_provider(previous)


def _by_name(exporter):
    return {span.name: span for span in exporter.get_finished_spans()}


def test_a_model_call_records_usage_and_the_cold_load(spans):
    with tracing.llm_span("sql", "llama3.1:latest", temperature=0, max_tokens=400) as span:
        tracing.record_ollama(span, OLLAMA_BODY, prompt="the prompt")

    span = _by_name(spans)["llm.sql"]
    attrs = span.attributes

    assert attrs["gen_ai.operation.name"] == "text_completion"
    assert attrs["gen_ai.provider.name"] == "ollama"
    assert attrs["gen_ai.request.model"] == "llama3.1:latest"
    assert attrs["gen_ai.request.max_tokens"] == 400
    assert attrs["gen_ai.usage.input_tokens"] == 812
    assert attrs["gen_ai.usage.output_tokens"] == 34
    # Nanoseconds from Ollama, milliseconds on the span.
    assert attrs["ollama.load_duration_ms"] == 3000.0
    assert attrs["ollama.eval_duration_ms"] == 600.0


def test_prompt_text_stays_out_unless_capture_is_on(spans, monkeypatch):
    with tracing.llm_span("route", "m") as span:
        tracing.record_ollama(span, OLLAMA_BODY, prompt="private question")

    monkeypatch.setattr(tracing, "_capture_content", True)

    with tracing.llm_span("rewrite", "m") as span:
        tracing.record_ollama(span, OLLAMA_BODY, prompt="private question")

    found = _by_name(spans)

    assert "gen_ai.prompt" not in found["llm.route"].attributes
    assert found["llm.rewrite"].attributes["gen_ai.prompt"] == "private question"
    assert found["llm.rewrite"].attributes["gen_ai.completion"] == "SELECT 1"


def test_chat_responses_are_captured_from_the_message(spans, monkeypatch):
    monkeypatch.setattr(tracing, "_capture_content", True)

    with tracing.llm_span("answer", "m", operation="chat") as span:
        tracing.record_ollama(span, {"message": {"content": "142 profiles [1]"}})

    attrs = _by_name(spans)["llm.answer"].attributes

    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.completion"] == "142 profiles [1]"


def test_a_failing_step_is_an_error_and_still_raises(spans):
    with pytest.raises(ValueError), tracing.span("sql.validate"):
        raise ValueError("only SELECT is allowed")

    span = _by_name(spans)["sql.validate"]

    assert span.status.status_code is StatusCode.ERROR
    assert span.events[0].name == "exception"


def test_attributes_drop_none_and_flatten_what_otel_cannot_hold(spans):
    with tracing.span(
        "step",
        empty=None,
        subjects=["1901393", "Arabian Sea"],
        odd={"a": 1},
    ):
        pass

    attrs = _by_name(spans)["step"].attributes

    assert "empty" not in attrs
    assert list(attrs["subjects"]) == ["1901393", "Arabian Sea"]
    assert attrs["odd"] == "{'a': 1}"


def test_the_file_exporter_writes_one_linked_record_per_span(tmp_path):
    path = tmp_path / "traces.jsonl"
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(tracing.JsonlSpanExporter(path)))
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("floatchat.answer") as root:
        with tracer.start_as_current_span("route") as child:
            child.set_attribute("floatchat.route", "data")

    records = [json.loads(line) for line in path.read_text().splitlines()]
    by_name = {record["name"]: record for record in records}

    assert len(records) == 2
    assert by_name["route"]["parent_id"] == by_name["floatchat.answer"]["span_id"]
    assert by_name["route"]["trace_id"] == format(root.get_span_context().trace_id, "032x")
    assert by_name["floatchat.answer"]["parent_id"] is None
    assert by_name["route"]["attributes"] == {"floatchat.route": "data"}
    assert by_name["route"]["status"] == "UNSET"


def test_no_span_means_no_trace_id():
    assert tracing.trace_id() is None


# ---------------------------------------------------------------------------
# One routed question, end to end through the service with its edges stubbed
# ---------------------------------------------------------------------------


def _service(monkeypatch, route):
    from src.utils import pipeline

    logged = []

    monkeypatch.setattr(pipeline, "pick_route", lambda q, **_: (route, "rules"))
    monkeypatch.setattr(pipeline, "log_result", lambda q, r: logged.append(r))

    svc = pipeline.RAGService.__new__(pipeline.RAGService)
    svc.cfg = AttrDict(
        {
            "router": {"fallback": "summaries", "enabled": False},
            "sql": {
                "allowed_tables": ["measurements"],
                "max_rows": 1000,
                "default_limit": 100,
                "timeout_ms": 5000,
                "repair": True,
            },
            "cache": {"enabled": False},
        }
    )

    return pipeline, svc, logged


def test_every_step_of_a_data_question_nests_under_one_root(spans, monkeypatch):
    pipeline, svc, logged = _service(monkeypatch, "data")

    monkeypatch.setattr(svc, "_summaries", lambda q: [])
    monkeypatch.setattr(svc, "_column_catalog", lambda: None)
    monkeypatch.setattr(
        pipeline,
        "generate_sql",
        lambda *a, **k: ("SELECT count(*) FROM measurements", False),
    )
    monkeypatch.setattr(pipeline, "validate", lambda sql, **_: sql + " LIMIT 100")
    monkeypatch.setattr(
        pipeline,
        "run_query",
        lambda sql, **_: {
            "columns": ["count"],
            "rows": [[42]],
            "row_count": 1,
            "elapsed_ms": 3.0,
        },
    )

    result = svc.answer("how many measurements are there")

    found = _by_name(spans)
    root = found["floatchat.answer"]

    for name in ("route", "sql.generate", "sql.validate", "db.query"):
        assert found[name].parent.span_id == root.context.span_id, name

    assert found["db.query"].attributes["db.response.returned_rows"] == 1
    assert found["db.query"].attributes["db.query.text"].endswith("LIMIT 100")
    assert found["sql.generate"].attributes["floatchat.sql_cached"] is False
    assert root.attributes["floatchat.route"] == "data"
    assert root.attributes["floatchat.answered"] is True

    # One id joins the response, the interaction log and the trace.
    assert result["trace_id"] == format(root.context.trace_id, "032x")
    assert logged[0]["trace_id"] == result["trace_id"]


def test_a_rejected_query_leaves_an_error_span_behind(spans, monkeypatch):
    pipeline, svc, _ = _service(monkeypatch, "data")

    def reject(sql, **_):
        raise pipeline.SQLRejected("only SELECT is allowed")

    monkeypatch.setattr(svc, "_summaries", lambda q: [])
    monkeypatch.setattr(svc, "_column_catalog", lambda: None)
    monkeypatch.setattr(pipeline, "generate_sql", lambda *a, **k: ("DROP TABLE x", False))
    monkeypatch.setattr(pipeline, "validate", reject)

    result = svc.answer("drop the table")

    found = _by_name(spans)

    assert result["refused"] is True
    assert found["sql.validate"].status.status_code is StatusCode.ERROR
    assert "db.query" not in found
    assert found["floatchat.answer"].attributes["floatchat.refused"] is True


def test_the_summaries_route_carries_the_trace_id_too(spans, monkeypatch):
    from src.utils.schemas import Answer

    _, svc, _ = _service(monkeypatch, "summaries")

    class FakeGenerator:
        def answer(self, question):
            with tracing.span("retrieve"):
                pass

            return Answer(
                question=question,
                answer="Float 1901393 recorded 142 profiles [1].",
                answered=True,
                confidence=Confidence(score=0.8, percent=80),
                sources=[Summary(kind="float", subject="1901393", text="t", score=0.8)],
            )

    svc.generator = FakeGenerator()

    result = svc.answer("tell me about float 1901393")

    found = _by_name(spans)

    assert found["retrieve"].parent.span_id == found["floatchat.answer"].context.span_id
    assert result["trace_id"] == format(found["floatchat.answer"].context.trace_id, "032x")


def test_a_chart_that_cannot_be_drawn_still_returns_the_rows(spans, monkeypatch):
    """The generic renderer is guarded like the ocean chart: a failure there
    must cost the chart, not the answer."""
    pipeline, svc, _ = _service(monkeypatch, "chart")

    def broken(result, title=None):
        raise TypeError("renderer bug")

    monkeypatch.setattr(svc, "_summaries", lambda q: [])
    monkeypatch.setattr(svc, "_column_catalog", lambda: None)
    monkeypatch.setattr(pipeline, "generate_sql", lambda *a, **k: ("SELECT 1 FROM profiles", False))
    monkeypatch.setattr(pipeline, "validate", lambda sql, **_: sql + " LIMIT 500")
    monkeypatch.setattr(
        pipeline,
        "run_query",
        lambda sql, **_: {"columns": ["n"], "rows": [[1]], "row_count": 1, "elapsed_ms": 1.0},
    )
    monkeypatch.setattr(pipeline, "render", broken)

    result = svc.answer("plot the count")

    assert result["answered"] is True
    assert result["rows"] == [[1]]
    assert result["chart_png"] is None
    assert _by_name(spans)["chart.render"].status.status_code is StatusCode.ERROR


def test_a_query_that_runs_past_the_year_is_repaired_before_it_is_run(spans, monkeypatch):
    """The real follow-up: "and in 2022?" came back as obs_time >= '2022-01-01'
    and was answered 63. The check sends it to the repair with its reason, and
    the database only ever sees the bounded query."""
    pipeline, svc, _ = _service(monkeypatch, "data")

    ran = []
    repair_errors = []

    def repair(question, broken, error, **_):
        repair_errors.append(error)
        return (
            "SELECT COUNT(*) FROM profiles WHERE region = 'Arabian Sea' "
            "AND obs_time >= '2022-01-01' AND obs_time < '2023-01-01'"
        )

    monkeypatch.setattr(svc, "_summaries", lambda q: [])
    monkeypatch.setattr(svc, "_column_catalog", lambda: None)
    monkeypatch.setattr(
        pipeline,
        "generate_sql",
        lambda *a, **k: (
            "SELECT COUNT(*) FROM profiles WHERE region = 'Arabian Sea' "
            "AND obs_time >= '2022-01-01'",
            False,
        ),
    )
    monkeypatch.setattr(pipeline, "validate", lambda sql, **_: sql + " LIMIT 500")
    monkeypatch.setattr(pipeline, "repair_sql", repair)

    def run(sql, **_):
        ran.append(sql)
        return {"columns": ["count"], "rows": [[0]], "row_count": 1, "elapsed_ms": 1.0}

    monkeypatch.setattr(pipeline, "run_query", run)

    result = svc.answer("How many profiles are in the Arabian Sea and in 2022?")

    assert result["answered"] is True
    assert result["sql_repaired"] is True
    assert result["answer"] == "count: 0"
    assert len(ran) == 1 and "obs_time < '2023-01-01'" in ran[0]
    assert "2023-01-01" in repair_errors[0]

    # Two validate spans: the open-ended query failed it, the repair passed.
    validates = [s for s in spans.get_finished_spans() if s.name == "sql.validate"]
    assert [v.status.status_code for v in validates] == [StatusCode.ERROR, StatusCode.UNSET]


def test_a_profile_count_through_measurements_is_repaired(spans, monkeypatch):
    """The benchmark miss: profiles per region counted measurement rows. The
    check sends it to the repair, and only the corrected query is run."""
    pipeline, svc, _ = _service(monkeypatch, "data")

    ran = []
    repair_errors = []

    monkeypatch.setattr(svc, "_summaries", lambda q: [])
    monkeypatch.setattr(svc, "_column_catalog", lambda: None)
    monkeypatch.setattr(
        pipeline,
        "generate_sql",
        lambda *a, **k: (
            "SELECT p.region, COUNT(*) FROM profiles p JOIN measurements m "
            "ON p.profile_id = m.profile_id WHERE m.qc_flag = 1 GROUP BY p.region",
            False,
        ),
    )
    monkeypatch.setattr(pipeline, "validate", lambda sql, **_: sql + " LIMIT 500")

    def repair(question, broken, error, **_):
        repair_errors.append(error)
        return "SELECT region, count(*) FROM profiles GROUP BY region ORDER BY region"

    def run(sql, **_):
        ran.append(sql)
        return {"columns": ["region", "count"], "rows": [["Arabian Sea", 205]],
                "row_count": 1, "elapsed_ms": 1.0}

    monkeypatch.setattr(pipeline, "repair_sql", repair)
    monkeypatch.setattr(pipeline, "run_query", run)

    result = svc.answer("Number of profiles per region")

    assert result["answered"] is True
    assert result["sql_repaired"] is True
    assert len(ran) == 1 and "measurements" not in ran[0]
    assert "no join to measurements" in repair_errors[0]


def test_a_filter_copied_from_the_context_is_repaired(spans, monkeypatch):
    """A value that is only in the retrieved summaries must not become a
    filter. The check sends it to the repair, which answers the question as
    asked."""
    from src.utils.schemas import Summary

    pipeline, svc, _ = _service(monkeypatch, "data")

    ran = []
    repair_errors = []
    summary = Summary(
        kind="float", subject="1902367",
        text="ARGO float 1902367 is a PROVOR_MT platform and part of project Argo INDIA.",
        score=0.7,
    )

    monkeypatch.setattr(svc, "_summaries", lambda q: [summary])
    monkeypatch.setattr(svc, "_column_catalog", lambda: None)
    monkeypatch.setattr(
        pipeline,
        "generate_sql",
        lambda *a, **k: (
            "SELECT count(*) FROM floats WHERE platform = 'PROVOR_MT'",
            False,
        ),
    )
    monkeypatch.setattr(pipeline, "validate", lambda sql, **_: sql + " LIMIT 500")

    def repair(question, broken, error, **_):
        repair_errors.append(error)
        return "SELECT count(*) FROM floats"

    def run(sql, **_):
        ran.append(sql)
        return {"columns": ["count"], "rows": [[80]], "row_count": 1, "elapsed_ms": 1.0}

    monkeypatch.setattr(pipeline, "repair_sql", repair)
    monkeypatch.setattr(pipeline, "run_query", run)

    result = svc.answer("How many floats are in the database?")

    assert result["answered"] is True
    assert result["answer"] == "count: 80"
    assert len(ran) == 1 and "PROVOR_MT" not in ran[0]
    assert "PROVOR_MT" in repair_errors[0]
