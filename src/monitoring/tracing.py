"""Per-step tracing with OpenTelemetry.

``logs/interactions.jsonl`` holds one line per question: what was asked, which
route answered and how long it took end to end. It cannot say where the time
went or which of the model calls behind one question produced a bad
rewrite. A trace can: every step of a question (rewrite, route, retrieve,
generate SQL, validate, query, repair, chart, answer) is a span under
one root, and every model call carries the GenAI semantic convention
attributes (model, token usage) plus the Ollama timings that show a cold load.

Where spans go:

- ``logs/traces.jsonl`` by default, one span per line. No collector needed,
  which matters on a machine that cannot spare the memory for one.
- an OTLP endpoint as well, when ``OTEL_EXPORTER_OTLP_ENDPOINT`` or
  ``monitoring.tracing.otlp_endpoint`` is set. Jaeger, Arize Phoenix and
  Langfuse all accept OTLP over HTTP, so switching viewers is configuration.

Prompt and completion text is not recorded unless
``monitoring.tracing.capture_content`` is on. Questions can be personal, and
the interaction log already keeps the question and answer.

Tracing must never break the user path, so a failure to set up leaves the
OpenTelemetry API's no-op tracer in place and every helper here still works.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from src.utils.config import get_config

_TRACER_NAME = "floatchat"

# Long enough for a text-to-SQL prompt with its schema, short enough that one
# span cannot swell the trace file by megabytes.
_CONTENT_LIMIT = 8000

_setup_lock = threading.Lock()
_ready = False
_provider: TracerProvider | None = None
_capture_content = False


def _settings() -> dict:
    try:
        monitoring = get_config().get("monitoring", {}) or {}
        return dict(monitoring.get("tracing", {}) or {})
    except Exception:
        return {}


class JsonlSpanExporter(SpanExporter):
    """Appends finished spans to a file, one JSON object per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            lines = [
                json.dumps(span_record(span), ensure_ascii=False, default=str)
                for span in spans
            ]
            self.path.parent.mkdir(parents=True, exist_ok=True)

            with self._lock, open(self.path, "a", encoding="utf-8") as fh:
                fh.write("".join(line + "\n" for line in lines))

            return SpanExportResult.SUCCESS
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass


def span_record(span: ReadableSpan) -> dict[str, Any]:
    """One finished span as a flat, readable dictionary."""
    start = span.start_time or 0
    end = span.end_time or start

    return {
        "trace_id": format(span.context.trace_id, "032x"),
        "span_id": format(span.context.span_id, "016x"),
        "parent_id": (
            format(span.parent.span_id, "016x") if span.parent else None
        ),
        "name": span.name,
        "start": datetime.fromtimestamp(start / 1e9, UTC).isoformat(),
        "duration_ms": round((end - start) / 1e6, 2),
        "status": span.status.status_code.name,
        "status_description": span.status.description,
        "attributes": dict(span.attributes or {}),
        "events": [
            {"name": event.name, "attributes": dict(event.attributes or {})}
            for event in span.events
        ],
    }


def setup_tracing() -> None:
    """Install the tracer provider once per process. Safe to call repeatedly."""
    global _ready, _provider, _capture_content

    if _ready:
        return

    with _setup_lock:
        if _ready:
            return

        _ready = True
        cfg = _settings()

        if not cfg.get("enabled", True):
            return

        try:
            _capture_content = bool(cfg.get("capture_content", False))

            provider = TracerProvider(
                resource=Resource.create(
                    {"service.name": cfg.get("service_name", "floatchat")}
                )
            )

            path = cfg.get("file", "logs/traces.jsonl")

            if path:
                # Simple rather than batched: a restarted API or a killed
                # evaluation should not lose the spans still waiting in a queue.
                provider.add_span_processor(
                    SimpleSpanProcessor(JsonlSpanExporter(path))
                )

            endpoint = cfg.get("otlp_endpoint")

            if endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )

                # With no argument the exporter reads the standard environment
                # variables itself, and the environment wins over the file.
                exporter = (
                    OTLPSpanExporter()
                    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
                    else OTLPSpanExporter(
                        endpoint=f"{str(endpoint).rstrip('/')}/v1/traces"
                    )
                )
                provider.add_span_processor(BatchSpanProcessor(exporter))

            _provider = provider
            trace.set_tracer_provider(provider)
        except Exception:
            logging.getLogger("rag").exception("Tracing setup failed")


def use_provider(provider: TracerProvider | None) -> None:
    """Point the helpers at a given provider. For tests."""
    global _ready, _provider

    _provider = provider
    _ready = provider is not None


def _tracer() -> trace.Tracer:
    setup_tracing()
    return (_provider or trace.get_tracer_provider()).get_tracer(_TRACER_NAME)


def _clean(attributes: dict[str, Any]) -> dict[str, Any]:
    """OpenTelemetry accepts primitives and lists of them, and never None."""
    cleaned = {}

    for key, value in attributes.items():
        if value is None:
            continue

        if isinstance(value, str | bool | int | float):
            cleaned[key] = value
        elif isinstance(value, list | tuple) and all(
            isinstance(item, str | bool | int | float) for item in value
        ):
            cleaned[key] = list(value)
        else:
            cleaned[key] = str(value)

    return cleaned


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[trace.Span]:
    """A span for one step. An exception marks it as an error and propagates."""
    with _tracer().start_as_current_span(
        name,
        attributes=_clean(attributes),
    ) as current:
        yield current


def set_attributes(current: trace.Span, **attributes: Any) -> None:
    """Attributes known only once the step has run."""
    current.set_attributes(_clean(attributes))


@contextmanager
def llm_span(
    step: str,
    model: str,
    operation: str = "text_completion",
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Iterator[trace.Span]:
    """A span for one model call, named after the step it serves.

    ``operation`` is the GenAI convention name: ``chat`` for Ollama's
    ``/api/chat`` and ``text_completion`` for ``/api/generate``.
    """
    with span(
        f"llm.{step}",
        **{
            "floatchat.step": step,
            "gen_ai.operation.name": operation,
            "gen_ai.provider.name": "ollama",
            "gen_ai.request.model": model,
            "gen_ai.request.temperature": temperature,
            "gen_ai.request.max_tokens": max_tokens,
            "ollama.num_ctx": _num_ctx(),
        },
    ) as current:
        yield current


def record_ollama(
    current: trace.Span,
    data: dict,
    prompt: str | None = None,
) -> None:
    """Token usage and timings from an Ollama response body.

    Ollama reports durations in nanoseconds. ``load_duration`` is the one to
    watch: a call that spent most of its time loading the model is a cold
    start, not a slow model, and that difference was invisible before.
    """

    def _ms(key: str) -> float | None:
        value = data.get(key)
        return round(value / 1e6, 2) if isinstance(value, int | float) else None

    set_attributes(
        current,
        **{
            "gen_ai.response.model": data.get("model"),
            "gen_ai.usage.input_tokens": data.get("prompt_eval_count"),
            "gen_ai.usage.output_tokens": data.get("eval_count"),
            "ollama.total_duration_ms": _ms("total_duration"),
            "ollama.load_duration_ms": _ms("load_duration"),
            "ollama.prompt_eval_duration_ms": _ms("prompt_eval_duration"),
            "ollama.eval_duration_ms": _ms("eval_duration"),
        },
    )

    if _capture_content:
        completion = data.get("response")

        if completion is None:
            completion = (data.get("message") or {}).get("content")

        set_attributes(
            current,
            **{
                "gen_ai.prompt": (prompt or "")[:_CONTENT_LIMIT] or None,
                "gen_ai.completion": (completion or "")[:_CONTENT_LIMIT] or None,
            },
        )


def _num_ctx() -> int | None:
    try:
        from src.generation.llm import num_ctx

        return num_ctx()
    except Exception:
        return None


def trace_id(current: trace.Span | None = None) -> str | None:
    """The hex trace id of a span, or of the current one, if it is recording."""
    current = current or trace.get_current_span()
    context = current.get_span_context()

    if not context.is_valid:
        return None

    return format(context.trace_id, "032x")
