"""Request and response contracts for the HTTP layer.

Two payoffs from declaring these. Bad input is rejected before it reaches any
of our code, with a readable error instead of a stack trace. And FastAPI builds
the interactive /docs page from them, so anyone can exercise the API without
reading the source.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str = Field(
        min_length=3,
        max_length=500,
    )
    route_override: Literal["documents", "data", "chart"] | None = None
    # Opaque, client chosen. Supplying the same one on a later request is what
    # makes "and in 2022?" resolvable; omitting it asks a standalone question.
    session_id: str | None = Field(default=None, max_length=128)


class ExportRequest(AskRequest):
    """Request body for /export: an /ask question plus the file format."""

    format: Literal["csv", "netcdf"] = "csv"


class AskResponse(BaseModel):
    """Response body returned by the /ask endpoint."""

    answer: str
    route: str
    route_decided_by: str
    confidence: float
    refused: bool = False
    reason: str = ""
    citations: list[dict[str, Any]] = []
    context_used: list[str] = []
    generated_sql: str | None = None
    sql_cached: bool | None = None
    # True when the first query failed and this is the second attempt,
    # written after the error was fed back. The SQL shown is then not
    # what the model first produced, which a client comparing it against
    # the question should be able to see.
    sql_repaired: bool | None = None
    columns: list[str] | None = None
    rows: list[list[Any]] | None = None
    row_count: int | None = None
    elapsed_ms: float | None = None
    db_elapsed_ms: float | None = None
    # The question as asked, and the standalone version that was actually run.
    # They differ only when a follow up was resolved against session history.
    question: str | None = None
    question_rewritten: str | None = None
    # A Plotly figure for the domain plots, present only when the result has
    # a shape one of them fits. chart_kind names it either way, so a client
    # that only renders images still knows what it was given.
    chart_spec: dict | None = None
    chart_kind: str | None = None


class HealthResponse(BaseModel):
    """Response body returned by the /health endpoint."""

    status: str
    documents_indexed: int
    measurements: int
    llm_reachable: bool