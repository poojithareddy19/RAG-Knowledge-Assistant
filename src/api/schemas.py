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


class AskResponse(BaseModel):
    """Response body returned by the /ask endpoint."""

    answer: str
    route: str
    route_decided_by: str
    confidence: float
    refused: bool = False
    reason: str = ""
    citations: list[dict[str, Any]] = []
    generated_sql: str | None = None
    sql_cached: bool | None = None
    columns: list[str] | None = None
    rows: list[list[Any]] | None = None
    row_count: int | None = None
    elapsed_ms: float | None = None


class HealthResponse(BaseModel):
    """Response body returned by the /health endpoint."""

    status: str
    documents_indexed: int
    measurements: int
    llm_reachable: bool