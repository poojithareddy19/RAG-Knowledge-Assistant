from typing import Any, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    route_override: Literal["documents", "data", "chart"] | None = None


class AskResponse(BaseModel):
    answer: str
    route: str
    route_decided_by: str
    confidence: float
    refused: bool = False
    citations: list[dict[str, Any]] = []
    generated_sql: str | None = None
    columns: list[str] | None = None
    rows: list[list[Any]] | None = None
    row_count: int | None = None
    elapsed_ms: float | None = None


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    measurements: int
    llm_reachable: bool