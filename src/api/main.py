"""HTTP layer.

Every endpoint delegates to RAGService, the same object the Streamlit pages
use. There is exactly one place where behaviour is defined, so the web page and
the API cannot drift apart.
"""

from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import AskRequest, AskResponse, HealthResponse
from src.utils.config import get_config
from src.utils.db import fetch_all
from src.utils.pipeline import RAGService

load_dotenv()

cfg = get_config()

app = FastAPI(
    title=cfg.api.get("title", "Grounded Data Assistant"),
    version="0.2.0",
    description=(
        "Grounded answers over documents and ARGO ocean measurements."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(cfg.api.get("cors_origins", ["*"])),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_service: RAGService | None = None


def service() -> RAGService:
    """Build the service on first request, not at import time."""
    global _service

    if _service is None:
        _service = RAGService()

    return _service


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check database and LLM dependencies."""

    try:
        _, docs = fetch_all(
            "SELECT count(*) FROM doc_chunks"
        )
        _, meas = fetch_all(
            "SELECT count(*) FROM measurements"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"database unreachable: {exc}",
        ) from exc

    base = os.environ.get(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )

    try:
        llm_ok = (
            httpx.get(
                f"{base}/api/tags",
                timeout=3,
            ).status_code
            == 200
        )
    except Exception:
        llm_ok = False

    return HealthResponse(
        status="ok",
        documents_indexed=docs[0][0],
        measurements=meas[0][0],
        llm_reachable=llm_ok,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Answer a question through the shared RAG service."""

    svc = service()

    if req.route_override == "documents":
        result = svc.answer_from_documents(req.question)
        result["route"] = "documents"
        result["route_decided_by"] = "override"

    elif req.route_override in ("data", "chart"):
        result = svc.answer_from_data(
            req.question,
            want_chart=(req.route_override == "chart"),
        )
        result["route"] = req.route_override
        result["route_decided_by"] = "override"

    else:
        result = svc.answer(req.question)

    # Binary chart data does not belong in a JSON response.
    result.pop("chart_png", None)

    # The API calls these fields "citations".
    result["citations"] = result.pop("sources", [])

    return AskResponse(
        **{
            key: value
            for key, value in result.items()
            if key in AskResponse.model_fields
        }
    )