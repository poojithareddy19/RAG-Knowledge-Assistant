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
from src.monitoring.tracing import span, trace_id
from src.utils.config import get_config
from src.utils.db import fetch_all
from src.utils.pipeline import RAGService

load_dotenv()

cfg = get_config()

app = FastAPI(
    title=cfg.api.get("title", "FloatChat"),
    version=str(cfg.app.get("version", "0.3.0")),
    description=(
        "Grounded answers over ARGO ocean float data: summaries of what the "
        "archive holds, and SQL generated against the measurements."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(cfg.api.get("cors_origins", ["*"])),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_service: RAGService | None = None

# session_id -> the recent (question, answer) pairs of that conversation.
#
# In process and in memory, which is worth stating plainly rather than
# discovering: it does not survive a restart, and it does not work across
# workers, because a second uvicorn worker has its own copy of this dict and a
# follow up routed there sees no history. For a single-process deployment it is
# enough, and a follow up that loses its history degrades to being answered as
# a standalone question rather than to an error. Redis or a sessions table is
# the fix when either of those stops being acceptable.
_history: dict[str, list[tuple[str, str]]] = {}


def service() -> RAGService:
    """Build the service on first request, not at import time."""
    global _service

    if _service is None:
        _service = RAGService()

    return _service


def _max_turns() -> int:
    return int(cfg.get("conversation", {}).get("max_turns", 4))


def remember(session_id: str, question: str, answer: str) -> None:
    """Append one exchange to a session, keeping only the recent turns."""
    turns = _history.setdefault(session_id, [])
    turns.append((question, answer))

    cap = _max_turns()

    if cap > 0 and len(turns) > cap:
        del turns[:-cap]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check database and LLM dependencies."""

    try:
        _, summaries = fetch_all(
            "SELECT count(*) FROM data_summaries"
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
        summaries_indexed=summaries[0][0],
        measurements=meas[0][0],
        llm_reachable=llm_ok,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Answer a question through the shared RAG service."""

    svc = service()

    if req.route_override == "summaries":
        with span("floatchat.answer", **{"floatchat.route_override": "summaries"}) as root:
            result = svc.answer_from_summaries(req.question)
        result["route"] = "summaries"
        result["route_decided_by"] = "override"
        result["trace_id"] = trace_id(root)

    elif req.route_override in ("data", "chart"):
        with span(
            "floatchat.answer",
            **{"floatchat.route_override": req.route_override},
        ) as root:
            result = svc.answer_from_data(
                req.question,
                want_chart=(req.route_override == "chart"),
            )
        result["route"] = req.route_override
        result["route_decided_by"] = "override"
        result["trace_id"] = trace_id(root)

    else:
        result = svc.answer(
            req.question,
            history=_history.get(req.session_id) if req.session_id else None,
        )

    if req.session_id:
        remember(
            req.session_id,
            req.question,
            str(result.get("answer", ""))[:400],
        )

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