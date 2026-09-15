from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.config import get_config
from src.utils.pipeline import RAGService


st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading models and vector store…")
def get_service() -> RAGService:
    return RAGService()


def _page_home(cfg):
    st.title("Production RAG Knowledge Assistant")
    st.caption(
        f"v{cfg.app.version} — grounded answers with citations, "
        "confidence and monitoring"
    )

    st.markdown(
        "This assistant answers questions **only** from your uploaded documents. "
        "Every answer carries citations (document, page, chunk) and a confidence "
        "score; when the evidence is too weak it declines instead of guessing."
    )

    svc = get_service()
    stats = svc.stats()

    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", stats["documents"])
    c2.metric("Indexed chunks", stats["chunks"])
    c3.metric("Embedding dim", stats["dimension"])

    st.info(
        "**Pipeline:** question → embed → vector retrieve → confidence gate → "
        "grounded LLM answer → citations. Configure everything in "
        "`config.yaml` / `.env`."
    )


def _page_upload(cfg):
    st.title("Upload Documents")

    svc = get_service()
    exts = [e.lstrip(".") for e in svc.supported_extensions()]

    st.write(f"Supported types: {', '.join(exts)}")

    files = st.file_uploader(
        "Upload one or more documents",
        type=exts,
        accept_multiple_files=True,
    )

    dedup = st.checkbox(
        "Skip documents already indexed (duplicate detection)",
        value=True,
    )

    if files and st.button("Ingest & index", type="primary"):
        progress = st.progress(0.0)
        results = []

        for i, file in enumerate(files, start=1):
            tmp_dir = Path(tempfile.mkdtemp(prefix="rag_upload_"))
            tmp_path = tmp_dir / file.name
            tmp_path.write_bytes(file.getvalue())

            try:
                with st.spinner(f"Processing {file.name}…"):
                    results.append(
                        svc.ingest_file(
                            tmp_path,
                            skip_duplicates=dedup,
                        )
                    )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            progress.progress(i / len(files))

        st.success("Ingestion complete.")
        st.dataframe(
            pd.DataFrame(results),
            use_container_width=True,
        )

        get_service.clear()


def _page_knowledge_base(cfg):
    st.title("Knowledge Base")

    svc = get_service()
    stats = svc.stats()

    c1, c2 = st.columns(2)
    c1.metric("Documents", stats["documents"])
    c2.metric("Total chunks", stats["chunks"])

    if not stats["manifest"]:
        st.warning(
            "No documents indexed yet. Go to **Upload Documents**."
        )
        return

    rows = [
        {
            "Document": name,
            "Pages": metadata.get("pages", 0),
            "Chunks": metadata.get("chunks", 0),
            "Embedding model": metadata.get("embedding_model", ""),
            "Last indexed": metadata.get("indexed_at", ""),
        }
        for name, metadata in stats["manifest"].items()
    ]

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
    )

    if st.button("⚠️ Reset knowledge base"):
        svc.reset()
        get_service.clear()
        st.rerun()


def _page_ask(cfg):
    st.title("Ask a Question")

    svc = get_service()

    if svc.stats()["chunks"] == 0:
        st.warning("Index some documents first.")
        return

    question = st.text_input(
        "Your question",
        placeholder=(
            "e.g. When are employees eligible for parental leave?"
        ),
    )

    if not question or not st.button("Ask", type="primary"):
        return

    with st.spinner("Retrieving and generating…"):
        ans = svc.ask(question)

    left, right = st.columns([3, 1])

    with left:
        if ans.answered:
            st.markdown("#### Answer")
            st.write(ans.answer)
        else:
            st.warning(ans.answer)
            st.caption(f"Why: {ans.reason}")

        if ans.error:
            st.error(ans.error)

    with right:
        st.metric(
            "Confidence",
            f"{ans.confidence.percent}%",
        )
        st.progress(ans.confidence.score)
        st.caption(
            f"total {ans.latency_ms.get('total_ms', 0):.0f} ms"
        )

    st.markdown(
        "#### Sources"
        if ans.answered
        else "#### Closest passages"
    )

    for source in ans.sources:
        with st.expander(
            f"[{source.rank}] "
            f"{source.chunk.doc_name} — "
            f"page {source.chunk.page} · "
            f"sim {source.score:.3f} · "
            f"{source.chunk.chunk_id}"
        ):
            st.write(source.chunk.text)

    with st.expander("🔍 Retrieval details"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Rank": source.rank,
                        "Score": round(source.score, 4),
                        "Document": source.chunk.doc_name,
                        "Page": source.chunk.page,
                        "Chunk length": source.chunk.char_len,
                    }
                    for source in ans.sources
                ]
            ),
            use_container_width=True,
        )

        st.json(ans.confidence.to_dict())

    if ans.prompt:
        with st.expander("📝 Exact prompt sent to the LLM"):
            st.code(ans.prompt)


def _page_evaluation(cfg):
    st.title("Evaluation")

    svc = get_service()

    gold_path = (
        Path(cfg.paths.processed_dir).parent
        / "evaluation"
        / "questions.csv"
    )

    st.write(f"Gold dataset: `{gold_path}`")

    k = st.slider(
        "k (top-k for metrics)",
        1,
        10,
        cfg.retrieval.top_k,
    )

    if st.button("Run retrieval evaluation", type="primary"):
        if svc.stats()["chunks"] == 0:
            st.warning(
                "Index the documents referenced by the gold set first."
            )
            return

        from src.evaluation.evaluator import evaluate_retrieval

        with st.spinner("Evaluating…"):
            report = evaluate_retrieval(
                svc.retriever,
                gold_path,
                k=k,
            )

        st.metric(
            "Questions",
            report["n_questions"],
        )

        aggregate = report["aggregate"]

        st.bar_chart(pd.Series(aggregate))

        st.dataframe(
            pd.DataFrame([aggregate]),
            use_container_width=True,
        )

        st.caption(
            "Generation metrics (faithfulness, groundedness, "
            "answer relevance) are computed via the LLM-judge / "
            "Ragas path — see docs."
        )


def _page_monitoring(cfg):
    st.title("Monitoring")

    from src.monitoring.analytics import summarize

    summary = summarize()

    if summary.get("total_queries", 0) == 0:
        st.info(
            "No interactions logged yet. Ask some questions first."
        )
        return

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total queries",
        summary["total_queries"],
    )

    c2.metric(
        "Avg latency",
        f"{summary['avg_latency_ms']} ms",
    )

    c3.metric(
        "Fallback rate",
        f"{summary['fallback_rate'] * 100:.0f}%",
    )

    c4.metric(
        "Errors",
        summary["error_count"],
    )

    c5, c6, c7 = st.columns(3)

    c5.metric(
        "Avg confidence",
        f"{summary['avg_confidence']}%",
    )

    c6.metric(
        "Avg LLM latency",
        f"{summary['avg_llm_latency_ms']} ms",
    )

    c7.metric(
        "Avg retrieval sim",
        summary["avg_retrieval_score"],
    )

    st.markdown("##### Confidence distribution")

    st.bar_chart(
        pd.Series(
            summary["confidence_values"],
            name="confidence",
        )
    )

    st.markdown("##### Most queried documents")

    if summary["most_queried_documents"]:
        st.dataframe(
            pd.DataFrame(
                summary["most_queried_documents"],
                columns=["Document", "Retrievals"],
            ),
            use_container_width=True,
        )


def _page_settings(cfg):
    st.title("⚙️ Settings")

    st.write(
        "Read-only view of the active configuration. "
        "Edit `config.yaml` / `.env` to change."
    )

    st.json(
        {
            "embedding_model": cfg.embeddings.model_name,
            "vectorstore": cfg.vectorstore.backend,
            "top_k": cfg.retrieval.top_k,
            "use_reranker": cfg.retrieval.use_reranker,
            "answer_threshold": cfg.confidence.answer_threshold,
            "confidence_weights": dict(cfg.confidence.weights),
            "provider": cfg.generation.provider,
            "model": cfg.generation.model,
        }
    )


def _page_ocean_data(cfg):
    """Display-only ocean data interface."""

    svc = get_service()

    st.header("🌊 Ocean Data")

    st.caption(
        "Ask about ARGO measurements. The question is routed, "
        "the SQL is generated, validated and shown to you before "
        "you are asked to believe the numbers."
    )

    question = st.text_input(
        "Question",
        placeholder=(
            "average surface temperature per year in the Arabian Sea"
        ),
        key="ocean_question",
    )

    if not st.button("Ask", key="ocean_ask") or not question.strip():
        return

    with st.spinner(
        "Routing, generating SQL, running query…"
    ):
        try:
            result = svc.answer(question)
        except Exception as exc:
            st.error(
                f"The query could not be completed: {exc}"
            )
            return

    st.write(
        f"**Route:** `{result['route']}` "
        f"(decided by {result['route_decided_by']})"
    )

    if result["route"] == "documents":
        st.info(
            "This was routed to the uploaded documents, not the database. "
            "Ask it on the Ask Questions page for citations and confidence."
        )
        st.write(result["answer"])
        return

    if result.get("refused"):
        st.warning(result["answer"])

        with st.expander(
            "SQL the model produced (rejected)"
        ):
            st.code(
                result.get("generated_sql") or "",
                language="sql",
            )

        return

    st.success(result["answer"])

    if result.get("chart_png"):
        st.image(result["chart_png"])
    elif result.get("chart_kind") == "table":
        st.caption(
            "This result set is not chartable, "
            "so here is the table."
        )

    if result.get("rows"):
        st.dataframe(
            pd.DataFrame(
                result["rows"],
                columns=result["columns"],
            ),
            use_container_width=True,
        )

    with st.expander("Generated SQL and timing"):
        st.code(
            result["generated_sql"],
            language="sql",
        )

        st.write(
            f"{result['row_count']} rows in "
            f"{result['elapsed_ms']} ms"
            + (
                " · served from cache"
                if result.get("sql_cached")
                else ""
            )
        )


PAGES = {
    "🏠 Home": _page_home,
    "📤 Upload Documents": _page_upload,
    "📚 Knowledge Base": _page_knowledge_base,
    "💬 Ask Questions": _page_ask,
    "🌊 Ocean Data": _page_ocean_data,
    "📊 Evaluation": _page_evaluation,
    "📈 Monitoring": _page_monitoring,
    "⚙️ Settings": _page_settings,
}


def main() -> None:
    cfg = get_config()

    st.sidebar.title("RAG Assistant")

    choice = st.sidebar.radio(
        "Navigate",
        list(PAGES.keys()),
    )

    st.sidebar.caption(
        f"Provider: {cfg.generation.provider} · "
        f"Model: {cfg.generation.model}"
    )

    PAGES[choice](cfg)


if __name__ == "__main__":
    main()