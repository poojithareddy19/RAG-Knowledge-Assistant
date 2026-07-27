from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.config import get_config
from src.utils.pipeline import RAGService

st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading models and vector store…")
def get_service() -> RAGService:
    return RAGService()


def _page_home(cfg):
    st.title("📚 Production RAG Knowledge Assistant")
    st.caption(f"v{cfg.app.version} — grounded answers with citations, confidence and monitoring")
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
        "**Pipeline:** question → embed → FAISS retrieve → confidence gate → "
        "grounded LLM answer → citations. Configure everything in `config.yaml` / `.env`."
    )


def _page_upload(cfg):
    st.title("📂 Upload Documents")
    svc = get_service()
    exts = [e.lstrip(".") for e in svc.supported_extensions()]
    st.write(f"Supported types: {', '.join(exts)}")
    files = st.file_uploader(
        "Upload one or more documents", type=exts, accept_multiple_files=True
    )
    dedup = st.checkbox("Skip documents already indexed (duplicate detection)", value=True)

    if files and st.button("Ingest & index", type="primary"):
        progress = st.progress(0.0)
        results = []
        for i, f in enumerate(files, start=1):
            # Write to a fresh unique temp directory using the *original*
            # filename (needed so citations show the real name). A per-file
            # directory avoids name collisions, so this works identically on
            # Windows and POSIX (Windows Path.rename raises on an existing
            # target, which the previous approach hit).
            tmp_dir = Path(tempfile.mkdtemp(prefix="rag_upload_"))
            tmp_path = tmp_dir / f.name
            tmp_path.write_bytes(f.getvalue())
            try:
                with st.spinner(f"Processing {f.name}…"):
                    results.append(svc.ingest_file(tmp_path, skip_duplicates=dedup))
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            progress.progress(i / len(files))
        st.success("Ingestion complete.")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        get_service.clear()  # refresh cached stats


def _page_knowledge_base(cfg):
    st.title("📚 Knowledge Base")
    svc = get_service()
    stats = svc.stats()
    c1, c2 = st.columns(2)
    c1.metric("Documents", stats["documents"])
    c2.metric("Total chunks", stats["chunks"])
    if not stats["manifest"]:
        st.warning("No documents indexed yet. Go to **Upload Documents**.")
        return
    rows = [
        {
            "Document": name,
            "Pages": m.get("pages", 0),
            "Chunks": m.get("chunks", 0),
            "Embedding model": m.get("embedding_model", ""),
            "Last indexed": m.get("indexed_at", ""),
        }
        for name, m in stats["manifest"].items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    if st.button("⚠️ Reset knowledge base"):
        svc.reset()
        get_service.clear()
        st.rerun()


def _page_ask(cfg):
    st.title("💬 Ask a Question")
    svc = get_service()
    if svc.stats()["chunks"] == 0:
        st.warning("Index some documents first.")
        return

    question = st.text_input("Your question", placeholder="e.g. When are employees eligible for parental leave?")
    if not (question and st.button("Ask", type="primary")):
        return

    with st.spinner("Retrieving and generating…"):
        ans = svc.ask(question)

    # Answer + confidence
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
        st.metric("Confidence", f"{ans.confidence.percent}%")
        st.progress(ans.confidence.score)
        st.caption(f"total {ans.latency_ms.get('total_ms', 0):.0f} ms")

    # Sources
    st.markdown("#### Sources" if ans.answered else "#### Closest passages")
    for s in ans.sources:
        with st.expander(
            f"[{s.rank}] {s.chunk.doc_name} — page {s.chunk.page} · sim {s.score:.3f} · {s.chunk.chunk_id}"
        ):
            st.write(s.chunk.text)

    # Explainability
    with st.expander("🔎 Retrieval details"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Rank": s.rank,
                        "Score": round(s.score, 4),
                        "Document": s.chunk.doc_name,
                        "Page": s.chunk.page,
                        "Chunk length": s.chunk.char_len,
                    }
                    for s in ans.sources
                ]
            ),
            use_container_width=True,
        )
        st.json(ans.confidence.to_dict())
    if ans.prompt:
        with st.expander("🧾 Exact prompt sent to the LLM"):
            st.code(ans.prompt)


def _page_evaluation(cfg):
    st.title("📈 Evaluation")
    svc = get_service()
    gold_path = Path(cfg.paths.processed_dir).parent / "evaluation" / "questions.csv"
    st.write(f"Gold dataset: `{gold_path}`")
    k = st.slider("k (top-k for metrics)", 1, 10, cfg.retrieval.top_k)
    if st.button("Run retrieval evaluation", type="primary"):
        if svc.stats()["chunks"] == 0:
            st.warning("Index the documents referenced by the gold set first.")
            return
        from src.evaluation.evaluator import evaluate_retrieval

        with st.spinner("Evaluating…"):
            report = evaluate_retrieval(svc.retriever, gold_path, k=k)
        st.metric("Questions", report["n_questions"])
        agg = report["aggregate"]
        st.bar_chart(pd.Series(agg))
        st.dataframe(pd.DataFrame([agg]), use_container_width=True)
        st.caption(
            "Generation metrics (faithfulness, groundedness, answer relevance) "
            "are computed via the LLM-judge / Ragas path — see docs."
        )


def _page_monitoring(cfg):
    st.title("📊 Monitoring")
    from src.monitoring.analytics import summarize

    s = summarize()
    if s.get("total_queries", 0) == 0:
        st.info("No interactions logged yet. Ask some questions first.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total queries", s["total_queries"])
    c2.metric("Avg latency", f"{s['avg_latency_ms']} ms")
    c3.metric("Fallback rate", f"{s['fallback_rate'] * 100:.0f}%")
    c4.metric("Errors", s["error_count"])
    c5, c6, c7 = st.columns(3)
    c5.metric("Avg confidence", f"{s['avg_confidence']}%")
    c6.metric("Avg LLM latency", f"{s['avg_llm_latency_ms']} ms")
    c7.metric("Avg retrieval sim", s["avg_retrieval_score"])

    st.markdown("##### Confidence distribution")
    st.bar_chart(pd.Series(s["confidence_values"], name="confidence"))
    st.markdown("##### Most queried documents")
    if s["most_queried_documents"]:
        st.dataframe(
            pd.DataFrame(s["most_queried_documents"], columns=["Document", "Retrievals"]),
            use_container_width=True,
        )


def _page_settings(cfg):
    st.title("⚙️ Settings")
    st.write("Read-only view of the active configuration. Edit `config.yaml` / `.env` to change.")
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


PAGES = {
    "🏠 Home": _page_home,
    "📂 Upload Documents": _page_upload,
    "📚 Knowledge Base": _page_knowledge_base,
    "💬 Ask Questions": _page_ask,
    "📈 Evaluation": _page_evaluation,
    "📊 Monitoring": _page_monitoring,
    "⚙️ Settings": _page_settings,
}


def main() -> None:
    cfg = get_config()
    st.sidebar.title("📚 RAG Assistant")
    choice = st.sidebar.radio("Navigate", list(PAGES.keys()))
    st.sidebar.caption(f"Provider: {cfg.generation.provider} · Model: {cfg.generation.model}")
    PAGES[choice](cfg)


if __name__ == "__main__":
    main()