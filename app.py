from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.config import get_config
from src.utils.export import to_csv_bytes, to_netcdf_bytes
from src.utils.pipeline import RAGService

st.set_page_config(
    page_title="FloatChat",
    page_icon="🌊",
    layout="wide",
)


# Plotly fetches the land and coastlines of a geo plot as TopoJSON at render
# time, and defaults to https://cdn.plot.ly/. Everything else here runs against
# a local database and a local model, so an ocean map that quietly comes up
# empty without internet access is the one networked dependency in the app.
# The files are committed under static/, served by Streamlit at /app/static/,
# and Plotly is pointed there. It appends "world_<resolution>m.json", so the
# files sit flat in static/ rather than under the un/ path the CDN uses.
PLOTLY_CONFIG = {"topojsonURL": "/app/static/"}


@st.cache_resource(show_spinner="Loading models and vector store…")
def get_service() -> RAGService:
    return RAGService()


def _page_home(cfg):
    st.title("FloatChat")
    st.caption(
        f"v{cfg.app.version}: ask the Argo float archive a question "
        "and get an answer you can check"
    )

    st.markdown(
        "Two kinds of question, one assistant. **What does this data mean?** is "
        "answered from the Argo manuals, with the document and page behind every "
        "claim. **What does this data say?** is answered by SQL generated against "
        "the measurements database, shown to you before you are asked to believe "
        "the numbers. Either way, when the evidence is too weak it declines "
        "instead of guessing."
    )

    svc = get_service()
    stats = svc.stats()

    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", stats["documents"])
    c2.metric("Indexed chunks", stats["chunks"])
    c3.metric("Embedding dim", stats["dimension"])

    st.info(
        "**Manual path:** question → embed → vector retrieve → confidence gate "
        "→ grounded answer → citations.\n\n"
        "**Data path:** question → retrieve what the database holds → generate "
        "SQL → validate → run read-only → table plus the query that made it.\n\n"
        "Configure everything in `config.yaml` / `.env`."
    )


def _page_ask(cfg):
    st.title("Ask a Question")

    svc = get_service()

    if svc.stats()["chunks"] == 0:
        st.warning(
            "The Argo manuals are not indexed yet. See **Load the manuals** "
            "in the README."
        )
        return

    question = st.text_input(
        "Your question",
        placeholder=(
            "e.g. What does a quality control flag of 4 mean?"
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
            f"{source.chunk.doc_name}, "
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
            width="stretch",
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
            width="stretch",
        )

        st.caption(
            "Generation metrics (faithfulness, groundedness, "
            "answer relevance) are computed via the LLM-judge / "
            "Ragas path, see docs."
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
            width="stretch",
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


def _download_buttons(svc, result, question, position=0):
    """Offer the result set as CSV and as NetCDF.

    Both, because they answer different questions. CSV opens anywhere and
    carries no units; the NetCDF carries the units, the column descriptions and
    the SQL that produced it, so the download stays reproducible after it
    leaves this page.
    """
    # The table above shows the first hundred rows; the file is all of them.
    full = svc.complete_result(result)

    csv_column, netcdf_column = st.columns(2)

    with csv_column:
        st.download_button(
            "Download CSV",
            data=to_csv_bytes(full),
            file_name="result.csv",
            key=f"csv_{position}",
            mime="text/csv",
            width="stretch",
        )

    with netcdf_column:
        try:
            blob = to_netcdf_bytes(full, title=question)
        except Exception as exc:
            st.caption(f"NetCDF export unavailable: {exc}")
            return

        st.download_button(
            "Download NetCDF",
            data=blob,
            file_name="result.nc",
            key=f"nc_{position}",
            mime="application/x-netcdf",
            width="stretch",
        )


def _remember(cfg, history, question, answer):
    """Append this exchange, keeping only the turns the rewriter may use.

    Lives in session state, so it is per browser session and disappears with
    it. That is the honest scope of the feature: nothing here is persisted.
    """
    history.append((question, str(answer)[:400]))

    cap = int(
        cfg.get("conversation", {}).get("max_turns", 4)
    )

    if cap > 0 and len(history) > cap:
        del history[:-cap]


def _page_ocean_data(cfg):
    """Conversational interface over the measurements database.

    A thread rather than a single question box, because the system supports
    follow-ups and one question-and-answer pane hides that. Each turn keeps
    its own chart, table and SQL, so scrolling back shows what was asked and
    what answered it rather than only the latest result.
    """

    svc = get_service()

    st.header("🌊 Ocean Data")

    st.caption(
        "Ask about ARGO floats and drifting buoys. The question is routed, "
        "the SQL is generated, validated and shown to you before you are "
        "asked to believe the numbers. Follow-ups work, and so do other "
        "languages: ask in Hindi, Tamil or Bengali and the answer comes back "
        "in the same language."
    )

    # Two stores doing different jobs. turns is everything needed to redraw
    # the thread; history is the (question, answer) pairs the rewriter reads,
    # capped at conversation.max_turns.
    turns = st.session_state.setdefault("ocean_turns", [])
    history = st.session_state.setdefault("ocean_history", [])

    if turns and st.button("Clear conversation", key="ocean_clear"):
        turns.clear()
        history.clear()
        st.rerun()

    for position, turn in enumerate(turns):
        with st.chat_message("user"):
            st.write(turn["question"])

        with st.chat_message("assistant"):
            _render_answer(svc, turn["question"], turn["result"], position)

    question = st.chat_input(
        "e.g. average surface temperature per year in the Arabian Sea"
    )

    if not question or not question.strip():
        return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Routing, generating SQL, running query…"):
            try:
                # A copy, because the service must not see the turn being asked.
                result = svc.answer(question, history=list(history))
            except Exception as exc:
                st.error(f"The query could not be completed: {exc}")
                return

    _remember(cfg, history, question, result.get("answer", ""))
    turns.append({"question": question, "result": result})

    # Rerun rather than draw the answer here. The replay loop above is then the
    # only thing that renders a turn, and a turn drawn once live and once on
    # the next pass was appending its text to itself in the same container.
    st.rerun()


def _render_answer(svc, question, result, position):
    """One assistant turn: the answer and everything backing it.

    ``position`` keeps the widget keys unique. Streamlit raises on two
    download buttons sharing a key, and a conversation is many turns each
    offering the same two downloads.
    """
    st.write(
        f"**Route:** `{result['route']}` "
        f"(decided by {result['route_decided_by']})"
    )

    if result["route"] == "documents":
        st.info(
            "This was routed to the Argo manuals, not the database. "
            "Ask it on the Ask Questions page for citations and confidence."
        )
        st.write(result["answer"])
        return

    if result.get("refused"):
        st.warning(result["answer"])

        with st.expander("SQL the model produced (rejected)"):
            st.code(result.get("generated_sql") or "", language="sql")

        return

    st.success(result["answer"])

    rewritten = result.get("question_rewritten")

    if rewritten and rewritten != question:
        # Shown rather than hidden: if the follow up was resolved into the
        # wrong question, or the translation changed its meaning, this line is
        # the only way to see it.
        st.caption(f"Answered as: {rewritten}")

    if result.get("language"):
        st.caption(
            f"Detected {result['language']}. The query ran in English and the "
            "answer was translated back."
        )

        if result.get("answer_english"):
            with st.expander("Answer before translation"):
                st.write(result["answer_english"])

    if result.get("chart_spec"):
        # A domain plot: a track, a cast, a section or a T-S cloud. Interactive,
        # because reading a position off a static map is guesswork.
        st.plotly_chart(
            result["chart_spec"],
            width="stretch",
            config=PLOTLY_CONFIG,
            key=f"chart_{position}",
        )
    elif result.get("chart_png"):
        st.image(result["chart_png"])
    elif result.get("chart_kind") == "table":
        st.caption("This result set is not chartable, so here is the table.")

    if result.get("rows"):
        st.dataframe(
            pd.DataFrame(result["rows"], columns=result["columns"]),
            width="stretch",
        )

        _download_buttons(svc, result, question, position)

    if result.get("context_used"):
        with st.expander("What the model was told the database contains"):
            for note in result["context_used"]:
                st.markdown(f"- {note}")

            st.caption(
                "Retrieved before the SQL was written. These are summaries "
                "used to resolve names and ranges; every number in the answer "
                "is computed by the query, not read from here."
            )

    with st.expander("Generated SQL and timing"):
        st.code(result["generated_sql"], language="sql")

        st.write(
            f"{result['row_count']} rows in "
            f"{result['elapsed_ms']} ms end to end"
            + (
                f" · {result['db_elapsed_ms']} ms in the database"
                if result.get("db_elapsed_ms") is not None
                else ""
            )
            + (
                " · SQL served from cache"
                if result.get("sql_cached")
                else ""
            )
        )


PAGES = {
    "🏠 Home": _page_home,
    "💬 Ask Questions": _page_ask,
    "🌊 Ocean Data": _page_ocean_data,
    "📊 Evaluation": _page_evaluation,
    "📈 Monitoring": _page_monitoring,
    "⚙️ Settings": _page_settings,
}


def main() -> None:
    cfg = get_config()

    st.sidebar.title("FloatChat")

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