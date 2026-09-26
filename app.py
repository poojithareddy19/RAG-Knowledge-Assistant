from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.router.tools import with_location
from src.utils.config import get_config
from src.utils.export import to_csv_bytes, to_netcdf_bytes, to_parquet_bytes
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
    """Offer the result set as CSV, Parquet and NetCDF.

    Three, because they answer different questions. CSV is plain ASCII text and
    opens anywhere, with no units. Parquet keeps the types, for pandas or Spark.
    The NetCDF carries the units, the column descriptions and the SQL that
    produced it, so the download stays reproducible after it leaves this page.
    """
    # The table above shows the first hundred rows; the file is all of them.
    full = svc.complete_result(result)

    csv_column, parquet_column, netcdf_column = st.columns(3)

    with csv_column:
        st.download_button(
            "Download CSV (ASCII)",
            data=to_csv_bytes(full),
            file_name="result.csv",
            key=f"csv_{position}",
            mime="text/csv",
            width="stretch",
        )

    with parquet_column:
        try:
            parquet = to_parquet_bytes(full, title=question)
        except Exception as exc:
            st.caption(f"Parquet export unavailable: {exc}")
        else:
            st.download_button(
                "Download Parquet",
                data=parquet,
                file_name="result.parquet",
                key=f"pq_{position}",
                mime="application/vnd.apache.parquet",
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


def _page_ask(cfg):
    """One question box over both sources.

    This was two pages. "Ask Questions" ran the manuals alone and showed
    citations and a confidence score; "Ocean Data" ran the router, and when the
    router chose the manuals it printed the answer and told the reader to go to
    the other page for the evidence. That is the routed path admitting it could
    not show its own working, and the sources were in the result all along.

    So the router decides, and whichever source answers shows what it rests on:
    the document and page for a manual claim, the query for a table. Asking the
    user to pick the backend first was asking them to know the answer before
    they asked the question.

    A thread rather than a single question box, because the system supports
    follow-ups and one question-and-answer pane hides that. Each turn keeps its
    own chart, table and SQL, so scrolling back shows what was asked and what
    answered it rather than only the latest result.
    """

    svc = get_service()

    st.header("💬 Ask")

    st.caption(
        "One question box over the Argo manuals and the measurements database. "
        "The question is routed; a manual answer carries its document and page, "
        "and a data answer carries the SQL that produced it, shown before you "
        "are asked to believe the numbers. Follow-ups work, and so do other "
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

    location = _location_panel()

    question = st.chat_input(
        "e.g. average surface temperature per year in the Arabian Sea"
    )

    if not question or not question.strip():
        return

    # What was typed is what the thread shows. What is sent may carry the
    # panel's position, and that version is shown under the answer as
    # "Answered as", so the substitution is never invisible.
    sent = with_location(question, location)

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Routing, generating SQL, running query…"):
            try:
                # A copy, because the service must not see the turn being asked.
                result = svc.answer(sent, history=list(history))
            except Exception as exc:
                st.error(f"The query could not be completed: {exc}")
                return

    _remember(cfg, history, question, result.get("answer", ""))
    turns.append({"question": question, "result": result})

    # Rerun rather than draw the answer here. The replay loop above is then the
    # only thing that renders a turn, and a turn drawn once live and once on
    # the next pass was appending its text to itself in the same container.
    st.rerun()




def _location_panel():
    """A position for questions that say "here", or None when switched off.

    The problem statement's third example is "what are the nearest ARGO floats
    to this location?", and a chat box has no location. This supplies one. It
    is off unless ticked, so a question that happens to say "here" is not
    silently pinned to a point the user set an hour ago and forgot.
    """
    with st.expander("📍 Location for \"near here\" questions"):
        use = st.checkbox(
            "Use this position when a question refers to a location",
            key="location_use",
        )

        left, right = st.columns(2)

        latitude = left.number_input(
            "Latitude (°N, south negative)",
            min_value=-90.0,
            max_value=90.0,
            value=10.0,
            step=0.5,
            key="location_lat",
        )

        longitude = right.number_input(
            "Longitude (°E, west negative)",
            min_value=-180.0,
            max_value=180.0,
            value=65.0,
            step=0.5,
            key="location_lon",
        )

    return (latitude, longitude) if use else None


def _render_tool(svc, question, result, position):
    """A turn answered by one of the MCP server's fixed tools."""
    if result.get("refused"):
        st.warning(result["answer"])
        return

    st.success(result["answer"])

    rewritten = result.get("question_rewritten")

    if rewritten and rewritten != question:
        st.caption(f"Answered as: {rewritten}")

    if result.get("chart_spec"):
        st.plotly_chart(
            result["chart_spec"],
            width="stretch",
            config=PLOTLY_CONFIG,
            key=f"chart_{position}",
        )

    if result.get("rows"):
        st.dataframe(
            pd.DataFrame(result["rows"], columns=result["columns"]),
            width="stretch",
        )

        _download_buttons(svc, result, question, position)

    with st.expander("MCP tool call and timing"):
        # In place of the SQL a generated answer carries. The thing to check is
        # that the right tool ran with the right arguments, and the statement it
        # ran is fixed in src/mcp/server.py rather than written for this turn.
        st.code(
            json.dumps(
                {"tool": result.get("mcp_tool"), "arguments": result.get("mcp_arguments")},
                indent=2,
            ),
            language="json",
        )

        st.write(
            f"{result.get('row_count', 0)} rows in {result.get('elapsed_ms')} ms "
            "end to end, over the Model Context Protocol"
        )


def _render_combined(svc, question, result, position):
    """A turn answered from the manuals and the database at once.

    The synthesised sentence is shown first because it is what was asked for,
    and both halves are shown underneath because it is the one answer in the
    system written from two sources, which makes it the one a reader most
    needs to be able to take apart.
    """
    if result.get("refused"):
        st.warning(result["answer"])
        return

    st.success(result["answer"])

    if result.get("combined_by") == "stapled":
        st.caption(
            "Shown as two answers rather than one. The sentence joining them "
            "could not be written, so each source appears as it came back."
        )

    parts = result.get("parts", {})
    documents = parts.get("documents", {})
    data = parts.get("data", {})

    left, right = st.columns(2)

    with left:
        with st.expander("From the manuals", expanded=False):
            st.caption(f"Asked as: {result.get('question_documents', '')}")

            if documents.get("answered"):
                st.write(documents.get("answer", ""))

                for source in documents.get("sources", [])[:3]:
                    st.caption(
                        f"{source.get('doc_name', '')} "
                        f"page {source.get('page', '')}"
                    )
            else:
                st.info(
                    "The manuals had nothing confident to add, so the answer "
                    "rests on the data alone."
                )

    with right:
        with st.expander("From the database", expanded=False):
            st.caption(f"Asked as: {result.get('question_data', '')}")

            if data.get("answered"):
                st.write(data.get("answer", ""))
                st.code(data.get("generated_sql") or "", language="sql")
            else:
                st.info(data.get("answer", "No rows were returned."))

    if result.get("rows"):
        st.dataframe(
            pd.DataFrame(result["rows"], columns=result["columns"]),
            width="stretch",
        )

        _download_buttons(svc, result, question, position)


def _render_documents(result):
    """A manual answer, with the evidence it rests on.

    This used to print the answer and send the reader to a second page for the
    citations, which was the routed path admitting it could not show its own
    working. The sources were in the result the whole time; nothing rendered
    them. An answer from the manuals without its document and page is the thing
    the confidence gate exists to prevent, so it is shown here.
    """
    if not result.get("answered"):
        st.warning(result["answer"])

        if result.get("reason"):
            st.caption(f"Why: {result['reason']}")
    else:
        st.write(result["answer"])

    sources = result.get("sources") or []

    left, right = st.columns([3, 1])

    with right:
        percent = result.get("confidence_percent")

        if percent is None:
            percent = round(float(result.get("confidence") or 0.0) * 100)

        st.metric("Confidence", f"{percent}%")
        st.progress(min(max(float(result.get("confidence") or 0.0), 0.0), 1.0))

    with left:
        if not sources:
            return

        st.markdown(
            "**Sources**" if result.get("answered") else "**Closest passages**"
        )

        for source in sources:
            chunk = source.get("chunk", {})

            with st.expander(
                f"[{source.get('rank', '?')}] "
                f"{chunk.get('doc_name', 'source')}, "
                f"page {chunk.get('page', '?')} · "
                f"sim {float(source.get('score') or 0.0):.3f}"
            ):
                st.write(chunk.get("text", ""))


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
        _render_documents(result)
        return

    if result["route"] == "both":
        _render_combined(svc, question, result, position)
        return

    if result["route"] == "tool":
        _render_tool(svc, question, result, position)
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
    "💬 Ask": _page_ask,
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