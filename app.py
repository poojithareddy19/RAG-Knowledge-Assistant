from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.config import get_config
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
        "Two kinds of question, one assistant. **What does the archive hold?** "
        "is answered from summaries of every float and region in the database, "
        "with the float or region behind every claim. **What does the data "
        "say?** is answered by SQL generated against the measurements, shown to "
        "you before you are asked to believe the numbers. Either way, when the "
        "evidence is too weak it declines instead of guessing."
    )

    svc = get_service()
    stats = svc.stats()

    c1, c2, c3 = st.columns(3)
    c1.metric("Summaries indexed", stats["summaries"])
    c2.metric("Embedding model", cfg.embeddings.model_name.split("/")[-1])
    c3.metric("Language model", cfg.generation.model)

    st.info(
        "**Summaries path:** question → embed → retrieve the nearest float and "
        "region summaries → confidence gate → grounded answer → citations.\n\n"
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
        / "summary_questions.csv"
    )

    st.write(f"Gold dataset: `{gold_path}`")

    k = st.slider(
        "k (top-k for metrics)",
        1,
        10,
        int(cfg.semantic.top_k),
    )

    if st.button("Run retrieval evaluation", type="primary"):
        if svc.stats()["summaries"] == 0:
            st.warning(
                "Build the semantic index first: "
                "python scripts/build_semantic_index.py"
            )
            return

        from src.evaluation.evaluator import evaluate_retrieval

        with st.spinner("Evaluating…"):
            report = evaluate_retrieval(
                svc.semantic,
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
            "Retrieval only: whether the float or region the gold set names "
            "is among the top-k summaries. The prose is scored separately by "
            "src/evaluation/generation_metrics.py, and the SQL by "
            "src/evaluation/sql_metrics.py."
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

    st.markdown("##### Most retrieved floats and regions")

    if summary["most_retrieved_subjects"]:
        st.dataframe(
            pd.DataFrame(
                summary["most_retrieved_subjects"],
                columns=["Subject", "Retrievals"],
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
            "summaries_table": cfg.semantic.table,
            "top_k": cfg.semantic.top_k,
            "min_similarity": cfg.semantic.min_similarity,
            "answer_threshold": cfg.semantic.answer_threshold,
            "confidence_weights": dict(cfg.semantic.confidence.weights),
            "provider": cfg.generation.provider,
            "model": cfg.generation.model,
            "sql_model": cfg.sql.model,
            "router_fallback": cfg.router.fallback,
        }
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

    This was two pages. "Ask Questions" ran the retrieval path alone and showed
    citations and a confidence score; "Ocean Data" ran the router, and when the
    router chose retrieval it printed the answer and told the reader to go to
    the other page for the evidence. That is the routed path admitting it could
    not show its own working, and the sources were in the result all along.

    So the router decides, and whichever source answers shows what it rests on:
    the float or region for a summary claim, the query for a table. Asking the
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
        "One question box over the float and region summaries and the "
        "measurements database. The question is routed; a summary answer "
        "names the float or region it rests on, and a data answer carries the "
        "SQL that produced it, shown before you are asked to believe the "
        "numbers. Follow-ups work, and so do other languages: ask in Hindi, "
        "Tamil or Bengali and the answer comes back in the same language."
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




def _render_summaries(result):
    """A summary answer, with the evidence it rests on.

    This used to print the answer and send the reader to a second page for the
    citations, which was the routed path admitting it could not show its own
    working. The sources were in the result the whole time; nothing rendered
    them. An answer without the float or region it was written from is the
    thing the confidence gate exists to prevent, so it is shown here.
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
            "**Sources**" if result.get("answered") else "**Closest summaries**"
        )

        for source in sources:
            with st.expander(
                f"[{source.get('rank', '?')}] "
                f"{source.get('kind', 'source')} "
                f"{source.get('subject', '?')} · "
                f"sim {float(source.get('score') or 0.0):.3f}"
            ):
                st.write(source.get("text", ""))


def _render_answer(svc, question, result, position):
    """One assistant turn: the answer and everything backing it.

    ``position`` keeps the widget keys unique. Streamlit raises on two charts
    sharing a key, and a conversation is many turns each drawing its own.
    """
    st.write(
        f"**Route:** `{result['route']}` "
        f"(decided by {result['route_decided_by']})"
    )

    rewritten = result.get("question_rewritten")

    if rewritten and rewritten != question:
        # Shown rather than hidden, on every route: if the follow up was
        # resolved into the wrong question, or the translation changed its
        # meaning, this line is the only way to see it. It was once shown for
        # data answers only, and a rewrite that turned a named float into
        # "that float" went unnoticed on the summaries route because of it.
        st.caption(f"Answered as: {rewritten}")

    if result.get("language"):
        st.caption(
            f"Detected {result['language']}. The query ran in English and the "
            "answer was translated back."
        )

        if result.get("answer_english"):
            with st.expander("Answer before translation"):
                st.write(result["answer_english"])

    if result["route"] == "summaries":
        _render_summaries(result)
        return

    if result.get("refused"):
        st.warning(result["answer"])

        with st.expander("SQL the model produced (rejected)"):
            st.code(result.get("generated_sql") or "", language="sql")

        return

    st.success(result["answer"])

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