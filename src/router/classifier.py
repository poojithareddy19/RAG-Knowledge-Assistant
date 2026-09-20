import json
import os
import re

import httpx

ROUTES = ("documents", "data", "chart", "both")


SYSTEM = """Classify the user's question into exactly one route.

documents - answered from uploaded text documents (policies, manuals, reports)
data - answered by querying the ocean measurements database (numbers,
       averages, counts, trends over time)
chart - the user explicitly wants a plot, chart, graph or visualisation
both - needs the manuals AND the database to be answered completely, such as
       a question asking what a term means and then for a count of it

Pick "both" only when neither half alone answers the question. A question
answerable from the database alone is "data" even if it mentions a manual.

Reply with JSON only: {"route": "...", "reason": "..."}
"""


# map, track and trajectory are here because in this domain they name a
# picture, not a number. "Where did float 1900083 go" is answered by drawing
# its path, and without these words it matches "float" in DATA_WORDS and comes
# back as a table of coordinates.
CHART_WORDS = re.compile(
    r"\b(plot|chart|graph|visuali[sz]e|draw|trend line|show me a"
    r"|map|track|trajector(?:y|ies)|where did)\b",
    re.I,
)


DATA_WORDS = re.compile(
    r"\b(average|mean|median|count|how many|per year|per month|between "
    r"\d{4}|temperature|salinity|floats?|profiles?|trend)\b",
    re.I,
)


# What names a written source rather than a measurement. "section" is here
# for a numbered section of a manual, which is why the chart test runs first:
# a depth-time section is a picture and matches CHART_WORDS on its verb.
DOC_WORDS = re.compile(
    r"\b(policy|document|handbook|contract|clause|section"
    r"|manual|guideline|glossary|specification"
    r"|according to|say about|defines?|definition|defined)\b",
    re.I,
)


# A question naming a manual and a measurement in one breath is usually
# still one question: "what does the manual say about temperature limits"
# needs no database. What makes it two is asking for a quantity as well,
# so the combined route keys off the counting words rather than the nouns.
AGGREGATION_WORDS = re.compile(
    r"\b(average|mean|median|count|how many|how much|total"
    r"|per year|per month|trend|maximum|minimum|highest|lowest)\b",
    re.I,
)


def rule_route(question):
    """Fast, free, handles the unambiguous cases. None means 'ask the model'."""
    if CHART_WORDS.search(question):
        return "chart"

    documents = DOC_WORDS.search(question)
    data = DATA_WORDS.search(question)

    # Naming a written source and asking for a number is the shape of a
    # question that neither half answers on its own.
    if documents and AGGREGATION_WORDS.search(question):
        return "both"

    if data and not documents:
        return "data"

    return None


def model_route(question, model=None, timeout=30):
    base = os.environ.get(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )

    model = model or os.environ.get("ROUTER__MODEL", "llama3.1:latest",
    )

    r = httpx.post(
        f"{base}/api/generate",
        json={
            "model": model,
            "prompt": f"{SYSTEM}\n\nQuestion: {question}\nJSON:",
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": 80,
            },
        },
        timeout=timeout,
    )

    r.raise_for_status()

    payload = json.loads(r.json()["response"])

    route = str(payload.get("route", "")).strip().lower()

    if route not in ROUTES:
        raise ValueError(
            f"model returned unknown route: {route!r}"
        )

    return route, payload.get("reason", "")


def route(question, fallback="documents", use_model=True):
    """Returns (route, how_it_was_decided)."""
    fast = rule_route(question)

    if fast:
        return fast, "rules"

    if use_model:
        try:
            picked, _ = model_route(question)
            return picked, "model"
        except Exception:
            pass

    return fallback, "fallback"
