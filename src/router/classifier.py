import json
import os
import re

import httpx

from src.generation.llm import num_ctx
from src.monitoring.tracing import llm_span, record_ollama

ROUTES = ("summaries", "data", "chart")


SYSTEM = """Classify the user's question into exactly one route.

summaries - answered from short descriptions of what the database holds:
            which floats and regions exist, when they reported, where,
            how deep, and which sensors they carry
data - answered by querying the ocean measurements database (numbers,
       averages, counts, trends over time)
chart - the user explicitly wants a plot, chart, graph or visualisation

A question that needs a number computed from the measurements is "data".
A question about what is available, or about one float or region in
general, is "summaries".

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


# What asks for a description rather than a computation. These phrasings ask
# what the database holds or what one float or region is like, which the
# summaries answer directly and a SQL query answers only by accident.
SUMMARY_WORDS = re.compile(
    r"\b(tell me about|describe|what do you know about|what is known about"
    r"|overview|what data (?:do you have|is (?:there|available))"
    r"|is there (?:any )?data|which regions? (?:are|is) covered|coverage"
    r"|what sensors|which sensors|what does .{0,40}\bmeasure"
    # Which floats carry a sensor is a fact the summaries state outright.
    # Sent to SQL, the model invented a dissolved_oxygen_qc column and the
    # scope guard refused a question the index could answer.
    r"|which floats? (?:measure|carry|carries|record)"
    # Up to three words for the sensor name: "a dissolved oxygen sensor".
    r"|(?:carry|carries|carrying|with) (?:an? )?(?:\w+ ){0,3}sensors?)\b",
    re.I,
)


def rule_route(question):
    """Fast, free, handles the unambiguous cases. None means 'ask the model'."""
    if CHART_WORDS.search(question):
        return "chart"

    if SUMMARY_WORDS.search(question):
        return "summaries"

    if DATA_WORDS.search(question):
        return "data"

    return None


def model_route(question, model=None, timeout=30):
    base = os.environ.get(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )

    model = model or os.environ.get("ROUTER__MODEL", "llama3.1:latest",
    )

    prompt = f"{SYSTEM}\n\nQuestion: {question}\nJSON:"

    with llm_span("route", model, temperature=0, max_tokens=80) as span:
        r = httpx.post(
            f"{base}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0,
                    "num_predict": 80,
                    "num_ctx": num_ctx(),
                },
            },
            timeout=timeout,
        )

        r.raise_for_status()

        data = r.json()
        record_ollama(span, data, prompt=prompt)

    payload = json.loads(data["response"])

    route = str(payload.get("route", "")).strip().lower()

    if route not in ROUTES:
        raise ValueError(
            f"model returned unknown route: {route!r}"
        )

    return route, payload.get("reason", "")


def route(question, fallback="summaries", use_model=True):
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
