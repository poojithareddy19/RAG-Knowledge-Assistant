import json
import os
import re

import httpx

ROUTES = ("documents", "data", "chart")


SYSTEM = """Classify the user's question into exactly one route.

documents - answered from uploaded text documents (policies, manuals, reports)
data - answered by querying the ocean measurements database (numbers,
       averages, counts, trends over time)
chart - the user explicitly wants a plot, chart, graph or visualisation

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


def rule_route(question):
    """Fast, free, handles the unambiguous cases. None means 'ask the model'."""
    if CHART_WORDS.search(question):
        return "chart"

    if DATA_WORDS.search(question) and not re.search(
        r"\b(policy|document|handbook|contract|clause|section)\b",
        question,
        re.I,
    ):
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
