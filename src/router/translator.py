"""Answer a question asked in another language, in that language.

The problem statement comes from an Indian ministry, so the languages that
matter most here are Hindi, Bengali, Tamil, Telugu, Malayalam and their
neighbours. All of them are written in their own scripts, which is what makes
the cheap half of this work: a question containing Devanagari is not English,
and no model call is needed to establish that.

The design mirrors the follow-up rewriter next door. Translate at the edge,
keep one language inside. The router, the SQL generator, the schema catalog
and the cache all continue to see English only, so nothing downstream learns
about language and the SQL cache does not fragment by it.

Two calls are spent when a question is not in English: one to bring it into
English, one to return the answer. English questions cost nothing, because
detection is a script test rather than a model call.

The honest limit of that trade is written down in ``looks_english``: a
question in French or Indonesian is Latin script and will be taken for
English. Setting ``multilingual.always_translate`` routes every question
through the model instead, which catches those at the cost of one call on
every question.
"""

from __future__ import annotations

import os
import re
import unicodedata

import httpx

from src.utils.config import get_config

# Scripts whose presence settles the question without asking a model. Keyed by
# the name unicodedata gives the character, which is stable across versions.
_SCRIPTS = {
    "DEVANAGARI": "Hindi or Marathi",
    "BENGALI": "Bengali",
    "TAMIL": "Tamil",
    "TELUGU": "Telugu",
    "MALAYALAM": "Malayalam",
    "KANNADA": "Kannada",
    "GUJARATI": "Gujarati",
    "GURMUKHI": "Punjabi",
    "ORIYA": "Odia",
    "SINHALA": "Sinhala",
    "ARABIC": "Arabic or Urdu",
    "CJK": "Chinese",
    "HIRAGANA": "Japanese",
    "KATAKANA": "Japanese",
    "HANGUL": "Korean",
    "CYRILLIC": "Russian",
    "GREEK": "Greek",
    "THAI": "Thai",
    "HEBREW": "Hebrew",
}

TO_ENGLISH = """You translate a question into English.

Rules:
- Output only the translated question. No prose, no explanation, no quotes.
- Translate the words. Do not answer the question, and do not add a place, a
  date or a quantity the question does not contain.
- Leave identifiers alone: a float number, a WMO id and a column name are not
  words to translate.
- If the question is already English, repeat it back unchanged.
"""

FROM_ENGLISH = """You translate an answer into {language}.

Rules:
- Output only the translation. No prose, no explanation, no quotes.
- Numbers, units, dates and identifiers stay exactly as they are.
- Do not add anything the answer does not say, and do not round or reword a
  number.
"""

MAX_CHARS = 600


def looks_english(text: str) -> tuple[bool, str | None]:
    """Is this Latin script, and if not, which script is it?

    Returns ``(True, None)`` for Latin text, which is the cheap and common
    case, and ``(False, language)`` when another script is present. The second
    value is a best guess at the language from the script alone, good enough to
    tell a model what to translate back into and not a claim of certainty:
    Devanagari is Hindi or Marathi, and this cannot tell them apart.
    """
    for character in text:
        if character.isspace() or not character.isalpha():
            continue

        name = unicodedata.name(character, "")

        for script, language in _SCRIPTS.items():
            if name.startswith(script):
                return False, language

    return True, None


def to_english(question: str) -> tuple[str, str | None]:
    """The question in English, and the language it arrived in.

    Returns the question unchanged with a language of ``None`` when it is
    already English, when the feature is off, or when the model call fails.
    A translation that cannot be made must cost the user nothing more than the
    behaviour they had before this module existed.
    """
    settings = _settings()

    if not settings.get("enabled", True):
        return question, None

    english, language = looks_english(question)

    if english and not settings.get("always_translate", False):
        return question, None

    try:
        translated = _clean(_call_model(TO_ENGLISH, question))
    except Exception:
        return question, None

    if not translated:
        return question, None

    # always_translate asks the model about English questions too, and the
    # answer for those is the question itself. Report no language so nothing
    # downstream tries to translate the answer back.
    if english and translated.lower() == question.strip().lower():
        return question, None

    return translated, language or "the language of the question"


def from_english(answer: str, language: str | None) -> str:
    """The answer in the language the question arrived in.

    Returns the English answer when there is no language to go back to, when
    the feature is off, or when the call fails, because an answer the user can
    read in the wrong language beats an error in the right one.
    """
    if not language or not answer:
        return answer

    settings = _settings()

    if not settings.get("enabled", True):
        return answer

    if not settings.get("answer_in_source_language", True):
        return answer

    try:
        translated = _clean(
            _call_model(FROM_ENGLISH.format(language=language), answer)
        )
    except Exception:
        return answer

    return translated or answer


def _clean(text: str) -> str:
    """The reply without the quotes and preambles a model likes to add."""
    stripped = str(text).strip()

    if not stripped:
        return ""

    line = stripped.splitlines()[0].strip()

    # Prefix before quotes, not after: the model writes Translation: "text",
    # so stripping quotes first leaves the opening one stranded.
    line = re.sub(r"^(translation|in english|english)\s*:\s*", "", line, flags=re.I)
    line = line.strip().strip('"').strip("'").strip()

    return line[:MAX_CHARS]


def _settings() -> dict:
    try:
        return dict(get_config().get("multilingual", {}) or {})
    except Exception:
        return {}


def _call_model(system: str, text: str, timeout: int = 60) -> str:
    """One translation call. Separated so tests can stub a single thing."""
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("GENERATION__MODEL", "llama3.1:latest")

    response = httpx.post(
        f"{base}/api/generate",
        json={
            "model": model,
            "prompt": f"{system}\n\n=== TEXT ===\n{text}\n\nOutput:",
            "stream": False,
            "options": {"temperature": 0, "num_predict": 200},
        },
        timeout=timeout,
    )

    response.raise_for_status()

    return response.json()["response"]
