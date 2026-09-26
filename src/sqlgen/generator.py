"""Generate safe PostgreSQL SELECT statements with optional caching."""

from __future__ import annotations

import hashlib
import os
import re

import httpx

from src.generation.llm import keep_alive, num_ctx
from src.monitoring.tracing import llm_span, record_ollama
from src.sqlgen.schema_context import build_context, data_coverage
from src.sqlgen.scope import out_of_scope
from src.utils import cache

SYSTEM = """You write PostgreSQL SELECT queries.

Rules:
- Output exactly one SQL statement and nothing else. No prose, no markdown.
- SELECT only. Never INSERT, UPDATE, DELETE, DROP, ALTER or CREATE.
- Use only the tables and columns in the schema below. Never assume a column
  exists on a table just because it exists on another one. qc_flag lives on
  measurements only.
- Always exclude rows where qc_flag <> 1 and where the measured value is NULL.
- When the question names a period (a year, a month, a range), bound it at
  BOTH ends. "in 2020" means obs_time >= '2020-01-01' AND obs_time <
  '2021-01-01', never an open-ended >= alone.
- Whenever the result is a series or a grouped breakdown, end with ORDER BY on
  the grouping column so the rows come back in a meaningful order. Do not add
  ORDER BY to a query that returns one aggregate row and has no GROUP BY:
  ordering by a column you did not group by is rejected by PostgreSQL.
- Never alias a column to a name that means something different. If the
  question asks for a quantity no column holds, that is not a licence to
  rename the nearest one.
- When the question asks where a float went, or asks for a map, a track or a
  trajectory, select latitude and longitude along with obs_time. A position is
  the answer to that question, and a query returning only a count or a date
  cannot be drawn on a map.
- If the question cannot be answered from this schema, output exactly:
UNANSWERABLE
"""

CONTEXT_RULE = """The notes below describe what this database actually holds:
which floats exist, where they worked and when. Use them only to work out what
the question refers to, such as which float id or which region name.

They are background, not query terms:

- Do not turn a value from a note into a WHERE condition unless the question
  itself asks for it. A note saying a float is an APEX platform on project
  INCOIS does not mean the question is about APEX platforms or that project.
  Those notes describe every float in the database, not a filter.
- Do not narrow a query to the dates or ranges a note happens to mention. If
  the question names no period, do not bound one.
- Never copy a number out of a note as an answer. Compute every number with
  SQL, even when a note appears to state it already.
"""

# Invalidation is automatic: cache_version digests the whole prompt, so any
# edit here already retires the old entries. This constant is only a
# human-readable marker of prompt lineage.
PROMPT_VERSION = "14"

# One worked example rather than a rule alone, because the rule says what to
# select and the example shows the shape: ordered by time, one row per cycle.
TRACK_EXAMPLE = """=== EXAMPLE ===
Question:\tShow the track of float 1900083
SQL:
SELECT float_id,
       obs_time,
       latitude,
       longitude
FROM profiles
WHERE float_id = 1900083
ORDER BY obs_time
"""

# The window bucket was the worst on the board, 1 in 3, and the failures were
# not arithmetic: the model reached for a subquery and a self join where a
# window function was wanted, or wrote one without the OVER clause. Three
# worked examples, one per function, against these tables and these column
# names. Whether this moves the bucket is measured, not assumed, and the
# before and after are both published in the README.
WINDOW_EXAMPLES = """=== WINDOW FUNCTION EXAMPLES ===
Question:\tYear over year change in mean surface temperature
SQL:
WITH yearly AS (
    SELECT date_trunc('year', p.obs_time) AS yr,
           avg(m.temperature_c) AS mean_temp
    FROM measurements m
    JOIN profiles p ON p.profile_id = m.profile_id
    WHERE m.pressure_dbar < 10
      AND m.qc_flag = 1
      AND m.temperature_c IS NOT NULL
    GROUP BY yr
)
SELECT yr,
       mean_temp - lag(mean_temp) OVER (ORDER BY yr) AS change
FROM yearly
ORDER BY yr

Question:\tRank the regions by mean surface temperature
SQL:
WITH regional AS (
    SELECT p.region,
           avg(m.temperature_c) AS mean_temp
    FROM measurements m
    JOIN profiles p ON p.profile_id = m.profile_id
    WHERE m.pressure_dbar < 10
      AND m.qc_flag = 1
      AND m.temperature_c IS NOT NULL
    GROUP BY p.region
)
SELECT region,
       mean_temp,
       rank() OVER (ORDER BY mean_temp DESC) AS temperature_rank
FROM regional
ORDER BY temperature_rank

Question:\tFirst and last yearly mean surface temperature for each region
SQL:
WITH yearly AS (
    SELECT p.region,
           date_trunc('year', p.obs_time) AS yr,
           avg(m.temperature_c) AS mean_temp
    FROM measurements m
    JOIN profiles p ON p.profile_id = m.profile_id
    WHERE m.pressure_dbar < 10
      AND m.qc_flag = 1
      AND m.temperature_c IS NOT NULL
    GROUP BY p.region, yr
)
SELECT DISTINCT
       region,
       first_value(mean_temp) OVER (
           PARTITION BY region ORDER BY yr
       ) AS first_year_temp,
       last_value(mean_temp) OVER (
           PARTITION BY region ORDER BY yr
           ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
       ) AS last_year_temp
FROM yearly
ORDER BY region
"""

# The drifting buoys are a second platform with a different shape, and the
# model kept reaching for the Argo habits: aliasing drifter_observations as
# "do" (a keyword the validator rejects) and grouping by a region column on
# drifters that does not exist, because on this platform region belongs to the
# fix rather than to the instrument. Catalog wording did not fix either. One
# worked example did.
# The velocity columns needed their own example for a reason the temperature
# example does not show. Speed from two components is the hypotenuse, and the
# model wrote the sum: max(eastward_velocity_m_s + northward_velocity_m_s),
# which is not a speed and is not even an upper bound on one. A rule saying so
# competes with every other rule in the prompt; the worked query does not.
#
# The first version of that example grouped by region, and the model learned
# the grouping along with the arithmetic: asked for the average in one named
# region, and for the single fastest current overall, it returned a per-region
# breakdown both times. Neither question asked for one. So the ungrouped shape
# is shown too. An example teaches everything it contains, including the parts
# that were incidental to the reason it was written.
DRIFTER_EXAMPLE = """=== DRIFTING BUOY EXAMPLE ===
Question:	Average sea surface temperature from the drifting buoys in each region
SQL:
SELECT o.region,
       avg(o.sst_c) AS mean_sst
FROM drifter_observations o
WHERE o.sst_c IS NOT NULL
GROUP BY o.region
ORDER BY o.region

Question:	Average surface current speed in each region
SQL:
SELECT o.region,
       avg(sqrt(
           power(o.eastward_velocity_m_s, 2)
           + power(o.northward_velocity_m_s, 2)
       )) AS mean_current_speed_m_s
FROM drifter_observations o
WHERE o.eastward_velocity_m_s IS NOT NULL
  AND o.northward_velocity_m_s IS NOT NULL
GROUP BY o.region
ORDER BY o.region

Question:	Average surface current speed in the Bay of Bengal
SQL:
SELECT avg(sqrt(
           power(o.eastward_velocity_m_s, 2)
           + power(o.northward_velocity_m_s, 2)
       )) AS mean_current_speed_m_s
FROM drifter_observations o
WHERE o.region = 'Bay of Bengal'
  AND o.eastward_velocity_m_s IS NOT NULL
  AND o.northward_velocity_m_s IS NOT NULL
"""

# Vertical profiles need the rows one depth level at a time, with the cast they
# belong to. The chart draws one line per cast, keyed on profile_id, so a query
# that returns the levels without it can only be drawn as one line through
# every cast, which is the zigzag the chart used to produce on its own. Shown
# only to questions about the vertical structure of a measured quantity: "how
# many profiles per year" is a count, and must not be taught to return levels.
PROFILE_EXAMPLE = """=== VERTICAL PROFILE EXAMPLE ===
Question:	Show temperature profiles in the Bay of Bengal in January 2021
SQL:
SELECT p.profile_id,
       p.float_id,
       p.obs_time,
       p.latitude,
       p.longitude,
       m.pressure_dbar,
       m.temperature_c
FROM measurements m
JOIN profiles p ON p.profile_id = m.profile_id
WHERE p.region = 'Bay of Bengal'
  AND p.obs_time >= '2021-01-01'
  AND p.obs_time < '2021-02-01'
  AND m.temperature_qc = 1
  AND m.temperature_c IS NOT NULL
ORDER BY p.profile_id, m.pressure_dbar
LIMIT 5000
"""

# The LIMIT above is the validator's ceiling, not a round number. Without it
# the validator's default of 500 applies, and a high-resolution float reports
# about a thousand levels a cast: on the first real run, "salinity profiles
# near the equator in January 2026" returned half of one cast, which was drawn
# as a complete profile that stopped at mid-depth.

# "Compare BGC parameters" is the problem statement's own second example, and
# the catalog saying what BGC means was read and overridden: the model compared
# temperature and salinity. One worked query did what the sentence could not,
# as it has every other time in this file. Each parameter is averaged under its
# own QC flag, because qc_flag summarises the core parameters and says nothing
# about a biogeochemical one. Shown only when the question names BGC.
BGC_EXAMPLE = """=== BGC EXAMPLE ===
Question:	Compare BGC parameters in the Bay of Bengal by month in 2020
SQL:
SELECT date_trunc('month', p.obs_time) AS month,
       avg(m.oxygen_umol_kg) FILTER (WHERE m.oxygen_qc = 1) AS mean_oxygen_umol_kg,
       avg(m.chlorophyll_mg_m3) FILTER (WHERE m.chlorophyll_qc = 1) AS mean_chlorophyll_mg_m3,
       avg(m.nitrate_umol_kg) FILTER (WHERE m.nitrate_qc = 1) AS mean_nitrate_umol_kg,
       avg(m.ph_total) FILTER (WHERE m.ph_qc = 1) AS mean_ph_total,
       avg(m.backscatter_700) FILTER (WHERE m.backscatter_qc = 1) AS mean_backscatter_700
FROM measurements m
JOIN profiles p ON p.profile_id = m.profile_id
WHERE p.region = 'Bay of Bengal'
  AND p.obs_time >= '2020-01-01'
  AND p.obs_time < '2021-01-01'
GROUP BY month
ORDER BY month
"""

_BGC_TOPIC = re.compile(r"\b(bgc|biogeochemi\w*)\b", re.I)


def bgc_example_applies(question: str) -> bool:
    """Whether the question names BGC parameters as a group."""

    return bool(_BGC_TOPIC.search(question or ""))


_PROFILE_TOPIC = re.compile(
    r"\b(salinity|temperature|temp|oxygen|chlorophyll|nitrate|ph|bgc|density"
    r"|vertical|depth)\s+profiles?\b"
    r"|\bprofiles?\s+of\s+(salinity|temperature|oxygen|chlorophyll|nitrate)\b",
    re.I,
)


def profile_example_applies(question: str) -> bool:
    """Whether this question asks for the vertical structure of a quantity."""

    return bool(_PROFILE_TOPIC.search(question or ""))


FENCE = re.compile(
    r"```(?:sql)?(.*?)```",
    re.S | re.I,
)


def _clean(text: str) -> str:
    """Remove markdown fences and trailing semicolons."""

    match = FENCE.search(text)

    if match:
        text = match.group(1)

    return text.strip().rstrip(";").strip()


# Wide on purpose. A false positive costs a longer prompt on one question; a
# false negative costs the buoy question the example was written to fix.
# "current" is left deliberately loose here, unlike in the scope gate, because
# including an extra example for "the current year" is harmless.
_DRIFTER_TOPIC = re.compile(
    r"\bdrift|\bbuoy|\bcurrent|\bvelocit|\bsst\b|\bsea surface\b",
    re.I,
)


def drifter_example_applies(question: str) -> bool:
    """Whether this question should be shown the drifting buoy examples."""

    return bool(_DRIFTER_TOPIC.search(question or ""))


def build_prompt(
    question: str,
    context: str = "",
    include_examples: bool = True,
) -> str:
    """Assemble the full prompt, with the semantic block only when there is one."""

    # The buoy schema goes with the buoy examples: only a question that could
    # be about the buoys is shown either.
    drifters = drifter_example_applies(question)

    sections = [
        SYSTEM,
        f"=== SCHEMA ===\n{build_context(include_examples, include_drifters=drifters)}",
    ]

    # Where the archive starts and ends, so a relative period in the question
    # is anchored to the data rather than to a clock the snapshot never reached.
    coverage = data_coverage()

    if coverage:
        sections.append(coverage)

    if include_examples:
        sections.append(TRACK_EXAMPLE)
        sections.append(WINDOW_EXAMPLES)

        # Only questions that could plausibly be about the buoys pay for the
        # buoy examples. Every example is a pattern the model can match, and
        # the two here are the only ones naming a table most questions must
        # not go near: an Argo question that reaches for drifter_observations
        # is answering from the wrong platform entirely.
        if drifters:
            sections.append(DRIFTER_EXAMPLE)

        if profile_example_applies(question):
            sections.append(PROFILE_EXAMPLE)

        if bgc_example_applies(question):
            sections.append(BGC_EXAMPLE)

    if context.strip():
        sections.append(
            f"{CONTEXT_RULE}\n=== WHAT IS IN THE DATABASE ===\n{context.strip()}"
        )

    sections.append(f"=== QUESTION ===\n{question}\n\nSQL:")

    return "\n\n".join(sections)


REPAIR = """The SQL below was written for the question below and failed. Rewrite it
so it runs.

Rules:
- Output exactly one corrected SQL statement and nothing else. No prose, no
  markdown, no explanation of what went wrong.
- Every rule you were given the first time still applies.
- Fix the cause the error names. Do not rewrite the parts that were not at
  fault, and do not answer a different question because this one is awkward.
- A derived table is not visible to a sibling derived table. If one subquery
  needs another's rows, both belong in WITH clauses, or the inner one has to
  be repeated.
- If the error says a column does not exist, it does not exist. Find the one
  that holds what the question asks for, or output UNANSWERABLE if there is
  none.
"""


def build_repair_prompt(
    question: str,
    broken_sql: str,
    error: str,
    context: str = "",
    include_examples: bool = True,
) -> str:
    """Assemble the prompt for a second attempt at a query that failed.

    The schema goes back in because the commonest repairable error is a column
    that does not exist, and a model asked to fix that without the catalog in
    front of it invents a second one.
    """

    sections = [
        REPAIR,
        f"=== SCHEMA ===\n"
        f"{build_context(include_examples, include_drifters=drifter_example_applies(question))}",
    ]

    if context.strip():
        sections.append(
            f"{CONTEXT_RULE}\n=== WHAT IS IN THE DATABASE ===\n{context.strip()}"
        )

    sections.append(f"=== QUESTION ===\n{question}")
    sections.append(f"=== THE QUERY THAT FAILED ===\n{broken_sql}")
    sections.append(f"=== THE ERROR ===\n{error}")
    sections.append("SQL:")

    return "\n\n".join(sections)


def cache_version(
    context: str = "",
    include_examples: bool = True,
) -> str:
    """Prompt version, extended by a digest of everything but the question.

    One rule: if any part of the prompt other than the question changes, a
    cached answer is not reused. That covers the retrieved context, the schema
    catalog, the worked examples and the instructions themselves.

    Keying only on the hand-maintained version constant was not enough. Adding
    the biogeochemical columns to the catalog changed what the model was told
    and left every cached entry looking valid, so the cache would have served
    SQL written by a model that had never heard of those columns.

    The empty question here means the drifter examples and the buoy half of
    the catalog, which only some questions are shown, would fall out of the
    digest and stop invalidating anything when edited. They are appended
    explicitly so the one rule above still holds for every section of the
    prompt, whoever sees it.
    """
    digest = hashlib.sha256(
        (
            build_prompt("", context, include_examples)
            + build_context(include_examples, include_drifters=True)
            + (
                DRIFTER_EXAMPLE + PROFILE_EXAMPLE + BGC_EXAMPLE
                if include_examples
                else ""
            )
        ).encode()
    ).hexdigest()[:12]

    return f"{PROMPT_VERSION}:{digest}"


def _complete(
    prompt: str,
    model: str | None,
    timeout: int,
    predict: int,
    step: str = "sql",
) -> str:
    """One completion, with the retry that covers a cold model load."""

    base = os.environ.get(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )

    model = model or os.environ.get(
        "GENERATION__MODEL",
        "llama3.1:latest",
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive(),
        "options": {
            "temperature": 0,
            "num_predict": predict,
            "num_ctx": num_ctx(),
        },
    }

    with llm_span(step, model, temperature=0, max_tokens=predict) as span:
        try:
            response = httpx.post(
                f"{base}/api/generate",
                json=payload,
                timeout=timeout,
            )
        except httpx.TimeoutException:
            # The first call after a restart pays for loading the model into
            # memory, which on its own can outlast the timeout. By now that load
            # has finished, so one retry is usually enough.
            span.add_event("retry after timeout")
            response = httpx.post(
                f"{base}/api/generate",
                json=payload,
                timeout=timeout,
            )

        response.raise_for_status()

        data = response.json()
        record_ollama(span, data, prompt=prompt)

    return _clean(data["response"])


def repair_sql(
    question: str,
    broken_sql: str,
    error: str,
    model: str | None = None,
    timeout: int = 90,
    context: str = "",
    include_examples: bool = True,
) -> str:
    """One more attempt at a query that failed, given the error it failed with.

    Not cached. The cache is keyed on the question, and two runs of the same
    question can fail differently, so a repair keyed that way would serve a fix
    for an error the query did not make.

    Deliberately one attempt rather than a loop. A model that cannot use the
    error message the first time does not usually do better with the same
    message again, and the wall clock on this hardware makes each try
    expensive. Whether one is worth it is measured, not assumed.
    """

    return _complete(
        build_repair_prompt(
            question,
            broken_sql,
            error,
            context,
            include_examples,
        ),
        model,
        timeout,
        400,
        step="sql_repair",
    )


def generate_sql(
    question: str,
    model: str | None = None,
    include_examples: bool = True,
    timeout: int = 90,
    use_cache: bool = True,
    return_cache_flag: bool = False,
    context: str = "",
) -> str | tuple[str, bool]:
    """Generate one SQL statement for a question.

    ``context`` is the semantic layer's description of what the database holds,
    retrieved for this question. Empty means the model sees the schema alone.

    ``use_cache=False`` is useful for evaluation because cached responses
    would make latency measurements misleading.
    """

    # Before anything else, and before the model is paid for. A question
    # asking for a quantity the schema does not hold has one correct answer
    # and it is not SQL, so there is nothing for the model to contribute.
    # Returning the sentence the prompt asks for keeps this on the existing
    # refusal path: the validator raises, and the caller reports it exactly
    # as it reports a refusal the model made itself.
    unheld = out_of_scope(question)

    if unheld:
        declined = f"UNANSWERABLE: {unheld}"

        return (
            (declined, False)
            if return_cache_flag
            else declined
        )

    model = model or os.environ.get(
        "GENERATION__MODEL",
        "llama3.1:latest",
    )

    version = cache_version(context, include_examples)

    if use_cache:
        hit = cache.get(
            question,
            model,
            include_examples,
            version=version,
        )

        if hit is not None:
            return (
                (hit, True)
                if return_cache_flag
                else hit
            )

    sql = _complete(
        build_prompt(
            question,
            context,
            include_examples,
        ),
        model,
        timeout,
        400,
    )

    # Cache successful SQL, including UNANSWERABLE decisions.
    if use_cache and sql:
        cache.put(
            question,
            model,
            sql,
            include_examples,
            version=version,
        )

    return (
        (sql, False)
        if return_cache_flag
        else sql
    )