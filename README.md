# FloatChat

Ask the Argo float archive a question in plain English and get an answer you can check.

Two kinds of question, one system. **What does this data mean?** is answered from the Argo manuals, with the document and page behind every claim. **What does this data say?** is answered by SQL generated against the measurements database, shown to you before you are asked to believe the numbers. A router decides which, so the user asks a question rather than choosing a tool.

Both halves are built around one constraint: being confidently wrong is worse than saying nothing. The manual path cites its source and declines when the evidence is thin. The data path shows the exact query that produced the table, and that query runs as a read-only database user after passing a validator.

> **Build status.** Working end to end: NetCDF ingestion of real Argo profiles with core and biogeochemical parameters and per-parameter QC, a semantic layer that tells the SQL generator what the database actually holds, retrieval over the Argo Quality Control Manual and the Argo user's manual with citations and refusal, text-to-SQL with a four-layer safety path, a query router, conversational follow-ups, ocean charts, CSV and NetCDF export, a Streamlit dashboard, a FastAPI service, and an MCP server. Measured: [0.620 execution accuracy](#results) on 52 SQL questions across two models, and [hit@5 of 0.200](#retrieval-results) on 25 manual questions, which is the weakest number here and is discussed rather than buried. See [Limitations](#limitations) and [Roadmap](#roadmap).

---

## Problem

Argo is about 4,000 autonomous floats drifting the world ocean, diving to 2,000 decibars and surfacing every ten days with a profile of temperature and salinity. All of it is public. Almost none of it is reachable.

Reaching it means knowing the NetCDF layout, the QC flag conventions, which variable to read in which data mode, and SQL. A researcher, a student or a policymaker who can describe what they want in one sentence cannot get it.

Worse, the two things they need to know are kept apart. The numbers live in the archive; what the numbers mean lives in a 118-page quality control manual and a 99-page format manual. Asking "what is the average surface temperature in the Arabian Sea" and asking "what does a QC flag of 4 mean" are the same user, minutes apart, and answering only one of them is answering neither.

So this is one assistant over both. The failure mode is the same on both paths, a fluent and unverifiable answer, so both are built with the evidence surfaced and a refusal path.

## Architecture

```
                         question
                            │
                   ┌────────▼────────┐
                   │     router      │  rules first, model fallback
                   └────┬───────┬────┘
              documents │       │ data / chart
          ┌─────────────▼─┐   ┌─▼──────────────────────────┐
          │ embed query   │   │ retrieve data summaries    │
          │ pgvector top-k│   │   (what the db holds)      │
          │ cross-encoder │   │ generate SQL (cached)      │
          │   rerank      │   │ validate: SELECT only,     │
          │ confidence    │   │ allowed tables, real       │
          │   gate        │   │ columns, LIMIT capped      │
          │               │   │ execute as read-only user  │
          └───────┬───────┘   └─────────┬──────────────────┘
                  │ below threshold?    │
              decline                 rows ──▶ optional chart
                  │                     │
                  ▼                     ▼
        answer + citations        table + the SQL that made it
                  │                     │
                  └────────┬────────────┘
                           ▼
                  one JSON log line
```

### The semantic layer

A schema tells a model which columns exist. It cannot tell it that float 1901393 is real, worked the Arabian Sea, and stopped reporting in 2021. Without that, "what did 1901393 measure off Goa" is answered by a model guessing whether such a float exists.

So every float and every region is summarised into a sentence, embedded, and stored in its own pgvector table ([`src/semantic/`](src/semantic/)):

> ARGO float 1901393 is an APEX platform and part of project ARGO INDIA. It recorded 142 profiles from 2015-03-02 to 2021-11-18. It reported in the Arabian Sea and Bay of Bengal. Its measurements span 4.0 to 1998.0 decibars of pressure, roughly that depth in metres.

The summaries closest to a question are retrieved and handed to the SQL generator before it writes anything, under an explicit instruction never to quote a number out of them. They resolve names and ranges; the query computes every value. Both UIs show which summaries were used, so the grounding is inspectable rather than invisible.

They live in their own table rather than alongside the manual chunks, because a question about what a QC flag means must not retrieve a float summary, and a question about a float must not retrieve a page of the format manual.

The router ([`src/router/classifier.py`](src/router/classifier.py)) tries cheap regex rules first and only pays for a model call on the ambiguous cases. Its fallback route is configurable, so a router outage degrades to a known behaviour rather than an error.

### Two independent hallucination guards, on the document path

1. **Retrieval-side.** A confidence threshold over retrieval signals, evaluated before the LLM is called. Unanswerable questions are refused without spending a token.
2. **Generation-side.** The prompt requires the model to emit `INSUFFICIENT_CONTEXT` when the passages do not support an answer, checked after generation.

Two rather than one because they fail independently: the first catches "nothing relevant was retrieved," the second catches "something was retrieved but it does not answer the question."

### Four layers of protection, on the data path

1. **Prompt.** Schema, conventions and worked examples, with an explicit `UNANSWERABLE` escape hatch.
2. **Validator** ([`src/sqlgen/validator.py`](src/sqlgen/validator.py)). One statement, `SELECT` only, no forbidden keywords or identifiers, no chaining, only allowed tables, qualified columns checked against the live `information_schema`, and a `LIMIT` capped or injected.
3. **Database role** ([`db/003_roles.sql`](db/003_roles.sql)). Generated SQL runs as `gda_ro`, which holds `SELECT` and nothing else, with writes explicitly revoked.
4. **Connection.** `read_only` on the session plus a `statement_timeout`, so a pathological query cannot hold the pool.

Layer 2 alone would be a regex arms race. Layers 3 and 4 mean a validator bug is not a security incident.

Full reasoning for each choice is in [`docs/design_decisions.md`](docs/design_decisions.md).

---

## Technology stack

| Concern         | Choice                                    | Notes                             |
| --------------- | ----------------------------------------- | --------------------------------- |
| Language        | Python 3.12+                              |                                   |
| Ocean ingestion | `xarray` + `netCDF4`                      | ARGO profile files, ERDDAP CSV    |
| Structured data | PostgreSQL 16                             | core CTD plus five BGC parameters |
| Vector store    | pgvector, cosine                          | FAISS backend still selectable    |
| Embeddings      | `sentence-transformers` (BAAI/bge-small)  | swappable via config              |
| Reranking       | cross-encoder `ms-marco-MiniLM-L-6-v2`    | on by default                     |
| Chunking        | `langchain-text-splitters`                | isolated to one file              |
| LLM             | Ollama / OpenAI / Gemini                  | selected via `.env`               |
| Text-to-SQL     | `sqlparse` validation + read-only role    |                                   |
| Charts          | Plotly for ocean plots, matplotlib else   | PNG produced in every case        |
| API             | FastAPI + Pydantic                        | same service object as Streamlit  |
| Agent interface | MCP server on stdio                       | four tools, none of them raw SQL  |
| UI              | Streamlit (8 pages)                       |                                   |
| Evaluation      | retrieval metrics + SQL execution metrics |                                   |
| Logging         | structured JSONL                          |                                   |
| Config          | `config.yaml` + environment overrides     |                                   |

## Project structure

```
RAG_Assistant/
├── app.py                     # Streamlit dashboard (presentation only)
├── config.yaml                # all tunable behaviour
├── docker-compose.yml         # pgvector Postgres + the API
├── src/
│   ├── ingestion/   loader.py · chunker.py · argo_netcdf.py · regions.py
│   ├── embeddings/  embedding_model.py
│   ├── vectorstore/ pgvector_store.py · vectordb.py      # pgvector or FAISS
│   ├── retrieval/   retriever.py · reranker.py · confidence.py
│   ├── generation/  prompt.py · llm.py · answer_generator.py
│   ├── router/      classifier.py                        # documents | data | chart
│   ├── semantic/    summaries.py · index.py              # what the db holds
│   ├── sqlgen/      generator.py · validator.py · executor.py · schema_context.py
│   ├── charts/      builder.py
│   ├── analytics/   trend_test.py · clustering.py · forecast.py
│   ├── evaluation/  metrics.py · evaluator.py · sql_metrics.py · ab_test.py
│   ├── monitoring/  logger.py · analytics.py
│   ├── api/         main.py · schemas.py                 # FastAPI
│   └── utils/       config.py · schemas.py · db.py · cache.py · pipeline.py
├── db/              001_schema.sql · 002_pgvector.sql · 003_roles.sql
│                    004_argo_qc.sql · 005_semantic_layer.sql
│                    006_bgc_parameters.sql
│                    schema_catalog.md · queries/         # 12 reference queries
├── scripts/         fetch_argo_index.py · load_argo_netcdf.py
│                    load_argo.py · build_semantic_index.py
│                    make_sample_data.py · run_ab_test.py
│                    mcp/            MCP server (four tools, no raw SQL)
├── data/            raw/ processed/ evaluation/ cache/
└── tests/
```

Everything routes through `RAGService` in [`src/utils/pipeline.py`](src/utils/pipeline.py). Streamlit and FastAPI both call that one object, so the web page and the API cannot drift apart.

---

## Installation

```bash
git clone https://github.com/poojithareddy19/RAG-Knowledge-Assistant.git
cd RAG-Knowledge-Assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`sentence-transformers` pulls in torch, and on Windows the default PyPI wheel bundles CUDA at roughly 2.4 GB. The configured embedding model runs on CPU, so install the CPU build first and skip the download:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Start PostgreSQL. The image is built from [`db/Dockerfile`](db/Dockerfile) rather than pulled, because this project needs two extensions and no published image carries both: pgvector for the document and summary embeddings, PostGIS for the basin polygons and profile positions. The schema, extensions and read-only role in `db/` are applied automatically on first boot.

```bash
docker compose up -d --build db
```

Those files run only when the volume is empty. On a database created before a migration existed, apply it by hand:

```bash
docker compose exec -T db psql -U gda -d gda < db/004_argo_qc.sql
docker compose exec -T db psql -U gda -d gda < db/007_postgis.sql
```

`007_postgis.sql` adds the `geom` column, backfills it from the existing latitude and longitude, indexes it and creates the `regions` table. It is safe to re-run. The polygons that fill that table are not committed and not downloaded: see [Region polygons](#region-polygons).

The default provider is Ollama, which needs no API key:

```bash
ollama pull llama3.1
```

### Load ocean data

Real profiles come from the Argo GDAC. [`scripts/fetch_argo_index.py`](scripts/fetch_argo_index.py) reads the global profile index, keeps the rows that are both tagged Indian Ocean and inside the box, and downloads them:

```bash
python scripts/fetch_argo_index.py --floats 40 --limit 800 --download
python scripts/fetch_argo_index.py --bgc --floats 10 --limit 200 --download
```

Both limits matter. `--floats` decides how many floats the sample covers and `--limit` how many of their cycles are taken. A file limit on its own buys one cycle each from hundreds of floats, which is a map with no tracks on it and a profile count of 1 for every float. Floats are visited region by region rather than in id order, because WMO ids are handed out in blocks per deployment programme, so id order is geography order: the first forty ids inside the box are forty floats south of the equator and not one in the Arabian Sea. `--include 1900083,1900162` takes named floats in full instead of sampling them, which is how the set gets a float with a unique profile count. The 58 MB index is cached under `data/raw/argo_index/` after the first run; `--refresh` re-downloads it.

The second command is not optional. A core-only load leaves every biogeochemical column NULL, and a NULL column reads as "no oxygen here" rather than as "this float does not carry that sensor".

ARGO is distributed as NetCDF, which is the loader to prefer. It takes files or directories and searches directories recursively:

```bash
python scripts/load_argo_netcdf.py data/raw/argo/
```

Re-running is safe: profiles already present are left alone rather than duplicated, so an interrupted load can simply be repeated.

Three ARGO conventions are handled in [`src/ingestion/argo_netcdf.py`](src/ingestion/argo_netcdf.py) rather than left to the caller. Profiles in delayed or adjusted mode are read from their `*_ADJUSTED` variables, because in those modes the raw variables are kept only for provenance and reading them anyway is the most common way to get ARGO wrong. Each parameter's QC flag is stored separately, because a level can have good temperature and bad salinity. And a delayed-mode profile whose `*_ADJUSTED` variables are declared but entirely fill falls back to the raw values, because taking the adjusted pair on the strength of its existence alone yields no depth and no value at every level, and the whole profile is then dropped without a word. One float in the sample lost all 54 of its profiles that way.

There is also an ERDDAP tabledap CSV loader, kept because a CSV export is often the quickest way to get a region loaded:

```bash
python scripts/load_argo.py path/to/argo_export.csv
```

Or generate synthetic ARGO-shaped data with a planted 0.02 °C/year warming signal, which is useful for checking that the trend tests detect what they should:

```bash
python scripts/make_sample_data.py
```

### Load the manuals

The document side answers questions about what the data means, so its corpus is the two documents that define that: the [Argo Quality Control Manual for CTD and Trajectory Data](https://doi.org/10.13155/33951) and the [Argo user's manual](https://doi.org/10.13155/29825). Both are public, and neither is committed here.

```bash
mkdir -p data/raw/manuals
curl -o data/raw/manuals/argo_quality_control_manual.pdf https://archimer.ifremer.fr/doc/00228/33951/32470.pdf
curl -o data/raw/manuals/argo_user_manual.pdf https://archimer.ifremer.fr/doc/00187/29825/120885.pdf
```

Then ingest them from the Upload Documents page, or:

```python
from src.utils.pipeline import RAGService

service = RAGService()
service.ingest_file("data/raw/manuals/argo_quality_control_manual.pdf")
service.ingest_file("data/raw/manuals/argo_user_manual.pdf")
```

That is 118 and 99 pages, 309 and 371 chunks. The retrieval gold set in `data/evaluation/questions.csv` is written against these two documents at these versions, so a newer release of either manual will shift page numbers and invalidate the anchors.

### Region polygons

Region used to be three latitude and longitude inequalities, which put the Gulf of Aden in the Arabian Sea and the whole Mozambique Channel in the Southern Indian Ocean. It is now a containment test against real basin polygons.

Those polygons are a third party dataset with its own licence, so this repository neither ships them nor downloads them on your behalf. Fetch the IHO Sea Areas layer from [marineregions.org](https://www.marineregions.org/downloads.php), export it as GeoJSON in EPSG:4326, save it as `data/raw/regions.geojson`, then:

```bash
python scripts/load_regions.py
```

The script fails with an explicit message when the file is absent rather than reaching for the network.

[`region_for`](src/ingestion/regions.py) keeps the inequalities as its fallback, because a float in open ocean sits inside no published basin and ingestion must not fail or write a NULL region over it. Each fallback is counted, and the loaders print that count when they finish, so a return to the coarse rule is visible rather than silent.

### Build the semantic layer

The summaries describe what is in the tables, so they are stale the moment new profiles land. Rebuild them after any load:

```bash
python scripts/build_semantic_index.py
```

Skipping this is not fatal. Retrieval returns nothing, and the SQL generator falls back to working from the schema alone, exactly as it did before the layer existed.

## Configuration

- **`config.yaml`** holds behaviour: chunk size, embedding model, top-k, confidence weights and threshold, router and SQL limits, cache TTL.
- **`.env`** holds secrets, provider selection and database URLs. Any `SECTION__KEY` variable overrides the matching `config.yaml` value, for example `RETRIEVAL__TOP_K=8`.

Two database URLs are required. `DATABASE__URL` owns the schema and runs ingestion; `DATABASE__READONLY_URL` is what generated SQL executes as. Keeping them separate is the point.

## Running

```bash
streamlit run app.py
```

Pages: Home, Upload Documents, Knowledge Base, Ask Questions, Ocean Data, Evaluation, Monitoring, Settings.

Or the HTTP service:

```bash
uvicorn src.api.main:app --reload
```

`POST /ask` takes `{"question": "...", "route_override": null, "session_id": null}` and returns the answer with citations or with the generated SQL, rows and timings. Passing the same `session_id` on a later request is what makes a follow up like "and in 2022?" resolvable. `GET /health` reports indexed documents, measurement count and whether the LLM is reachable. Interactive docs at `/docs`.

`POST /export` takes the same body plus `"format": "csv" | "netcdf"` and streams the result set as a file. It returns every row rather than the hundred `/ask` includes for display, and the NetCDF carries units, column descriptions and the generating SQL in its attributes, so a download stays readable and reproducible after it leaves the API.

Both UIs show the same two things the answer depends on: for documents, the retrieved chunks with similarity scores and the exact prompt sent to the model; for data, the SQL. If an answer looks wrong, you can see immediately whether retrieval failed, generation failed, or the query was simply right and surprising.

## Testing

```bash
pytest -q
```

Covers the deterministic core: chunking, metrics, confidence, vector-store contract, router rules, SQL cache and the SQL validator. No API key, database or GPU required. The validator suite runs every query in `db/queries/` through the validator, so a reference query the validator would reject fails the build.

---

## Confidence and refusal

Confidence combines three interpretable retrieval signals, normalised to 0-1:

- **mean similarity** across the top-k retrieved chunks
- **support**, the fraction of top-k chunks above a similarity floor
- **spread**, derived from score variance

The combination is **multiplicative, not additive** ([`src/retrieval/confidence.py`](src/retrieval/confidence.py)). Mean similarity is the primary signal; support and spread are folded into a 0.5-1.0 multiplier that can penalise a strong primary signal but cannot manufacture one. Below a hard floor of `MIN_MEAN_SIM = 0.28` the score is zero regardless of the other two.

That shape was chosen after an additive version let a question with very low mean similarity clear the threshold on support and spread alone, which is the failure mode that matters: a question the corpus cannot answer getting waved through.

Below `confidence.answer_threshold` the system returns the closest passages and an explanation instead of an answer.

Retrieval signals rather than asking the LLM to self-report, because a model cannot know that the corpus lacks an answer, only that its context window does. Retrieval statistics measure corpus-to-question fit directly and cost nothing extra.

## Model Context Protocol

The assistant is also an MCP server ([`src/mcp/server.py`](src/mcp/server.py)), so an MCP client can query the ocean data as tools rather than through the web page or the API.

```bash
python -m src.mcp.server
```

It speaks stdio, so a client launches it rather than connecting to a port:

```json
{
  "mcpServers": {
    "argo": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/path/to/RAG_Assistant"
    }
  }
}
```

Four tools:

| Tool | Arguments | Returns |
| --- | --- | --- |
| `query_argo` | `question` | answer, generated SQL, columns, rows |
| `list_floats` | `region`, `year` (both optional) | float ids with profile counts and date ranges |
| `get_profile` | `float_id`, `cycle` (optional) | one float's depth levels |
| `describe_schema` | none | the column catalog as text |

**There is deliberately no tool that accepts SQL.** The four safety layers exist so that generated SQL is constrained before it reaches the database, and a `run_sql` tool would hand an external client a way around all of them. A test asserts that no registered tool takes a statement, under any argument name, and that every tool's schema closes `additionalProperties` so one cannot arrive under a different one.

`list_floats` and `get_profile` do not go through the model at all. Their shape is fixed, so they are parameterised queries written by hand in the server module, capped and executed as the read-only role.

## Evaluation

Two harnesses, because the two paths fail differently.

**Retrieval** ([`src/evaluation/evaluator.py`](src/evaluation/evaluator.py)). Recall@K, Precision@K, Hit@K, MRR, nDCG@K against `data/evaluation/questions.csv`. Relevance is anchored at **document and page** level rather than chunk ID, because chunk IDs renumber whenever chunk size changes, which would invalidate the entire gold set every time a chunking parameter is tuned.

#### Retrieval results

25 questions written against the two manuals in the corpus, each anchored to the page that answers it. Every anchor was checked to exist in the index, so a miss is retrieval failing rather than the gold set pointing at nothing.

| Metric | k=1 | k=3 | k=5 |
| --- | --- | --- | --- |
| **Hit@k** | 0.080 | 0.200 | **0.200** |
| Recall@k | 0.080 | 0.200 | 0.240 |
| Precision@k | 0.080 | 0.067 | 0.048 |
| nDCG@k | 0.080 | 0.145 | 0.163 |
| MRR | | | 0.127 |

**This is bad, and it is the most useful number in this README.** Five questions of twenty-five find their page in the top five.

The failure has a clear shape. The right *document* is in the top five for 18 of 25 questions, so the system knows which manual answers a question about the gradient test. It lands on the wrong *page* of it: the nearest retrieved page is off by 2, 3, 6, 8, 9, 10, 12, 19, 20, 28, 44 and 47 pages on the questions it misses. It finds the right book and the wrong chapter.

The most likely cause is visible in the chunks. Every page of both manuals begins with the same running header, `NN Argo Data Management Quality Control Manual for CTD and Trajectory Data Version 3.9`, so a large and identical block of text sits at the top of hundreds of chunks and flattens the distance between them. Stripping running headers during chunking is the obvious first fix and has not been done.

An unrelated 392-page machine learning textbook was in the corpus from an earlier version of this project and was taking 25% of every top-five. Removing it changed the score by nothing at all: the same five questions hit and the same twenty miss, because the right page was not in the candidate set either way. That is worth knowing before blaming a distractor for a retrieval problem.

Reproduce it from the Evaluation page of the dashboard, or:

```python
from src.evaluation.evaluator import evaluate_retrieval
from src.utils.pipeline import RAGService

report = evaluate_retrieval(RAGService().retriever, "data/evaluation/questions.csv", k=5)
```

The number is published rather than tuned. The previous version of this file admitted it had no retrieval numbers at all; having a bad one that says where the problem is beats having none.

**Text-to-SQL** ([`src/evaluation/sql_metrics.py`](src/evaluation/sql_metrics.py)). The metric is **execution accuracy**: run the generated query and a hand-written reference query and compare their result sets. Whether a query parses says nothing about whether it answered the question.

Comparing results rather than SQL text is deliberate, because there are many correct ways to write the same query. A generated query that reaches the same rows through a join the reference did not need is still a right answer, and string comparison would mark it wrong.

```bash
python -m src.evaluation.sql_metrics --runs 3
```

`--runs N` scores the whole set N times and reports each metric as a mean with a standard deviation, because one run cannot separate a real change from sampling noise. The SQL cache is forced off whenever N is above 1: a cached repeat replays the first run's SQL, so the spread would describe the cache rather than the model.

```python
from src.evaluation.sql_metrics import evaluate_runs, summarise_buckets

summaries, buckets, detail = evaluate_runs(runs=3)
```

### Results

52 questions against **real Argo GDAC profiles**: 80 floats, 1,099 profiles and 175,364 measurements from the Indian Ocean, spanning 2001-07-24 to 2025-10-05 across the Arabian Sea, the Bay of Bengal and the Southern Indian Ocean. 49,296 measurements carry a dissolved oxygen reading.

Two models, both local through Ollama, on the same questions and the same prompt. `llama3.1:8b` is the default the system ships with; `qwen2.5-coder:7b` is a code specialist, run to find out whether the two ceilings this benchmark keeps hitting are the model's or the prompt's.

| Metric | `llama3.1:8b` | `qwen2.5-coder:7b` |
| ------------------------- | ----- | ----- |
| **Execution accuracy**    | **0.620 +/- 0.015** | **0.522** |
| Validation pass rate      | 1.000 | 1.000 |
| Execution rate            | 0.957 | 0.891 |
| **Correct refusal rate**  | 0.833 (5/6) | **1.000 (6/6)** |
| False refusal rate        | 0.000 | 0.000 |
| Median latency            | 37.7 s | 46.5 s |

| Bucket | Questions | `llama3.1:8b` | `qwen2.5-coder:7b` |
| ------------ | --- | ----- | ----- |
| filter       | 8   | 1.000 | 0.625 |
| easy         | 8   | 0.938 | 0.625 |
| groupby      | 6   | 0.500 | **0.833** |
| join         | 8   | 0.500 | 0.500 |
| qc           | 5   | 0.400 | 0.400 |
| bgc          | 5   | 0.400 | 0.200 |
| window       | 6   | 0.333 | 0.333 |
| unanswerable | 6   | 5/6 refused | **6/6 refused** |

llama3.1 is the mean of two runs, qwen a single run. Quantisation differs and is worth stating: llama3.1 is Q4_K_M, qwen is `7b-instruct-q3_K_M`, one step lower, because the larger file would not fit on the disk it was run from. Some of qwen's weakness on the easy buckets plausibly belongs to that rather than to the model.

Neither model is simply better. qwen is worse overall, much worse on the simple buckets, clearly better on grouped aggregates, and perfect at refusing. Picking one is a trade, not an upgrade.

#### Did the worked examples move the window bucket?

No.

Three worked examples were added to the prompt, one each for `lag`, `rank` and `first_value`, written against these exact tables and column names. The bucket did not move:

| | Execution accuracy, window bucket |
| --- | --- |
| llama3.1, before the examples | 0.333 (2/6) |
| llama3.1, after the examples  | 0.333 (2/6), in both runs |
| qwen2.5-coder, with the examples | 0.333 (2/6) |

The same two questions pass and the same four fail, and "year over year change in mean surface temperature" still fails for llama3.1 despite a near-identical `lag` example sitting in its prompt. qwen gets that one right and loses the running total instead, so the two models fail on different questions and arrive at the same score.

Overall accuracy for llama3.1 went **down** slightly after the examples, 0.652 to 0.620, driven by `join` and `qc` rather than by anything the examples touched. A longer prompt is not free. The honest reading is that few-shot examples did not address this ceiling, and the extra prompt length cost a little elsewhere. They are kept because they are correct and because removing them would leave the comparison unmeasured, not because they helped.

#### The seafloor question, on a second model

This is the one result where the second model settles something.

Asked for the seafloor depth beneath each float, `llama3.1:8b` answers `MAX(m.pressure_dbar) AS seafloor_depth`: the deepest the float descended, renamed to the thing that was asked for. A float profiles to around 2000 decibars over a seabed often three times deeper, so the query is not merely wrong, it looks entirely reasonable. It has done this in every run recorded here, before and after the catalog was reworded to say plainly that `pressure_dbar` is not the depth of the seabed.

`qwen2.5-coder:7b` refuses it, and refuses the other five unanswerable questions too, at a false refusal rate of 0.000. Its answer is exactly `UNANSWERABLE`.

So the earlier conclusion holds and now has evidence behind it: **that fabrication was a model limitation, not a wording problem.** Three prompt fixes were tried against it and each traded the caught fabrication for several refusals of legitimate questions, which is a worse system. A different model fixed it for free. What that costs is 0.098 of execution accuracy elsewhere, which is the shape of the choice rather than a reason to dismiss it.

Validation pass rate is 1.000 for both models while accuracy is 0.620 and 0.522. Every generated query was well formed, safe and executable, and a third to a half still answered the wrong question. That gap is the entire argument for measuring results rather than liveness.

#### On repeated runs

The three-run evaluation on the previous prompt produced byte-identical SQL for all 52 questions and an identical score in all three runs, which is what temperature 0 should do. It is not an absolute guarantee: on the current prompt, two full runs agreed on 51 of 52 verdicts, with one `easy` question flipping. So the spread quoted above is real but small, and a difference of a point or two between configurations is not evidence of anything.

Reproducing either column:

```bash
python -m src.evaluation.sql_metrics --runs 2
python -m src.evaluation.sql_metrics --model qwen2.5-coder:7b-instruct-q3_K_M
```

The `model` is recorded on every row of the output, so two models' results can be concatenated and grouped without a run header to keep track of.

#### What changed when the data became real

The same gold set scored 0.739 against 40 synthetic floats. The two figures are not comparable, because the data and six of the questions both changed, but the per-bucket movement says where the difference came from. Both columns below are `llama3.1:8b` before the window examples were added, so this is a comparison of data rather than of prompts.

| Bucket | Synthetic | Real | |
| ------- | ----- | ----- | --- |
| bgc     | 1.000 | 0.400 | the old score was measured on columns that were entirely NULL |
| groupby | 0.500 | 0.333 | |
| qc      | 0.800 | 0.600 | |
| window  | 0.167 | 0.333 | |
| join    | 0.625 | 0.625 | |

Most of the headline drop is the `bgc` bucket, and it is a correction rather than a regression. With every biogeochemical column NULL, the reference query and the generated query agreed by both finding nothing, and five questions scored perfect without testing any arithmetic. Against floats that actually carry oxygen, chlorophyll and pH sensors, two of five are right.

Six questions had to be repaired before the set could run at all, because they were written against data that no longer exists: two asked about years with no profiles in the real sample, one asked about synthetic float 2900007, one counted nitrate readings that no sampled float carries, and the two "which float has the most profiles" questions were restored now that real floats have genuinely different profile counts rather than 730 each.

### What these numbers do not tell you

- **One model, one prompt, one schema.** The set has never been run against a second model, so nothing here separates what this model cannot do from what no model could do with this prompt.
- **Execution accuracy has false positives.** A wrong query can collapse to the right number when the data happens to cooperate, and the comparison cannot tell that apart from understanding.
- **The BGC sample is narrow.** Ten biogeochemical floats were loaded and none of them carries a NITRATE sensor, so that column is empty and the `bgc` bucket tests oxygen, chlorophyll, pH and backscatter only.
- **Profile counts are shaped by the download, not by the ocean.** The sample takes a bounded number of cycles per float, so "which float reported most" is answerable but is a fact about what was fetched rather than about the fleet.
- **The document retrieval gold set is still a template.** `questions.csv` is five rows referencing documents not committed here. Only the SQL side has real numbers.

## Beyond retrieval

Three analyses that run against the measurements table directly, included because "the model wrote some SQL" is not by itself evidence that the data supports a claim:

- **Trend tests** ([`src/analytics/trend_test.py`](src/analytics/trend_test.py)): regression and Mann-Kendall on regional surface series, reporting significance rather than a slope with no error bar.
- **Clustering** ([`src/analytics/clustering.py`](src/analytics/clustering.py)): k-means over profile features into data-driven water-mass groups, with silhouette scoring.
- **Quantile forecasting** ([`src/analytics/forecast.py`](src/analytics/forecast.py)): LightGBM quantile regression on monthly regional means, so the output is an interval.

There is also a paired A/B harness ([`src/evaluation/ab_test.py`](src/evaluation/ab_test.py)) with a sample-size calculation, for comparing two pipeline configurations without reading noise as improvement.

---

## Design decisions

| Decision                                 | Rationale                                                                                                     |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Router with rules before the model        | Most questions are unambiguous. Paying for a model call on those is latency and cost for nothing               |
| Read-only role for generated SQL          | A validator bug becomes a failed query rather than a data-loss incident                                        |
| Column check against `information_schema` | Read from the live database, not the markdown catalog, so it cannot drift away from the real tables            |
| Summaries in their own table              | A question about a float must not retrieve a page of the manuals, and the reverse                              |
| Context is named, never quoted            | The prompt forbids copying a number out of a summary, so every value in an answer comes from the query         |
| SQL cache keyed on prompt version         | Re-running a demo should be fast, but a changed prompt must invalidate everything cached under the old one. The retrieved context is part of that key |
| Chunking isolated per page                | A chunk straddling two pages makes its citation ambiguous. Per-page chunking keeps "page 42" honest            |
| Multiplicative confidence                 | Secondary signals should modulate the primary one, never rescue it                                             |
| Retrieval-based confidence                | LLMs are structurally uncalibrated about corpus coverage; retrieval signals measure fit directly and are free  |
| Reranker preserves cosine scores          | The cross-encoder decides ordering only, so confidence still reads a comparable scale                          |
| Temperature 0                             | The task is faithful extraction, not creative writing. Also makes evaluation runs comparable                   |
| Doc/page evaluation anchors               | Chunking parameters can be tuned without destroying the ground truth set                                       |
| One service facade                        | Streamlit and FastAPI share `RAGService`, so behaviour is defined once                                         |

## Why RAG rather than fine-tuning

RAG keeps knowledge external and updatable: re-index instead of re-train. It makes answers attributable, so a reader can check the source. And it avoids baking sensitive documents into model weights. Fine-tuning changes style and format, not the facts a system can cite.

For the ocean data the argument is stronger still. The answer to "average surface temperature in the Arabian Sea in 2021" is a number in a table that changes as floats report. No amount of training bakes that in correctly; a query reads it.

## Limitations

**Data.** The database holds real Argo profiles: 80 floats, 1,099 profiles and 175,364 measurements from the Indian Ocean. Ten of those floats carry biogeochemical sensors, which populates oxygen, chlorophyll, pH and backscatter, but none carries a NITRATE sensor, so that one column is still empty. The sample takes a bounded number of cycles per float rather than every cycle, so per-float profile counts reflect the download rather than the fleet. Real QC flags arrive only through the NetCDF loader; the CSV loader still writes `qc_flag = 1` for every row, so a database built from CSV has a QC column that means nothing. Region is assigned by three latitude/longitude inequalities in [`src/ingestion/regions.py`](src/ingestion/regions.py), which is coarse for a field that nearly every query groups by.

**Conversation.** A follow up is rewritten into a standalone question before routing, which keeps the router, the SQL generator and the cache stateless and keeps the cache keyed on what was actually asked. The history behind that rewrite is not persisted: the API holds it in a process-local dict, so it does not survive a restart and does not work across workers, and the Streamlit page holds it in session state, so it disappears with the browser session. Moving it to Redis or a sessions table is the fix and has not been done. A follow up whose history is lost degrades to being answered as a standalone question rather than to an error, and a rewrite that goes wrong is visible: the page shows what the question was answered as, and both versions are written to the log.

**Architecture.** Semantic retrieval now feeds SQL generation, but a single question still gets a single answer: the router picks documents or data, so a question genuinely needing both sources gets one. Summaries are rebuilt only when the script is run, so they drift from the tables between loads. Summarising is per float and per region; a database with thousands of floats will want coarser grouping than one summary each.

**Retrieval and generation.** Retrieval over the manuals is the weakest part of this system: hit@5 of 0.200 on 25 questions, finding the right manual for 18 of 25 but the right page for 5. The running header repeated on every page of both manuals is the prime suspect and stripping it is untried. No models are trained here. Extraction quality depends on the source PDF; scanned PDFs need OCR, which is not wired in. Confidence is a heuristic over retrieval signals, not a calibrated probability, so the threshold needs tuning against your own gold set. English-only embeddings. No individual document deletion.

**Charts.** Trajectories, depth profiles, depth-time sections and T-S diagrams are drawn as interactive Plotly figures, dispatched on column names because latitude and longitude are two ordinary floats to a dtype check. Everything else falls through to the matplotlib line and bar builder, and a PNG is produced in every case so an image client keeps working. The dispatch is first-match, so a result carrying positions is always drawn as a track even when it also carries measurements.

**Evaluation.** Text-to-SQL scores 0.620 execution accuracy on the shipped model and 0.522 on a second one, and complex queries are much worse than either average suggests: the window bucket is 1 in 3 on both, and three worked examples aimed squarely at it moved nothing. The shipped model still fabricates a seafloor depth rather than refusing; the second model refuses it and pays for that elsewhere. Both columns come from one gold set of 52 questions written by the same person who wrote the schema, which is a real limit on what they can show. The document retrieval side has no published numbers at all.

## Roadmap

Done:

- [x] NetCDF ingestion with `xarray`, carrying real QC flags through
- [x] BGC-ARGO parameters in the schema, parser and catalog
- [x] Real Argo GDAC profiles, replacing the synthetic set: 80 floats, 1,099 profiles, 175,364 measurements
- [x] A real BGC float in the data, so the bgc questions test values rather than a shared NULL
- [x] Float and region metadata summarised into pgvector, so semantic retrieval informs SQL generation
- [x] Repeat runs, so a difference can be told from noise
- [x] Trajectory maps, depth profiles, depth-time sections and T-S diagrams
- [x] Conversational follow-ups, by rewriting a follow up into a standalone question
- [x] CSV and NetCDF export, with units and the generating query attached
- [x] PostGIS geometry on profiles, with a spatial region lookup and `ST_DWithin` distance queries
- [x] An MCP server, with no tool that accepts SQL
- [x] Few-shot window-function examples, measured: they did not move the bucket
- [x] A second model on the same 52 questions, which closed the refusal gap
- [x] A document retrieval gold set that is not a five-row template, with numbers published

Not done, honestly:

- [ ] Retrieval is weak: hit@5 of 0.200. Strip the running headers from chunks and measure again
- [ ] Load the IHO basin polygons, so `region` comes from geometry rather than the fallback inequalities
- [ ] Answer a single question from documents and data together, which is still the largest architectural gap
- [ ] Real QC flags on the CSV loader too
- [ ] Persist conversation history, which currently dies with the process
- [ ] A nitrate-carrying BGC float, since none of the ten sampled floats has that sensor
- [ ] LLM-judge / Ragas generation metrics (faithfulness, groundedness, answer relevance)
- [ ] Hybrid search (BM25 + dense), which would likely help the exact-phrase manual questions most
- [ ] GitHub Actions CI
- [ ] Multilingual query support
- [ ] OCR path for scanned PDFs
