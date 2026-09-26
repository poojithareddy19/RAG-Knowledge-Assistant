# FloatChat

Ask the Argo float archive a question in your own language and get an answer you can check.

Two kinds of question, one system. **What does this data mean?** is answered from the Argo manuals, with the document and page behind every claim. **What does this data say?** is answered by SQL generated against the measurements database, shown to you before you are asked to believe the numbers. A router decides which, and when a question needs both it answers from both, so the user asks a question rather than choosing a tool.

Both halves are built around one constraint: being confidently wrong is worse than saying nothing. The manual path cites its source and declines when the evidence is thin. The data path shows the exact query that produced the table, and that query runs as a read-only database user after passing a validator.

> **Build status.** Working end to end: NetCDF ingestion of real Argo profiles with core and biogeochemical parameters and per-parameter QC, a semantic layer that tells the SQL generator what the database actually holds, retrieval over the Argo Quality Control Manual and the Argo user's manual with citations and refusal, text-to-SQL with a four-layer safety path, a query router that can answer from the manuals and the database at once, conversational follow-ups, multilingual questions answered in the language they were asked in, a second in-situ platform in 187 drifting buoys, ocean charts including overlaid profile comparisons, CSV, Parquet and NetCDF export, a Streamlit dashboard, a FastAPI service, and an MCP server the chat itself calls. Measured: [0.567 execution accuracy](#results) on 67 SQL questions across two models, [7/7 correct refusals](#closing-it-without-the-model), and [hit@5 of 0.960](#retrieval-results) on 25 manual questions. See [Limitations](#limitations) and [Roadmap](#roadmap).

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

The router ([`src/router/classifier.py`](src/router/classifier.py)) tries cheap regex rules first and only pays for a model call on the ambiguous cases. Its fallback route is configurable, so a router outage degrades to a known behaviour rather than an error. A fourth route, `both`, is described below.

### Answering from both at once

"What does the QC manual say about a flag of 4, and how many of my profiles have one" is two questions wearing one coat, and for most of this project's life it got one answer. The `both` route answers it from both sources.

The compound question is **split before either backend sees it** ([`src/router/planner.py`](src/router/planner.py)). This is not decoration. Sent whole to the retriever, the counting clause is noise in the embedding; sent whole to the SQL generator, the manual clause invites a join against a table that does not exist. Each backend answers its own half, and the halves are rejoined by a synthesis step ([`src/generation/synthesis.py`](src/generation/synthesis.py)).

That synthesis step is the only place in the project where a model is shown two sources at once, which makes it the only place that can invent a relationship between them. It is written against that. It is given the document answer and the rows and told to join them with a sentence, not to reason about them: it may not round, total, average or compare, because every number in the result is already correct and any arithmetic it does is arithmetic nobody checked.

**Nothing is lost when the extra machinery fails.** A split that cannot be made sends the whole question to each backend, which is what the system did before. A synthesis that cannot be written falls back to both answers under their own headings, which is a worse answer and a true one. A half that refuses is not an error, it is one source having nothing to say, and the answer is written from the other; only the case where neither half answers is a refusal. Both raw results travel back under `parts` and the page shows them side by side, because the one answer in the system written from two sources is the one a reader most needs to be able to take apart.

Combined confidence is the **lower** of the two halves that answered. An answer resting on a confident manual passage and an empty query is not a confident answer.

The cost is honest: a combined question is four model calls where a data question is one. The rule layer keeps it rare, routing to `both` only when a question names a written source *and* asks for a quantity, so "what does the manual say about temperature limits" stays a document question.

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
| Ocean ingestion | `xarray` + `netCDF4`                      | Argo NetCDF, drifters over ERDDAP |
| Structured data | PostgreSQL 16                             | core CTD plus five BGC parameters |
| Vector store    | pgvector, cosine                          | FAISS backend still selectable    |
| Embeddings      | `sentence-transformers` (BAAI/bge-small)  | swappable via config              |
| Reranking       | cross-encoder `ms-marco-MiniLM-L-6-v2`    | on by default                     |
| Chunking        | `langchain-text-splitters`                | isolated to one file              |
| LLM             | Ollama, local, no API key                 | `llama3.1:8b`, `qwen2.5-coder:7b` |
| Text-to-SQL     | `sqlparse` validation + read-only role    |                                   |
| Charts          | Plotly for ocean plots, matplotlib else   | PNG produced in every case        |
| API             | FastAPI + Pydantic                        | same service object as Streamlit  |
| Agent interface | MCP server on stdio, and the chat is a client | five tools, none of them raw SQL |
| UI              | Streamlit (6 pages)                       |                                   |
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
│   ├── router/      classifier.py                        # documents | data | chart | both
│   │               planner.py                           # splits a compound question in two
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
│                    build_semantic_index.py · load_regions.py
│                    run_ab_test.py
│                    mcp/            MCP server and the chat's client (five tools, no raw SQL)
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

A database created before the vector index was changed from IVFFlat to HNSW still has the old one, and the old one silently loses most of its recall. Replace it:

```bash
docker compose exec -T db psql -U gda -d gda -c   "DROP INDEX IF EXISTS idx_chunks_embedding;
   CREATE INDEX idx_chunks_embedding ON doc_chunks USING hnsw (embedding vector_cosine_ops);"
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

### Load the manuals

The document side answers questions about what the data means, so its corpus is the two documents that define that: the [Argo Quality Control Manual for CTD and Trajectory Data](https://doi.org/10.13155/33951) and the [Argo user's manual](https://doi.org/10.13155/29825). Both are public, and neither is committed here.

```bash
mkdir -p data/raw/manuals
curl -o data/raw/manuals/argo_quality_control_manual.pdf https://archimer.ifremer.fr/doc/00228/33951/32470.pdf
curl -o data/raw/manuals/argo_user_manual.pdf https://archimer.ifremer.fr/doc/00187/29825/120885.pdf
```

Then ingest them:

```python
from src.utils.pipeline import RAGService

service = RAGService()
service.ingest_file("data/raw/manuals/argo_quality_control_manual.pdf")
service.ingest_file("data/raw/manuals/argo_user_manual.pdf")
```

That is 118 and 99 pages, 309 and 371 chunks. The retrieval gold set in `data/evaluation/questions.csv` is written against these two documents at these versions, so a newer release of either manual will shift page numbers and invalidate the anchors.

### Region polygons

Region used to be three latitude and longitude inequalities, which put the Gulf of Aden in the Arabian Sea and the whole Mozambique Channel in the Southern Indian Ocean. It is now a containment test against real basin polygons.

Those polygons are a third party dataset with its own licence, so this repository neither ships them nor downloads them on your behalf. The IHO Sea Areas layer is served by [Marine Regions](https://www.marineregions.org/) over a public WFS, so fetching the three basins this project uses is one request:

```bash
curl -o data/raw/regions.geojson   "https://geo.vliz.be/geoserver/MarineRegions/wfs?service=WFS&version=1.0.0&request=GetFeature&typeName=MarineRegions:iho&outputFormat=application/json&CQL_FILTER=name%20IN%20('Arabian%20Sea','Bay%20of%20Bengal','Indian%20Ocean')"

python scripts/load_regions.py
```

`scripts/load_regions.py` reads only that file and fails with an explicit message when it is absent, rather than reaching for the network itself. Marine Regions asks to be cited when its data is used; see their [citation guidance](https://www.marineregions.org/disclaimer.php).

Loading the polygons does not reassign the profiles already in the database, because `region` is written at ingestion time. Backfill them:

```bash
docker compose exec -T db psql -U gda -d gda -c "
WITH hit AS (
    SELECT p.profile_id, r.name
    FROM profiles p
    JOIN LATERAL (
        SELECT r.name FROM regions r
        WHERE ST_Contains(r.geom, ST_SetSRID(ST_MakePoint(p.longitude, p.latitude), 4326))
        ORDER BY ST_Area(r.geom) LIMIT 1
    ) r ON true
)
UPDATE profiles p SET region = hit.name FROM hit WHERE hit.profile_id = p.profile_id;"

python scripts/build_semantic_index.py
```

The smallest containing polygon wins, so a point inside both a marginal sea and the ocean around it is assigned the sea.

[`region_for`](src/ingestion/regions.py) keeps the inequalities as its fallback, because a float can sit inside no polygon this project loaded and ingestion must not fail or write a NULL region over it. That fallback is not hypothetical: the IHO publishes the Gulf of Aden and the Mozambique Channel as their own sea areas, so a float there matches none of the three and falls back to the old rule. 32 of 1,099 profiles, about 3%, are in that position. Each fallback is counted and the loaders print the count when they finish, so a return to the coarse rule is visible rather than silent.

What changed when the polygons replaced the inequalities, over 1,099 real profiles:

| Region | By inequalities | By polygon |
| --- | --- | --- |
| Southern Indian Ocean | 484 | 612 |
| Bay of Bengal | 321 | 282 |
| Arabian Sea | 294 | 205 |

128 profiles, nearly one in eight, were in the wrong basin. The inequalities put everything north of 5N and west of 78E in the Arabian Sea, which swept in a slice of ocean the IHO does not consider part of it.

### Asking in another language

The problem statement comes from an Indian ministry, so the languages that matter most are Hindi, Bengali, Tamil, Telugu, Malayalam and their neighbours. Ask in any of them and the answer comes back in the same language.

```
अरब सागर में औसत सतही तापमान क्या है
  detected   Hindi or Marathi
  as English What is the average surface temperature of the Arabian Sea?
  answer     5 row(s) प्राप्त होते हैं column वर्ष, क्षेत्र, mean_surface_temperature
```

Translation happens at the edge in [`src/router/translator.py`](src/router/translator.py), in front of routing and beside the follow-up rewriter. Everything inside stays English: the router, the SQL generator, the schema catalog and the SQL cache never learn that language exists, so the cache does not fragment into one entry per language for the same question.

**Detection is a script test, not a model call.** A question containing Devanagari is not English and no model is needed to establish that, so an English question costs nothing at all. That is the cheap half. The honest limit is written into the module: French and Indonesian are Latin script and will be taken for English. Setting `multilingual.always_translate` routes every question through the model instead, which catches those at the cost of one call on every question.

Two things are visible rather than hidden. The page shows the detected language and the English the query actually ran as, and the English answer is kept under an expander. A translation that changes the meaning of a question is the obvious new failure mode here, and it is only catchable if you can see both.

Every failure degrades to the behaviour that existed before the feature: a model that times out, returns nothing, or is switched off leaves the question exactly as typed.

This does not affect the [published numbers](#results). English questions skip the translator entirely and the SQL prompt is unchanged, so the 52-question benchmark measures the same system it did before.

### Load the drifting buoys

The second in-situ platform, and the point at which "extensible to other observations" stops being a claim. Argo floats profile the water column and surface every ten days; drifters ride the surface and report where the current carried them. Same ocean, different instrument, different table shape.

```bash
python scripts/load_drifters.py --year 2023
```

Source is the Global Drifter Program 6-hourly quality controlled product, served by [Ifremer's ERDDAP](https://erddap.ifremer.fr/erddap/). Public, no key, cached under `data/raw/drifters/` after the first run. The Indian Ocean box is the same one the Argo selection uses, so the two platforms describe the same water.

They get their own tables ([`db/008_drifters.sql`](db/008_drifters.sql)) rather than being forced into `profiles` and `measurements`. A drifter has no cycle, no depth levels and no descent, and pretending otherwise would put NULLs down every column that makes a profile a profile. What they share is what questions are asked in: a position, a time, a region and a temperature.

What that shared shape bought, with no code changed:

- the trajectory chart draws a buoy track exactly as it draws a float track
- `region` grouping works across both, from the same basin polygons
- `ST_DWithin` distance queries work on both, from the same geography column
- CSV and NetCDF export work on both

What it cost is in [Results](#results), and it is not nothing.

#### A fill value that read as a current

ERDDAP says "not measured" in two different ways and the loader only understood one. Alongside `NaN`, which it handled, the feed writes the numeric fill value `-999999`, which parses as a perfectly good float and was stored as one. 189 of 139,971 fixes arrived that way, 0.14% of the table, and they were invisible until the surface current was actually queried:

| Mean surface current speed | With the fill value | Cleaned |
| --- | --- | --- |
| Arabian Sea | 4011.854 m/s | 0.240 m/s |
| Bay of Bengal | 6243.030 m/s | 0.329 m/s |
| Southern Indian Ocean | 1606.288 m/s | 0.258 m/s |

The cleaned column is what a surface drifter measures; the other is thirteen times the speed of sound in water. Nothing in the test suite caught this, because the parser was tested against `NaN` and an empty cell and both were handled correctly. It survived ingestion, QC and a published benchmark, and what exposed it was reading a number off the screen and knowing it was impossible.

Fixed in three places, because one of them will be bypassed eventually:

- [`_number`](scripts/load_drifters.py) drops the sentinel at parse time, along with `NaN`
- a `CHECK` constraint in [`db/008_drifters.sql`](db/008_drifters.sql) refuses it at the database, where a loader written later cannot get round it
- the [catalog](db/schema_catalog.md) gives the units and the speed formula, since the model had been writing `u + v` for a hypotenuse

The 189 rows already loaded were set to NULL rather than deleted: the position, time and temperature on those fixes are good, and only the velocity was ever the fill value.

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

Pages: Home, Ask Questions, Ocean Data, Evaluation, Monitoring, Settings. The document corpus is the two Argo manuals, loaded by the snippet above rather than uploaded through the UI, because the corpus is fixed rather than something a user curates.

Or the HTTP service:

```bash
uvicorn src.api.main:app --reload
```

`POST /ask` takes `{"question": "...", "route_override": null, "session_id": null}` and returns the answer with citations or with the generated SQL, rows and timings. Passing the same `session_id` on a later request is what makes a follow up like "and in 2022?" resolvable. `GET /health` reports indexed documents, measurement count and whether the LLM is reachable. Interactive docs at `/docs`.

`POST /export` takes the same body plus `"format": "csv" | "netcdf"` and streams the result set as a file. It returns every row rather than the hundred `/ask` includes for display, and the NetCDF carries units, column descriptions and the generating SQL in its attributes, so a download stays readable and reproducible after it leaves the API.

Both UIs show the same two things the answer depends on: for documents, the retrieved chunks with similarity scores and the exact prompt sent to the model; for data, the SQL. If an answer looks wrong, you can see immediately whether retrieval failed, generation failed, or the query was simply right and surprising.

### The React front end

A second client over the same API, in [`web/`](web/). The Streamlit page reaches past the API into the pipeline directly, so it cannot find the places where the response is awkward to consume; a UI written against `AskResponse` can.

```bash
uvicorn src.api.main:app --reload    # the service it talks to
cd web && npm install && npm run dev
```

Same principle as the dashboard: the route, the SQL and the citations travel with every answer, and a refusal renders as a result rather than an error.

## Deploying

Three containers, with nginx the only thing listening and the database and API internal:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Naming both files is deliberate; `docker-compose.override.yml` is local-only and would serve the working tree. [`docs/deployment.md`](docs/deployment.md) covers loading the archive into a fresh volume, the settings whose defaults are committed passwords, and why Ollama is not a container here.

**Verified, not hosted.** Both images build (API 3.4 GB, front end 74 MB), the arrangement comes up with only nginx published, and a request traverses nginx to the API to the database on the real archive: `/api/health` reports 175,364 measurements through the proxy, and a question through the same path returns the scope gate's refusal in eight seconds. CI builds both images on every push and publishes them to `ghcr.io/poojithareddy19/floatchat-{api,web}` on `main`, tagged with the commit.

Proving it found three defects that `docker compose config` had passed: the Dockerfile pulled the CUDA build of torch into a GPU-less container and died out of memory; nginx resolved the API's hostname at startup and refused to start without it; and the front end's healthcheck asked `localhost`, got IPv6, and reported a working server as down. All three are fixed and written up in [`docs/deployment.md`](docs/deployment.md).

There is no cloud host, by choice. The images are published and the arrangement is proven; putting it on a paid instance is a decision, not a remaining task.

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

The assistant is an MCP server ([`src/mcp/server.py`](src/mcp/server.py)), and **the chat on the web page is one of its clients**.

That second half is recent. For most of this project's life the server was a separate door: an external client such as Claude Desktop could use its tools, and the chat never did, so the problem statement's "use Model Context Protocol" described an accessory rather than the system. Now [`src/mcp/client.py`](src/mcp/client.py) opens a real MCP session against the server from inside the app, runs the initialize handshake, calls a tool by name and reads the JSON it returns. It is the same request any client makes; only the transport differs, a pair of in-memory streams instead of stdin and stdout, so there is no subprocess or port per question.

The chat uses it for questions with one fixed shape, detected by rules in [`src/router/tools.py`](src/router/tools.py) before the router runs. "What are the nearest ARGO floats to 10.5N 65.2E?" goes to `nearest_floats`; "show the profile of float 1900083" goes to `get_profile`. Anything carrying an extra condition, a year, a month, a measured quantity or a statistic, is left to the SQL path, which can honour it: a tool returning the right floats for the wrong year is a wrong answer delivered confidently. If the server cannot be reached the question falls through to generated SQL rather than being lost. The page shows the tool and its arguments where a generated answer shows its SQL.

"Nearest floats to *this location*" needs a location a chat box does not have, so the page has a location panel. It is off unless ticked, and a question that names its own position keeps it. A nearest-floats question with no position at all is answered by asking for one rather than by guessing.

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

Five tools:

| Tool | Arguments | Returns |
| --- | --- | --- |
| `query_argo` | `question` | answer, generated SQL, columns, rows |
| `list_floats` | `region`, `year` (both optional) | float ids with profile counts and date ranges |
| `get_profile` | `float_id`, `cycle` (optional) | one float's depth levels |
| `nearest_floats` | `latitude`, `longitude`, `limit` (optional) | floats ranked by their closest profile, with the distance in km |
| `describe_schema` | none | the column catalog as text |

`nearest_floats` answers the problem statement's third example. Per float it keeps the single closest profile, so a float that cycled past the same spot forty times is one float near here, not forty, and it measures in metres on the sphere through PostGIS `geography` rather than in degrees, which mean different distances at different latitudes.

**There is deliberately no tool that accepts SQL.** The four safety layers exist so that generated SQL is constrained before it reaches the database, and a `run_sql` tool would hand an external client a way around all of them. A test asserts that no registered tool takes a statement, under any argument name, and that every tool's schema closes `additionalProperties` so one cannot arrive under a different one.

`list_floats`, `get_profile` and `nearest_floats` do not go through the model at all. Their shape is fixed, so they are parameterised queries written by hand in the server module, capped and executed as the read-only role.

## Evaluation

Two harnesses, because the two paths fail differently.

**Retrieval** ([`src/evaluation/evaluator.py`](src/evaluation/evaluator.py)). Recall@K, Precision@K, Hit@K, MRR, nDCG@K against `data/evaluation/questions.csv`. Relevance is anchored at **document and page** level rather than chunk ID, because chunk IDs renumber whenever chunk size changes, which would invalidate the entire gold set every time a chunking parameter is tuned.

#### Retrieval results

25 questions written against the two manuals in the corpus, each anchored to the page that answers it. Every anchor was checked to exist in the index, so a miss is retrieval failing rather than the gold set pointing at nothing.

| Metric | k=1 | k=3 | k=5 |
| --- | --- | --- | --- |
| **Hit@k** | 0.640 | 0.920 | **0.960** |
| Recall@k | 0.640 | 0.920 | 0.960 |
| Precision@k | 0.640 | 0.307 | 0.192 |
| nDCG@k | 0.640 | 0.862 | 0.941 |
| MRR | | | 0.788 |

**The first version of this table read 0.200 at k=5, and the gap between the two is the most useful thing in this file.**

The cause was the vector index. `db/002_pgvector.sql` created `ivfflat (embedding vector_cosine_ops) WITH (lists = 100)`, a setting copied from pgvector's guidance for tens of thousands of rows. IVFFlat partitions the vectors into `lists` clusters and, at the default `ivfflat.probes = 1`, searches exactly one cluster. With 668 chunks in 100 lists that is roughly seven candidates per query.

The symptom that gave it away was not the low score. It was this:

```
gold page in top-20 candidates: 5/25
gold page in top-50 candidates: 5/25   <- identical
gold page in top-100 candidates: 24/25
```

Asking for more candidates returned the same rows, which no honest ranking does. Switching the index to HNSW changed hit@5 from 0.160 to 0.960 with nothing else touched.

Two things this cost, both worth recording:

**The obvious hypothesis was wrong.** Every page of both manuals carries the same running header, so the first fix attempted was stripping repeated headers and footers during chunking. It was implemented, tested and measured, and it made retrieval *worse*: hit@5 0.840 with stripping against 0.960 without, on a fixed index. The header is useful signal, not noise, because it names the manual a chunk came from. The code was reverted. Had the index bug not been found first, that change would have shipped as an improvement on the strength of a number that rose from 0.160 for unrelated reasons.

**One published metric was wrong.** `recall_at_k` summed hits over the retrieved list, so two chunks from the same relevant page counted twice and recall could exceed 1. It now counts distinct pages.

Reproduce from the Evaluation page of the dashboard, or:

```python
from src.evaluation.evaluator import evaluate_retrieval
from src.utils.pipeline import RAGService

report = evaluate_retrieval(RAGService().retriever, "data/evaluation/questions.csv", k=5)
```

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

67 questions against **real data from both platforms**: 80 Argo floats, 1,099 profiles and 175,364 measurements from the Indian Ocean spanning 2001-07-23 to 2026-09-17, and 187 drifting buoys with 139,971 six-hourly fixes through 2023, across the Arabian Sea, the Bay of Bengal and the Southern Indian Ocean. 49,296 measurements carry a dissolved oxygen reading.

The set was 52 questions and entirely Argo until the `drifter` bucket was added. Every figure below is the larger set, so it is not comparable line by line with the 0.620 and 0.587 quoted elsewhere in this README, which were measured on the 52.

Two models, both local through Ollama, on the same questions and the same prompt. `llama3.1:8b` is the default the system ships with; `qwen2.5-coder:7b` is a code specialist, run to find out whether the two ceilings this benchmark keeps hitting are the model's or the prompt's.

| Metric | `llama3.1:8b` | `qwen2.5-coder:7b` |
| ------------------------- | ----- | ----- |
| **Execution accuracy**    | **0.567** | **0.483** |
| Validation pass rate      | 0.983 | 0.967 |
| Execution rate            | 0.933 | 0.850 |
| **Correct refusal rate**  | **1.000 (7/7)** | 0.857 (6/7) |
| False refusal rate        | 0.017 | 0.050 |
| Median latency            | 31.5 s | 50.2 s |

| Bucket | Questions | `llama3.1:8b` | `qwen2.5-coder:7b` |
| ------------ | --- | ----- | ----- |
| filter       | 8   | 0.875 | 0.500 |
| easy         | 8   | 0.875 | 0.250 |
| groupby      | 6   | 0.833 | 0.833 |
| join         | 8   | 0.500 | 0.500 |
| drifter      | 14  | 0.429 | **0.571** |
| qc           | 5   | 0.400 | 0.400 |
| bgc          | 5   | 0.400 | **0.600** |
| window       | 6   | 0.167 | 0.167 |
| unanswerable | 7   | **7/7 refused** | 6/7 refused |

Both columns are one run each on the same 67 questions, the same prompt and the
same timeout, taken back to back with nothing else using the machine. This is
the first time the two models have been compared on equal terms; the earlier
qwen figures were measured on a different, smaller set.

Six of the seven unanswerable questions are refused by the [scope gate](#closing-it-without-the-model)
and will not move between runs or between models. The seventh is the one added
with the drifter bucket, and the gate does not catch it, for the reason given
when it was built: the gate keys on the quantity asked for, and depth is a
quantity this database holds, just not for this platform. Asked how deep each
buoy dived, the model joined the buoys to `measurements` and averaged
`pressure_dbar` into a column it called `mean_depth`. The two platforms share
no key and the catalog says so; the join it invented is the platform boundary
being crossed, not a depth being measured.

The false refusal rate counts three things that are not the same. Most of it is
`ReadTimeout`: the model did not answer in time, which the harness cannot tell
apart from a decline. That had a cause worth naming. **The harness was running
a stricter timeout than the system it measures**, using a function default of
90 seconds while `config.yaml` gave the app 120, so generations that would have
finished were scored as refusals. Both now read the same config.

The remaining one is a real rejection, and it is correct. The query put a
derived table's alias inside a sibling subquery, which PostgreSQL does not
allow: `ERROR: relation "yearly" does not exist`. The validator caught invalid
SQL before the database did, which is its job. An earlier draft of this
paragraph called that a validator limitation. It was not; the query was broken.

llama3.1 was previously the mean of two runs, qwen a single run. Both were measured before the IHO basin polygons replaced the latitude and longitude inequalities, which moved 128 profiles between regions. The set was re-run afterwards to check that it had not quietly shifted the score: 0.630, inside the 0.609 to 0.630 the two earlier runs produced, with every bucket unchanged. Reference and generated queries both read the same database, so a change in the data moves them together. Quantisation differs and is worth stating: llama3.1 is Q4_K_M, qwen is `7b-instruct-q3_K_M`, one step lower, because the larger file would not fit on the disk it was run from. Some of qwen's weakness on the easy buckets plausibly belongs to that rather than to the model.

Neither model is simply better. qwen is worse overall, much worse on the simple buckets, clearly better on grouped aggregates, and perfect at refusing. Picking one is a trade, not an upgrade.

#### What adding a second platform cost

The drifting buoys were added after the table above was measured, which changed the prompt and the schema the model sees. Re-running the same 52 questions on `llama3.1:8b` afterwards:

| | Before drifters | After drifters |
| --- | --- | --- |
| Execution accuracy | 0.620 | 0.609 |
| **Correct refusal rate** | **0.833 (5/6)** | **0.667 (4/6)** |

Accuracy is flat, inside the run-to-run spread. The refusal rate is not, and the cause is specific: drifters carry surface current velocities, so the database now contains a current speed for the first time. Asked "what is the current speed at 1000 decibars", the model stops refusing and answers, because something called current now exists. It is still wrong, since a drifter measures the surface and nothing else, and the catalog says so in the sentence directly under the column.

**Adding a data source widened the hallucination surface.** That is the honest cost of the extension, in the one metric this project claims to care most about, and prompt wording did not close it. It belongs next to the feature rather than in a footnote.

#### Closing it, without the model

Three attempts at prompt wording failed, which is already recorded above. The catalog said "this is the only current speed in the database, and it is at the surface only" in the sentence directly under the column, and said of `pressure_dbar` that it is "not the depth of the seabed, which this database does not record". Both were read and both were overridden, the second by aliasing `max(pressure_dbar)` to `seafloor_depth`.

So the rule stopped being advice to a model and became code that runs before one is called. [`src/sqlgen/scope.py`](src/sqlgen/scope.py) refuses two kinds of question:

- a quantity with no column anywhere in the schema: wind, rainfall, the depth of the seabed, waves, marine life, pollution
- a quantity held only at the surface, asked for at depth: the buoy velocities, which exist for the sea surface and for nothing below it

Both lists are read off the schema rather than off this gold set. The refusal returns `UNANSWERABLE` with the reason attached, so it travels the path a model's own refusal already took and the user is told which quantity is missing rather than that the question failed.

| Correct refusal rate | Before drifters | After drifters | With the scope gate |
| --- | --- | --- | --- |
| `llama3.1:8b`, on the original six | 0.833 (5/6) | 0.667 (4/6) | **1.000 (6/6)** |
| False refusals contributed by the gate | - | - | **0 of 60** |

That row is the six unanswerable questions the set had at the time, held constant so the three columns compare. A seventh was added later with the drifter bucket and the gate does not catch it, which is why the [headline](#results) reads 6/7 rather than 6/6. The gate keys on the quantity asked for, and the question asks how deep each drifting buoy dived: depth is a quantity this database holds, just not for that platform. It is the limit below, arriving immediately and from the first new questions written after the gate was built.

**These two numbers are exact rather than sampled.** The gate is a lexical test over the question, with no model call, so it does not move between runs and does not depend on which model is shipped: `qwen2.5-coder`'s perfect refusal score, [reported below](#the-seafloor-question-on-a-second-model) as evidence that the gap was the model's, is now something neither model is asked to supply. The six refusals are also immediate, where they previously cost a full generation each.

The limit is written into the module rather than left for a reader to discover. This is lexical matching, so a paraphrase it does not carry goes to the model exactly as before: "how deep is the ocean under float 1900083" needed a pattern added during testing, and there will be others. It can only add a refusal, never remove one, so a question it says nothing about is generated and validated on the path it always used.

#### What the drifter bucket found

For most of this project's life the gold set had 52 questions and not one was about the drifting buoys. The platform was reachable, exported, charted and refused against, and none of it was measured. 14 answerable questions and one unanswerable were added. The bucket scores **0.429**, second worst on the board.

The failures are not spread evenly. They are three specific things, and the first two are caused by the fix that preceded them.

**The worked example taught the shape along with the query.** The example added for the velocity columns is `Average surface current speed in each region`, which groups. Asked for the average in one named region, and asked for the single fastest current overall, the model returned a per-region breakdown both times. Neither question asked for one. The example fixed the arithmetic it was written for, `sqrt(u^2 + v^2)` rather than `u + v`, and taught a `GROUP BY` nobody asked for alongside it.

**It did not fix what it was written for.** The buoy example exists because the model grouped by a `region` column on `drifters`, which does not exist, since region belongs to the fix rather than to the instrument. Asked which buoys reported from more than one region, the model wrote `SELECT count(*) FROM drifters WHERE region IS NOT NULL GROUP BY region`. The same mistake, with the example in the prompt.

**The platform boundary is not holding.** Asked to compare float and buoy temperatures, the model joined `measurements` to `drifter_observations` directly and the query ran until the statement timeout killed it. Asked how deep the buoys dived, it joined them to `measurements` and averaged `pressure_dbar`. The catalog says in as many words that the two platforms do not join and that a question comparing them is answered by aggregating each separately. It is read and overridden, which is the same failure mode as the seafloor alias and suggests the same remedy: a rule that holds has to be code, not prose.

A fourth failure is worth separating because it is not about buoys at all. Asked how many drifting buoys are in the database, the model wrote `SELECT count(*) FROM drifters WHERE buoy_type = 'SVPB'`, inventing a filter from a value that appears in the retrieved summaries. That is exactly what the `CONTEXT_RULE` in the prompt forbids, in the same prompt that forbids it.

**None of this was visible before the questions existed.** The A/B that cleared the worked example of costing accuracy was run on the 52-question set, where the example is shown to no question at all, so it could not have detected the over-grouping it causes. A benchmark that does not cover a feature will report that the feature is fine.

#### The window bucket is a ceiling, and here is the evidence

Three independent attempts to move `window` off 0.167, and a fourth line of evidence that it is not the prompt:

| Attempt | Result |
| --- | --- |
| Three worked examples, one per window function, against these exact tables | 0.333 to 0.333 on the set of the day; no movement |
| A repair attempt given the error the query failed with | no window question recovered |
| A second model, `qwen2.5-coder:7b`, on the same 67 questions | **0.167, identical** |

The third is the one that settles it. A general 8B model and a 7B code specialist, different architectures and different training, produce the same score on the same six questions. Whatever is failing is not something one model knows and the other does not.

**qwen's earlier advantage did not survive a fair comparison.** On the 52-question set it scored 0.333 on this bucket, double llama, and that was quoted here as a model difference. Measured on the same questions with the same prompt it is 0.167. A single-run difference that looked like a property of the model was noise, which is what the caveat now printed above every single-run result exists to warn about. Retracting it is cheaper than defending it.

So the reasonable conclusion is that window functions over this schema are beyond a 7-8B model at this quantisation, and the next thing worth trying is a larger model rather than a better prompt. That is a hardware question, and it is on the roadmap rather than in this section.

#### Neither model is better, they are wrong about different things

| Bucket | `llama3.1:8b` | `qwen2.5-coder:7b` |
| --- | --- | --- |
| easy | **0.875** | 0.250 |
| filter | **0.875** | 0.500 |
| drifter | 0.429 | **0.571** |
| bgc | 0.400 | **0.600** |

qwen is 0.084 worse overall and better on the two hardest data buckets. It collapses on `easy`, which is eight questions of the form "how many floats are in the database", and some of that plausibly belongs to quantisation rather than to the model: qwen is `q3_K_M`, a step below llama's `Q4_K_M`, because the larger file would not fit on the disk it was first run from.

Picking one is a trade rather than an upgrade, and the refusal row is no longer part of it: the [scope gate](#closing-it-without-the-model) declines before either model is called, so that number is the same whichever is shipped.

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

That conclusion was right and is now moot. Both halves of it assumed the choice was which model to ask, and the [scope gate](#closing-it-without-the-model) does not ask one: `llama3.1:8b` refuses the seafloor question now, not because it learned anything but because the question no longer reaches it. The lesson worth keeping is the one the section was written to make, that a fabrication surviving three rewordings is telling you something about where the rule belongs, rather than asking for a fourth.

Validation pass rate is 0.983 while accuracy is 0.567. Nearly every generated query was well formed, safe and executable, and four in ten still answered the wrong question. That gap is the entire argument for measuring results rather than liveness.

#### On repeated runs

The three-run evaluation on an earlier prompt produced byte-identical SQL for all 52 questions and an identical score in all three runs, which is what temperature 0 should do. That is not what happens now, and the gap has widened rather than closed.

Two full runs were taken back to back on this machine, the second differing only in whether the drifting buoy examples were in the prompt, which on this gold set means they were absent from all 52 questions either way. The runs were therefore identical in everything the model saw. They scored identically, 27 of 46, and **disagreed on four questions**, two flipping each way:

| Question | Ungated | Gated |
| --- | --- | --- |
| How many measurements are deeper than 1000 decibars? | wrong | right |
| How many profiles were recorded between 2010 and 2015? | right | wrong |
| Average surface temperature for each month of the year | wrong | right |
| What is the average pH of the water? | right | wrong |

Temperature 0 selects the most likely token; it does not make the arithmetic that ranks them reproducible, and under memory pressure this machine is visibly less reproducible than the one the earlier runs were taken on. Four of 46 is a swing of 0.087 available to any single run.

**So a single run cannot separate a real change from noise here, and several comparisons in this README were made from single runs.** A difference of one or two questions between configurations is not evidence of anything, including the 0.620 to 0.587 move above. The `--runs N` flag exists for this and should be used for any claim that matters; it was not affordable for these runs at roughly 30 minutes each on this hardware, which is a limitation of the measurement rather than of the system.

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

### Scoring the prose, not just the retrieval

Two of the three things worth measuring were measured. Retrieval was scored with hit@5, recall and nDCG. Text-to-SQL was scored by executing the generated query against a reference. Nothing scored the **sentence handed to the user**, which is the half the project's central claim rests on: "being confidently wrong is worse than saying nothing" is a statement about generated text, and the only evidence for it was that retrieval had found the right page.

[`src/evaluation/generation_metrics.py`](src/evaluation/generation_metrics.py) scores it, in two groups.

**Deterministic, no model call.** These cost nothing and cannot drift:

- `answered_rate` - how often the system answers rather than declining. A system that refuses everything scores perfectly on faithfulness, so no quality number means anything without this beside it.
- `citation_accuracy` - of the answers given, how many cite the document and page the gold set names. An answer citing the wrong page is not grounded, and no model is needed to check that.
- `lexical_support` - the share of the answer's content words that appear in the passages it was given. A **proxy** rather than a measure, and reported as one: paraphrase scores low and fluent copying scores high. It is a floor. An answer near zero is not reading its sources.

**Judged, one model call per question, behind `--judge`:** faithfulness, answer relevance and correctness against the gold answer.

```bash
python -m src.evaluation.generation_metrics            # deterministic only
python -m src.evaluation.generation_metrics --judge    # adds the graded three
```

The honest limit is written into the module: the judge is the same local model that wrote the answer, so it is a poor judge of its own output and shares its own blind spots. Read `citation_accuracy` first, which nothing can talk its way out of, and treat a judged score as a smoke alarm rather than a measurement. A judge that cannot be reached is recorded as unknown and left out of the mean, so an outage cannot look like a quality drop.

### What these numbers do not tell you

- **One model, one prompt, one schema.** The set has never been run against a second model, so nothing here separates what this model cannot do from what no model could do with this prompt.
- **Execution accuracy has false positives.** A wrong query can collapse to the right number when the data happens to cooperate, and the comparison cannot tell that apart from understanding.
- **The BGC sample is narrow.** Ten biogeochemical floats were loaded and none of them carries a NITRATE sensor, so that column is empty and the `bgc` bucket tests oxygen, chlorophyll, pH and backscatter only.
- **Profile counts are shaped by the download, not by the ocean.** The sample takes a bounded number of cycles per float, so "which float reported most" is answerable but is a fact about what was fetched rather than about the fleet.
- **The document gold set is small.** `questions.csv` is 25 questions with real ground-truth passages, but it was written by the same person who chose the corpus, which is the honest limit on what hit@5 and the generation scores show.
- **The drifter bucket is new and shallow.** 14 questions is enough to show the platform is the weakest path and not enough to characterise it. It covers one year of buoys and three hull types, and no question tests the trajectory or distance queries the platform also supports.
- **Single runs.** Every figure above except the refusal rates comes from one run, and two identical back-to-back runs disagreed on four of 46 questions. Differences smaller than about a tenth are not resolvable without `--runs N`.

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

**Data.** The database holds real Argo profiles: 80 floats, 1,099 profiles and 175,364 measurements from the Indian Ocean. Ten of those floats carry biogeochemical sensors, which populates oxygen, chlorophyll, pH and backscatter, but none carries a NITRATE sensor, so that one column is still empty. The sample takes a bounded number of cycles per float rather than every cycle, so per-float profile counts reflect the download rather than the fleet. Region comes from IHO basin polygons in PostGIS, with the old latitude and longitude inequalities kept as the fallback for a position inside none of the three loaded basins, which is 3% of profiles.

**Conversation.** A follow up is rewritten into a standalone question before routing, which keeps the router, the SQL generator and the cache stateless and keeps the cache keyed on what was actually asked. The history behind that rewrite is not persisted: the API holds it in a process-local dict, so it does not survive a restart and does not work across workers, and the Streamlit page holds it in session state, so it disappears with the browser session. Moving it to Redis or a sessions table is the fix and has not been done. A follow up whose history is lost degrades to being answered as a standalone question rather than to an error, and a rewrite that goes wrong is visible: the page shows what the question was answered as, and both versions are written to the log.

**Architecture.** A question needing both sources is now answered from both, but the join between them is a sentence rather than a computation: the synthesis step is forbidden to do arithmetic, so it can report that a flag means bad data and that 812 measurements carry one, and it cannot tell you what fraction of the record that is unless the query already did. Whether a question is compound is decided by regex and, failing that, by the router model, and a compound question phrased without any of the document words reaches the data path whole. Summaries are rebuilt only when the script is run, so they drift from the tables between loads. Summarising is per float and per region; a database with thousands of floats will want coarser grouping than one summary each.

**Retrieval and generation.** Retrieval over the manuals reaches hit@5 of 0.960 on 25 questions, but that gold set was written by the same person who chose the corpus, which is the honest limit on what it shows. One question in twenty-five is still missed at k=5. No models are trained here. Extraction quality depends on the source PDF; scanned PDFs need OCR, which is not wired in. Confidence is a heuristic over retrieval signals, not a calibrated probability, so the threshold needs tuning against your own gold set. Embeddings are English-only, so the manual retrieval path does not benefit from the translator the way the SQL path does. Language detection is by script, so a Latin-script language other than English is taken for English unless `always_translate` is set. No individual document deletion.

**Charts.** The map basemap is served locally. Plotly fetches the land and coastlines of a geo plot as TopoJSON at render time and defaults to `cdn.plot.ly`, which made the one networked dependency in an otherwise offline app a map that came up empty with no error to explain it. The files are committed under `static/` and Plotly is pointed there through `topojsonURL`. Trajectories, depth profiles, depth-time sections and T-S diagrams are drawn as interactive Plotly figures, dispatched on column names because latitude and longitude are two ordinary floats to a dtype check. Everything else falls through to the matplotlib line and bar builder, and a PNG is produced in every case so an image client keeps working. The dispatch is first-match, so a result carrying positions is always drawn as a track even when it also carries measurements.

**Evaluation.** Text-to-SQL scores 0.567 execution accuracy over 67 questions on the shipped model and 0.483 on a second one, measured on the same questions. Complex queries are much worse than either average suggests: the window bucket is 1 in 6 on **both** models, and worked examples, a repair attempt and a change of model have each failed to move it. Figures are single runs, and two back-to-back runs of an identical configuration disagreed on four of 46 questions, so none is precise to better than about a tenth. The repair loop is worth 2 recovered queries of 6 attempts on llama and cannot touch the 21 questions that fail by returning the wrong rows without an error. The seafloor fabrication and the current at 1000 decibars are both refused now, but by a lexical gate in front of the model rather than by anything the model learned, so the gold set measures the gate on those six questions and not the generator. A paraphrase the gate does not carry reaches the model exactly as before. Both columns come from one gold set of 52 questions written by the same person who wrote the schema, which is a real limit on what they can show. The document retrieval side has no published numbers at all.

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
- [x] CSV, Parquet and NetCDF export, with units and the generating query attached
- [x] PostGIS geometry on profiles, with a spatial region lookup and `ST_DWithin` distance queries
- [x] IHO basin polygons loaded and the existing profiles backfilled, moving 128 of 1,099 into a different basin
- [x] A second in-situ platform: 187 drifting buoys, 139,971 fixes, sharing the charts, regions and export with no code changed
- [x] A conversational thread rather than a single question box, with each turn keeping its own chart, table and SQL
- [x] Multilingual questions, answered in the language they were asked in, with detection that costs nothing for English
- [x] One answer from the manuals and the database together, split in two and rejoined, with both halves shown
- [x] An MCP server, with no tool that accepts SQL
- [x] The chat as an MCP client: fixed-shape questions such as the floats nearest a point go to the server over the protocol, not around it
- [x] Relative periods anchored to the archive: the SQL prompt is told where each platform's data starts and ends, read from the tables
- [x] One trace per cast in profile plots, drawn whenever the rows are profiles rather than only when the question says "plot"
- [x] Few-shot window-function examples, measured: they did not move the bucket
- [x] A second model on the same 52 questions, which closed the refusal gap
- [x] A document retrieval gold set that is not a five-row template, with numbers published
- [x] Retrieval fixed: the IVFFlat vector index was searching one cluster of a hundred. HNSW took hit@5 from 0.160 to 0.960
- [x] The refusal the drifters reopened, closed by a scope gate in front of the model rather than a fourth rewording: it refuses 6 of the 7 unanswerable questions and none of the answerable ones
- [x] ERDDAP's numeric fill value dropped at parse time, at the database and in the catalog, after 189 fixes stored `-999999` as a surface current
- [x] The drifting buoy examples shown only to questions that could be about the buoys, so the other 51 do not carry them
- [x] Drifter questions in the gold set, 14 answerable and one not, which found the buoy path is the weakest measured and that the buoy example causes the over-grouping it was meant to cure
- [x] Generation scored, not just retrieval: citation accuracy and answer rate deterministically, faithfulness and relevance behind a judge flag
- [x] A repair attempt on a query that failed, given the error it failed with, with refusals never repaired
- [x] Cross-platform joins rejected in the validator, including the trivial-subquery form a model found to get round the first version
- [x] A React front end over the API, which is the second client the response contract needed
- [x] The Docker stage, verified: both images build, the production compose comes up, and a request traverses nginx to the API to the database on real data. CI builds both images on every push and publishes them to GHCR on main
- [x] GitHub Actions running the suite and the linter on 3.11 and 3.12
- [x] A single run now says so in its own output, rather than leaving the caveat to a README paragraph
- [x] Both models on the same 67 questions, which retracted an earlier claim: qwen's advantage on the window bucket was noise, and both score 0.167 there
- [x] A repair guard, after the repair loop answered how deep the buoys dived with sea surface temperature aliased to min_pressure

Not done, honestly:

- [ ] Persist conversation history, which currently dies with the process
- [ ] Load the data the problem statement's own examples ask for. There are no profiles in the equatorial band in March 2023, and no BGC readings in the Arabian Sea in the last six months of the archive, so both examples return correct, empty queries
- [ ] Rename the Southern Indian Ocean region, which is the fallback for anything outside the named basins and so includes floats north of the equator
- [ ] A judge that is not the model under test, which is the honest limit of the generation scores
- [ ] A larger model on the same 67 questions. Three prompt attempts, a repair loop and a second 7B model have all left the window bucket at 0.167, so the next honest experiment is more capacity rather than better wording
- [ ] A cloud host. Deliberately not: the images are published and the arrangement is proven locally, and putting it on a paid instance is a decision rather than a task
- [ ] A nitrate-carrying BGC float, since none of the ten sampled floats has that sensor
- [ ] Hybrid search (BM25 + dense), which would likely help the exact-phrase manual questions most
- [ ] OCR path for scanned PDFs
