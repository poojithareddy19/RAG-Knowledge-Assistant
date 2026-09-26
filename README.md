# FloatChat

Ask the Argo float archive a question in plain English and get an answer you can check.

Two kinds of question, one system. **What does the archive hold?** is answered from one-sentence summaries of every float and every region in the database, retrieved by embedding, with the float or region behind every claim. **What does the data say?** is answered by SQL generated against the measurements database, shown to you before you are asked to believe the numbers. A router decides which, so the user asks a question rather than choosing a tool.

Both halves are built around one constraint: being confidently wrong is worse than saying nothing. The summaries path cites its source and declines when the evidence is thin. The data path shows the exact query that produced the table, and that query runs as a read-only database user after passing a validator.

> **Build status.** Working end to end: NetCDF ingestion of real Argo profiles with core and biogeochemical parameters and per-parameter QC, a semantic layer of float and region summaries in pgvector that both answers questions directly with citations and tells the SQL generator what the database actually holds, text-to-SQL with a four-layer safety path, a rule-first query router, conversational follow-ups, a second in-situ platform in 187 drifting buoys, ocean charts including overlaid profile comparisons, a FastAPI service and a React front end that draws the charts, with the Docker stage proven and images published from CI. Measured: [0.617 execution accuracy](#results) on 67 SQL questions, [7/7 correct refusals](#closing-it-without-the-model), and [hit@5 of 0.957](#retrieval-results) on 23 summary questions. See [Limitations](#limitations) and [Roadmap](#roadmap).

---

## Problem

This is a build against problem statement SIH25040 from the Ministry of Earth Sciences: a retrieval-augmented conversational interface over ARGO ocean data, for the Indian Ocean, with a vector store of metadata and summaries, text-to-SQL over a relational store, and a dashboard that shows profiles, trajectories and comparisons.

Argo is about 4,000 autonomous floats drifting the world ocean, diving to 2,000 decibars and surfacing every ten days with a profile of temperature and salinity. All of it is public. Almost none of it is reachable.

Reaching it means knowing the NetCDF layout, the QC flag conventions, which variable to read in which data mode, and SQL. A researcher, a student or a policymaker who can describe what they want in one sentence cannot get it.

Two questions come first, minutes apart, from the same user. "What is in here?" is answered by describing the floats and regions the archive holds. "What is the average surface temperature in the Arabian Sea?" is answered by a query. The failure mode is the same on both paths, a fluent and unverifiable answer, so both are built with the evidence surfaced and a refusal path.

## Architecture

```
                         question
                            │
                   ┌────────▼────────┐
                   │     router      │  rules first, model fallback
                   └────┬───────┬────┘
              summaries │       │ data / chart
          ┌─────────────▼─┐   ┌─▼──────────────────────────┐
          │ embed query   │   │ retrieve data summaries    │
          │ named float   │   │   (what the db holds)      │
          │   looked up   │   │ generate SQL (cached)      │
          │ pgvector top-k│   │ validate: SELECT only,     │
          │ confidence    │   │ allowed tables, real       │
          │   gate        │   │ columns, LIMIT capped      │
          │               │   │ execute as read-only user  │
          └───────┬───────┘   └─────────┬──────────────────┘
                  │ below threshold?    │
              decline                 rows ──▶ optional chart
                  │                     │
                  ▼                     ▼
      answer + float/region cited  table + the SQL that made it
                  │                     │
                  └────────┬────────────┘
                           ▼
                  one JSON log line
```

### The semantic layer

A schema tells a model which columns exist. It cannot tell it that float 1901393 is real, worked the Arabian Sea, and stopped reporting in 2021. Without that, "what did 1901393 measure off Goa" is answered by a model guessing whether such a float exists.

So every float and every region is summarised into a sentence, embedded, and stored in a pgvector table ([`src/semantic/`](src/semantic/)):

> ARGO float 1902458 is a NAVIS_A platform and part of project GO-BGC, WHOI. It recorded 20 profiles from 2023-06-14 to 2023-12-21. It reported in the Arabian Sea. It is a BGC float and also measures dissolved oxygen, chlorophyll, pH and particle backscatter.

The table does two jobs.

**It is the corpus of the summaries route.** A question about what the archive holds, "tell me about float 1902458", "which floats measure oxygen in the Arabian Sea", "describe the Bay of Bengal data", is answered from the nearest summaries by a model that is forbidden prior knowledge and required to cite them. The citation is the float or region itself, which a reader can then check against the tables the summary was computed from. A float named in the question is looked up literally rather than embedded, because float summaries are near-identical sentences whose only distinguishing token is a long number, which is what embeddings represent worst.

**It is the context for SQL generation.** The summaries closest to a data question are handed to the SQL generator before it writes anything, under an explicit instruction never to quote a number out of them. They resolve names and ranges; the query computes every value. Both UIs show which summaries were used, so the grounding is inspectable rather than invisible.

The router ([`src/router/classifier.py`](src/router/classifier.py)) tries cheap regex rules first and only pays for a model call on the ambiguous cases. Descriptive phrasings such as "tell me about" and "describe" go to the summaries; quantities, averages and counts go to SQL; "plot", "map" and "track" go to a chart. Its fallback route is the summaries path, which declines when nothing is close enough, so a question the router cannot place ends in a refusal rather than in SQL written against a guess.

### Two independent hallucination guards, on the summaries path

1. **Retrieval-side.** A confidence threshold over retrieval signals, evaluated before the model is called. Unanswerable questions are refused without spending a token.
2. **Generation-side.** The prompt requires the model to emit `INSUFFICIENT_CONTEXT` when the summaries do not support an answer, checked after generation.

Two rather than one because they fail independently: the first catches "nothing relevant was retrieved," the second catches "something was retrieved but it does not answer the question."

### Four layers of protection, on the data path

1. **Prompt.** Schema, conventions and worked examples, with an explicit `UNANSWERABLE` escape hatch.
2. **Validator** ([`src/sqlgen/validator.py`](src/sqlgen/validator.py)). One statement, `SELECT` only, no forbidden keywords or identifiers, no chaining, only allowed tables, qualified columns checked against the live `information_schema`, no join across the two platforms, and a `LIMIT` capped or injected.
3. **Database role** ([`db/003_roles.sql`](db/003_roles.sql)). Generated SQL runs as `gda_ro`, which holds `SELECT` and nothing else, with writes explicitly revoked.
4. **Connection.** `read_only` on the session plus a `statement_timeout`, so a pathological query cannot hold the pool.

Layer 2 alone would be a regex arms race. Layers 3 and 4 mean a validator bug is not a security incident.

Full reasoning for each choice is in [`docs/design_decisions.md`](docs/design_decisions.md).

---

## Technology stack

| Concern         | Choice                                    | Notes                             |
| --------------- | ----------------------------------------- | --------------------------------- |
| Language        | Python 3.11+                              | tested on 3.11 and 3.12           |
| Ocean ingestion | `xarray` + `netCDF4`                      | Argo NetCDF, drifters over ERDDAP |
| Structured data | PostgreSQL 16 + PostGIS                   | core CTD plus five BGC parameters |
| Vector store    | pgvector, cosine, in the same database    | one summary per float and region  |
| Embeddings      | `sentence-transformers` (BAAI/bge-small)  | swappable via config              |
| LLM             | Ollama, local, no API key                 | `llama3.1:8b`, `qwen2.5-coder:7b` |
| Text-to-SQL     | `sqlparse` validation + read-only role    |                                   |
| Charts          | Plotly for ocean plots, matplotlib else   | PNG produced in every case        |
| API             | FastAPI + Pydantic                        | the one entry point to the system |
| UI              | React, charts drawn with Plotly           | a client of the API only          |
| Evaluation      | retrieval, generation and SQL execution metrics |                             |
| Logging         | structured JSONL                          | one line per question             |
| Tracing         | OpenTelemetry, GenAI conventions          | JSONL by default, OTLP optional   |
| Config          | `config.yaml` + environment overrides     |                                   |
| Delivery        | Docker images built and published by CI   | no cloud host, by choice          |

## Project structure

```
FloatChat-ARGO-RAG/
├── config.yaml                # all tunable behaviour
├── docker-compose.yml         # pgvector + PostGIS Postgres, the API
├── docker-compose.prod.yml    # nginx, the web bundle, the API, the database
├── src/
│   ├── ingestion/   argo_netcdf.py · regions.py
│   ├── embeddings/  embedding_model.py
│   ├── semantic/    summaries.py · index.py · confidence.py   # the corpus, and the SQL context
│   ├── generation/  prompt.py · llm.py · answer_generator.py  # the summaries route
│   ├── router/      classifier.py · rewriter.py
│   ├── sqlgen/      generator.py · validator.py · executor.py · schema_context.py · scope.py
│   ├── charts/      builder.py · ocean.py
│   ├── evaluation/  metrics.py · evaluator.py · generation_metrics.py · sql_metrics.py · ab_test.py
│   ├── monitoring/  logger.py · tracing.py
│   ├── api/         main.py · schemas.py                 # FastAPI
│   └── utils/       config.py · schemas.py · db.py · cache.py · pipeline.py
├── db/              001_schema.sql … 009_drop_doc_chunks.sql
│                    schema_catalog.md · queries/         # 12 reference queries
├── scripts/         fetch_argo_index.py · load_argo_netcdf.py · load_regions.py
│                    load_drifters.py · build_semantic_index.py · check_regions.py
│                    run_ab_test.py · verify_routes.py
├── web/             React front end over the API, with the map basemaps in public/topojson/
├── data/            raw/ processed/ evaluation/ cache/
└── tests/
```

Everything routes through `RAGService` in [`src/utils/pipeline.py`](src/utils/pipeline.py). FastAPI calls that one object, and the React front end calls FastAPI, so behaviour is defined in exactly one place.

---

## Installation

```bash
git clone https://github.com/poojithareddy19/FloatChat-ARGO-RAG.git
cd FloatChat-ARGO-RAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`sentence-transformers` pulls in torch, and on Windows the default PyPI wheel bundles CUDA at roughly 2.4 GB. The configured embedding model runs on CPU, so install the CPU build first and skip the download:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Start PostgreSQL. The image is built from [`db/Dockerfile`](db/Dockerfile) rather than pulled, because this project needs two extensions and no published image carries both: pgvector for the summary embeddings, PostGIS for the basin polygons and profile positions. The schema, extensions and read-only role in `db/` are applied automatically on first boot.

```bash
docker compose up -d --build db
```

Those files run only when the volume is empty. On a database created before a migration existed, apply it by hand:

```bash
docker compose exec -T db psql -U gda -d gda < db/004_argo_qc.sql
docker compose exec -T db psql -U gda -d gda < db/007_postgis.sql
docker compose exec -T db psql -U gda -d gda < db/009_drop_doc_chunks.sql
```

`007_postgis.sql` adds the `geom` column, backfills it from the existing latitude and longitude, indexes it and creates the `regions` table. It is safe to re-run. The polygons that fill that table are not committed and not downloaded: see [Region polygons](#region-polygons). `009` removes the table behind a retrieval path over the Argo manuals that this project no longer has.

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

The rebuild is an upsert followed by a prune, so a float that has left the tables leaves the index in the same transaction. That was not always so: forty summaries of synthetic floats outlived the synthetic data by a week and were retrieved as context for floats that did not exist.

Skipping the rebuild costs both routes. The summaries route has nothing to answer from and declines; the SQL generator falls back to working from the schema alone, exactly as it did before the layer existed.

One thing the rebuild found, worth recording because it is the kind of bug a summary hides well. A biogeochemical reading is stored as its own row with no temperature on it and no core QC flag, and the summary query filtered the join on the core flag. Every one of the 49,296 oxygen readings was thrown away before it was counted, so not one of the sixteen floats that carry a sensor was described as carrying one, and "which floats measure oxygen" had nothing to find. Each BGC parameter is now counted under its own flag.

## Configuration

- **`config.yaml`** holds behaviour: embedding model, top-k, the similarity floor, confidence weights and threshold, router and SQL limits, cache TTL.
- **`.env`** holds secrets, provider selection and database URLs. Any `SECTION__KEY` variable overrides the matching `config.yaml` value, for example `SEMANTIC__TOP_K=8`.

Two database URLs are required. `DATABASE__URL` owns the schema and runs ingestion; `DATABASE__READONLY_URL` is what generated SQL executes as. Keeping them separate is the point.

## Running

Two processes: the API, and the React front end that talks to it.

```bash
uvicorn src.api.main:app --reload
```

```bash
cd web && npm install && npm run dev
```

Then open http://localhost:5173. The page is one box over both routes: a thread rather than a single question, with each turn keeping its own chart, table, SQL or citations.

`POST /ask` takes `{"question": "...", "route_override": null, "session_id": null}` and returns the answer with citations or with the generated SQL, rows and timings. `route_override` is one of `summaries`, `data` or `chart` when the caller wants to bypass the router. Passing the same `session_id` on a later request is what makes a follow up like "and in 2022?" resolvable. `GET /health` reports how many summaries are indexed, the measurement count and whether the LLM is reachable. Interactive docs at `/docs`.

The page shows what each answer depends on: for the summaries, the float or region each claim rests on with its similarity score; for data, the SQL, the rows and the chart drawn from them. If an answer looks wrong, you can see immediately whether retrieval failed, generation failed, or the query was simply right and surprising.

### The React front end

In [`web/`](web/), and a client of the API and nothing else, as one on another server would be. That is what makes it worth having: a UI written against `AskResponse` finds the places where the response is awkward to consume. A refusal renders as a result rather than an error. Charts are drawn from the Plotly figure the API returns, with Plotly loaded the first time an answer has one, so a page without charts does not pay 4.8 MB for it.

There used to be a Streamlit dashboard beside it, with evaluation and monitoring pages. It was removed so the system is reached one way. Evaluation runs from the command line, below, and monitoring reads `logs/interactions.jsonl` and `logs/traces.jsonl`.

### Tracing

`logs/interactions.jsonl` says what each question got. `logs/traces.jsonl` says where its time went. Every question is one OpenTelemetry trace: a `floatchat.answer` root with a span per step under it (`rewrite`, `route`, `retrieve`, `sql.generate`, `sql.validate`, `db.query`, `chart.ocean`, `chart.render`), and an `llm.<step>` span for each place the model is called. Model spans carry the GenAI semantic convention attributes (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`) and Ollama's own timings, so a slow answer can be told apart from a cold model: in one live run, 17.8 s of the first model call's 32 s was `ollama.load_duration_ms`.

The `trace_id` is returned by `POST /ask` and written into `interactions.jsonl`, so one id joins the response, the log line and the trace.

To view traces in a UI instead of a file, point the exporter at any OTLP/HTTP collector. Jaeger, Arize Phoenix and Langfuse all accept it:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 uvicorn src.api.main:app
```

Prompt and completion text is left off the spans unless `monitoring.tracing.capture_content` is turned on in `config.yaml`.

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

Covers the deterministic core: the summary wording, the summaries route with a fake index and a fake model, retrieval metrics, confidence, router rules, the SQL cache, the scope gate and the SQL validator. No API key, database or GPU required. The validator suite runs every query in `db/queries/` through the validator, so a reference query the validator would reject fails the build.

---

## Confidence and refusal

Confidence on the summaries path combines three interpretable retrieval signals, normalised to 0-1:

- **mean similarity** across the top-k retrieved summaries
- **support**, the fraction of top-k summaries above a similarity floor
- **spread**, derived from score variance

The combination is **multiplicative, not additive** ([`src/semantic/confidence.py`](src/semantic/confidence.py)). Mean similarity is the primary signal; support and spread are folded into a 0.5-1.0 multiplier that can penalise a strong primary signal but cannot manufacture one. Below a hard floor of `MIN_MEAN_SIM = 0.28` the score is zero regardless of the other two.

That shape was chosen after an additive version let a question with very low mean similarity clear the threshold on support and spread alone, which is the failure mode that matters: a question the corpus cannot answer getting waved through.

Below `semantic.answer_threshold` the system returns the closest summaries and an explanation instead of an answer. Before that, `semantic.min_similarity` drops any summary that is not close to the question at all, so an off-topic question usually arrives at the gate with nothing retrieved and is declined without a model call.

Retrieval signals rather than asking the LLM to self-report, because a model cannot know that the corpus lacks an answer, only that its context window does. Retrieval statistics measure corpus-to-question fit directly and cost nothing extra.

On the data path the refusal is different in kind. A lexical [scope gate](#closing-it-without-the-model) declines questions about quantities the schema does not hold before the model is called, and the validator declines SQL that reaches for a table or a join it must not.

## Evaluation

Three harnesses, because the three things that can go wrong fail differently: retrieval can miss, the prose can drift from what was retrieved, and the SQL can run and still answer the wrong question.

**Retrieval** ([`src/evaluation/evaluator.py`](src/evaluation/evaluator.py)). Recall@K, Precision@K, Hit@K, MRR, nDCG@K against `data/evaluation/summary_questions.csv`. Relevance is anchored at the **subject** level, the float or region a summary describes, so the wording of a summary can change without invalidating the labels.

#### Retrieval results

26 questions about what the archive holds, 23 answerable and 3 that the summaries cannot answer. Each answerable question names the floats or regions an answer must rest on; the unanswerable three are kept for the generation scores below and left out here, where there is nothing for retrieval to find.

| Metric | k=4 (as configured) | k=5 |
| --- | --- | --- |
| **Hit@k** | **0.957** | **0.957** |
| Recall@k | 0.819 | 0.833 |
| Precision@k | 0.337 | 0.287 |
| nDCG@k | 0.857 | 0.854 |
| MRR | 0.899 | 0.899 |

Twenty of the twenty-three are found at rank one. Named floats are exact, because an identifier in the question is looked up rather than embedded. The one miss is "which region has the longest coverage in time", which retrieves floats rather than regions: every float summary also says when it reported, and nothing in that question names a region. The two partial hits are the multi-float questions, where the gold set names four or six floats and top-k holds only some of them, which is what recall@k below hit@k means.

Reproduce with:

```bash
python -m src.evaluation.evaluator --k 5
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

`llama3.1:8b` is the default the system ships with. The first column is the current system, measured on 2026-09-26 after the [prompt stopped being cut](#the-prompt-was-being-cut). The other two are the earlier measurements, kept for comparison: `llama3.1:8b` and `qwen2.5-coder:7b`, a code specialist, on the same 67 questions with the previous prompt and Ollama's default 4,096-token window. They were run to find out whether the ceilings this benchmark keeps hitting are the model's or the prompt's.

| Metric | `llama3.1:8b`, current | `llama3.1:8b`, earlier | `qwen2.5-coder:7b`, earlier |
| ------------------------- | ----- | ----- | ----- |
| **Execution accuracy**    | **0.617** (37/60) | 0.567 (34/60) | 0.483 |
| Validation pass rate      | 1.000 | 0.983 | 0.967 |
| Execution rate            | 0.967 | 0.933 | 0.850 |
| **Correct refusal rate**  | **1.000 (7/7)** | 1.000 (7/7) | 0.857 (6/7) |
| False refusal rate        | 0.000 | 0.017 | 0.050 |
| Median latency            | 38.5 s | 31.5 s | 50.2 s |

| Bucket | Questions | `llama3.1:8b`, current | `llama3.1:8b`, earlier | `qwen2.5-coder:7b`, earlier |
| ------------ | --- | ----- | ----- | ----- |
| easy         | 8   | 1.000 | 0.875 | 0.250 |
| filter       | 8   | 0.750 | 0.875 | 0.500 |
| groupby      | 6   | 0.667 | 0.833 | 0.833 |
| join         | 8   | 0.625 | 0.500 | 0.500 |
| drifter      | 14  | 0.571 | 0.429 | 0.571 |
| qc           | 5   | 0.600 | 0.400 | 0.400 |
| bgc          | 5   | 0.400 | 0.400 | 0.600 |
| window       | 6   | 0.167 | 0.167 | 0.167 |
| unanswerable | 7   | **7/7 refused** | 7/7 refused | 6/7 refused |

Every column is one run. The two earlier columns were taken back to back on the same prompt and the same timeout, so they compare with each other; the current column differs from them in the prompt, the context window and the machine's memory, and compares with them only as a before and after. **0.567 to 0.617 is three questions, inside the swing of about a tenth that two identical runs have [shown here](#on-repeated-runs), so it does not show that the fix raised accuracy.** Five questions moved to right and two to wrong. What it does show is that the fix cost nothing, and that the number now describes the prompt the system was written to send.

#### The prompt was being cut

The SQL prompt had grown to about 4,165 tokens, and Ollama was running `llama3.1` with its default window of 4,096. A prompt over the window is not refused. Ollama logs `truncating input prompt limit=2050 prompt=4161 keep=4 new=2050` and keeps the first 4 tokens and the last 2,046, so the model saw the question, the retrieved summaries and the tail of the examples, and none of the rules or most of the schema. Nothing in the application could see it: the trace reported 2,050 input tokens, which read as a prompt smaller than expected rather than one cut in half. The mismatch between that count and 15,559 characters of prompt is what gave it away.

It also explained the timeouts. Ollama reuses the part of a prompt that matches the previous one, which is why a follow-up SQL call normally costs seconds. A cut prompt starts with its own tail, so nothing matched, every call re-read about 2,050 tokens at roughly 12 a second on this CPU, and most went past the 120 second limit into the retry.

Two changes, both in this run:

- **Every Ollama call now asks for a 6,144-token window** (`ollama.num_ctx` in `config.yaml`). The same value on every call, because Ollama reloads the model when it changes. 8,192 would not fit in this machine's memory beside the database and the embedding model.
- **The buoy schema is shown only to questions that could be about the buoys**, gated the same way as the buoy examples already were. It was a sixth of the prompt and every Argo question paid for it. A typical question is now 3,570 tokens by Ollama's own count, and the worst case, a question shown every gated example and the buoy schema with the longest summaries in the database, is 4,915, which leaves room for the output. A test holds that worst case under the configured window, so the prompt cannot quietly outgrow it again.

No prompt was cut during the current run. Whether the earlier columns were measured on cut prompts is not known: they predate the Ollama logs that survive, and the prompt then was estimated at about 3,260 tokens before the retrieved summaries were added, close enough to the limit that it cannot be ruled either way.

The two questions that went wrong went wrong the same way, and it is worth recording because it may be the fix showing through. "How many measurements are deeper than 1000 decibars" gained `qc_flag = 1 AND temperature_c IS NOT NULL`, and "number of profiles per region" gained a join to surface measurements with a quality filter. Neither question asked for either. Those are the schema's own conventions ("ignore rows where `qc_flag <> 1`", "always exclude rows with `temperature_c IS NULL`"), which a cut prompt never showed the model, applied where they do not belong. That reading is plausible rather than proven, and a single run cannot establish it.

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

For most of this project's life the gold set had 52 questions and not one was about the drifting buoys. The platform was reachable, exported, charted and refused against, and none of it was measured. 14 answerable questions and one unanswerable were added. The bucket scored **0.429**, second worst on the board, in the earlier run this section analyses; the current run scores it 0.571, with the two SVPB questions below among the ones that moved.

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

Both columns here are the earlier run, on the same prompt, which is what makes them comparable.

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

Validation pass rate is 1.000 while accuracy is 0.617. Every generated query was well formed and safe, and nearly four in ten still answered the wrong question. That gap is the entire argument for measuring results rather than liveness.

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

Retrieval finding the right float says nothing about the sentence handed to the user, which is the half the project's central claim rests on: "being confidently wrong is worse than saying nothing" is a statement about generated text.

[`src/evaluation/generation_metrics.py`](src/evaluation/generation_metrics.py) scores it, in two groups.

**Deterministic, no model call.** These cost nothing and cannot drift:

- `answered_rate` - how often the system answers rather than declining. A system that refuses everything scores perfectly on faithfulness, so no quality number means anything without this beside it.
- `citation_accuracy` - of the answers given, how many were written from the float or region the gold set names. An answer about the Arabian Sea built from the Bay of Bengal's summary is not grounded, and no model is needed to check that.
- `lexical_support` - the share of the answer's content words that appear in the summaries it was given. A **proxy** rather than a measure, and reported as one: paraphrase scores low and fluent copying scores high. It is a floor. An answer near zero is not reading its sources.

**Judged, one model call per question, behind `--judge`:** faithfulness, answer relevance and correctness against the gold answer.

```bash
python -m src.evaluation.generation_metrics            # deterministic only
python -m src.evaluation.generation_metrics --judge    # adds the graded three
```

#### Generation results

The 26 questions above through the whole summaries route, `llama3.1:8b` writing the answers, one run:

| Metric | Value |
| --- | --- |
| Answered | 0.885 (23 of 26) |
| Citation accuracy, of those answered | **0.957** (22 of 23) |
| Lexical support, of those answered | 0.770 |

The three declines are the interesting rows, because they are three different mechanisms doing their job:

- **"What is the wind speed over the Arabian Sea?"** passed the retrieval gate at 79% confidence, because the Arabian Sea summary is close to any question naming the Arabian Sea, and was then declined by the model with `INSUFFICIENT_CONTEXT`. That is the second guard catching what the first let through.
- **"Tell me about float 2900007"** names a float that does not exist. Nothing is found by identifier, the nearest summaries are other floats at 84% confidence, and the model declined rather than describing one of them as the float asked for.
- **"Which region has the longest coverage in time?"** was declined at the gate with nothing retrieved. Every float summary also says when it reported, and nothing in the question names a region, so the nearest summaries were floats and below the similarity floor. A correct decline for the wrong reason: the answer was in the index and retrieval did not reach it.

The one miscitation is the failure worth reading. **"Describe the drifting buoys in the Bay of Bengal"** is unanswerable, because the summaries cover Argo floats and say nothing about buoys. The model answered anyway, from the Bay of Bengal summary and three float summaries, and called the floats drifting buoys, adding a note that the term "is not used in the summaries" but the floats "are likely what is meant." That is exactly the confidently helpful answer the guards exist to prevent, and both guards let it through: retrieval scored 81% because the region is real, and the model chose to reinterpret rather than abstain. It is recorded here rather than fixed with a prompt line, because a prompt line that suppresses this sentence would be tuned to one question.

The honest limit is written into the module: the judge is the same local model that wrote the answer, so it is a poor judge of its own output and shares its own blind spots. Read `citation_accuracy` first, which nothing can talk its way out of, and treat a judged score as a smoke alarm rather than a measurement. A judge that cannot be reached is recorded as unknown and left out of the mean, so an outage cannot look like a quality drop.

### What these numbers do not tell you

- **One prompt, one schema, two models.** Nothing here separates what these models cannot do from what no model could do with this prompt.
- **Execution accuracy has false positives.** A wrong query can collapse to the right number when the data happens to cooperate, and the comparison cannot tell that apart from understanding.
- **The BGC sample is narrow.** Sixteen floats carry a biogeochemical sensor and none carries a NITRATE sensor, so that column is empty and the `bgc` bucket tests oxygen, chlorophyll, pH and backscatter only.
- **Profile counts are shaped by the download, not by the ocean.** The sample takes a bounded number of cycles per float, so "which float reported most" is answerable but is a fact about what was fetched rather than about the fleet.
- **The summary gold set is small and self-written.** 26 questions, written by the same person who wrote the summaries, against a corpus of 83 sentences. Hit@5 on that says retrieval works on this corpus; it does not say what happens with thousands of floats.
- **The drifter bucket is new and shallow.** 14 questions is enough to show the platform is the weakest path and not enough to characterise it. It covers one year of buoys and three hull types, and no question tests the trajectory or distance queries the platform also supports.
- **Single runs.** Every SQL figure above except the refusal rates comes from one run, and two identical back-to-back runs disagreed on four of 46 questions. Differences smaller than about a tenth are not resolvable without `--runs N`.

There is also a paired A/B harness ([`src/evaluation/ab_test.py`](src/evaluation/ab_test.py)) with a sample-size calculation, for comparing two pipeline configurations without reading noise as improvement.

---

## Design decisions

| Decision                                 | Rationale                                                                                                     |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Summaries as the retrieval corpus         | The problem statement asks for retrieval over metadata and summaries of the data, not over documents about it |
| One table, two readers                    | The same summaries answer descriptive questions and ground the SQL generator, so there is one thing to rebuild |
| Named floats looked up, not embedded      | Float summaries differ only by a long number, which embeddings rank worst; a question that names a float gets that float |
| No vector index on the summaries table    | Eighty-three rows scan exactly in under a millisecond; an approximate index over so few rows loses recall for nothing |
| Router with rules before the model        | Most questions are unambiguous. Paying for a model call on those is latency and cost for nothing               |
| Fallback route is the summaries           | It declines when nothing is close, so an unplaced question ends in a refusal rather than in guessed SQL       |
| Read-only role for generated SQL          | A validator bug becomes a failed query rather than a data-loss incident                                        |
| Column check against `information_schema` | Read from the live database, not the markdown catalog, so it cannot drift away from the real tables            |
| Context is named, never quoted            | The SQL prompt forbids copying a number out of a summary, so every value in a data answer comes from the query |
| SQL cache keyed on prompt version         | Re-running a demo should be fast, but a changed prompt must invalidate everything cached under the old one. The retrieved context is part of that key |
| Multiplicative confidence                 | Secondary signals should modulate the primary one, never rescue it                                             |
| Retrieval-based confidence                | LLMs are structurally uncalibrated about corpus coverage; retrieval signals measure fit directly and are free  |
| Rules that hold are code                  | The scope gate and the platform-join rule live in front of the model, after prompt wording failed three times |
| Temperature 0                             | The task is faithful extraction, not creative writing. Also makes evaluation runs comparable                   |
| Subject-level evaluation anchors          | Summary wording can be tuned without destroying the ground truth set                                           |
| One service facade                        | FastAPI is the only caller of `RAGService`, so behaviour is defined once                                       |

## Why RAG rather than fine-tuning

RAG keeps knowledge external and updatable: re-index instead of re-train. When a new float lands, one script rebuilds its summary and the answer changes the same minute. It makes answers attributable, so a reader can check the float or region an answer rests on. Fine-tuning changes style and format, not the facts a system can cite.

For the measurements the argument is stronger still. The answer to "average surface temperature in the Arabian Sea in 2021" is a number in a table that changes as floats report. No amount of training bakes that in correctly; a query reads it.

## Limitations

**Data.** The database holds real Argo profiles: 80 floats, 1,099 profiles and 175,364 measurements from the Indian Ocean. Sixteen of those floats carry a biogeochemical sensor, which populates oxygen, chlorophyll, pH and backscatter, but none carries a NITRATE sensor, so that one column is still empty. The sample takes a bounded number of cycles per float rather than every cycle, so per-float profile counts reflect the download rather than the fleet. Region comes from IHO basin polygons in PostGIS, with the old latitude and longitude inequalities kept as the fallback for a position inside none of the three loaded basins, which is 3% of profiles.

**Conversation.** A follow up is rewritten into a standalone question before routing, which keeps the router, the SQL generator and the cache stateless and keeps the cache keyed on what was actually asked. The history behind that rewrite is not persisted: the API holds it in a process-local dict, so it does not survive a restart and does not work across workers, and the React page generates a new session id on every load, so a refresh starts a new conversation. Moving it to Redis or a sessions table is the fix and has not been done. A follow up whose history is lost degrades to being answered as a standalone question rather than to an error, and a rewrite that goes wrong is visible: the page shows what the question was answered as, and both versions are written to the log.

**Architecture.** A question is answered from one route. One that needs a description and a computation, "which floats measure oxygen and how many readings do they hold", is routed to whichever half the wording favours and answered by that half alone. Summaries are rebuilt only when the script is run, so they drift from the tables between loads. Summarising is per float and per region; a database with thousands of floats will want coarser grouping than one summary each, and the exact scan that is right for 83 rows will want an index.

**Retrieval and generation.** Retrieval over the summaries reaches hit@5 of 0.957 on 23 questions, but that gold set was written by the same person who wrote the summaries, which is the honest limit on what it shows. A summary carries statistics computed at index time, and a question that asks for a number under a condition the summaries do not carry has to reach the SQL path through the router, which decides by wording. Confidence is a heuristic over retrieval signals, not a calibrated probability, so the threshold needs tuning against your own gold set. Questions are expected in English: the embeddings, the prompts and the SQL cache are English-only.

**Charts.** The map basemap is served locally. Plotly fetches the land and coastlines of a geo plot as TopoJSON at render time and defaults to `cdn.plot.ly`, which made the one networked dependency in an otherwise offline app a map that came up empty with no error to explain it. The files are committed under `web/public/topojson/` and Plotly is pointed there through `topojsonURL`. Trajectories, depth profiles, depth-time sections and T-S diagrams are drawn as interactive Plotly figures, dispatched on column names because latitude and longitude are two ordinary floats to a dtype check. Everything else falls through to the matplotlib line and bar builder, and a PNG is produced in every case so an image client keeps working. The dispatch is first-match, so a result carrying positions is always drawn as a track even when it also carries measurements.

**Evaluation.** Text-to-SQL scores 0.617 execution accuracy over 67 questions on the shipped model. The earlier measurements, 0.567 on the same model and 0.483 on a second one, were taken on a previous prompt that may have been cut by the model's context window, and the second model has not been re-run since. Complex queries are much worse than either average suggests: the window bucket is 1 in 6 on **both** models, and worked examples, a repair attempt and a change of model have each failed to move it. Figures are single runs, and two back-to-back runs of an identical configuration disagreed on four of 46 questions, so none is precise to better than about a tenth. The repair loop recovered 2 queries of 4 attempts in the current run and cannot touch the 21 questions that fail by returning the wrong rows without an error. The seafloor fabrication and the current at 1000 decibars are both refused now, but by a lexical gate in front of the model rather than by anything the model learned, so the gold set measures the gate on those six questions and not the generator. A paraphrase the gate does not carry reaches the model exactly as before. Both columns come from one gold set written by the same person who wrote the schema, which is a real limit on what they can show.

## Roadmap

Done:

- [x] NetCDF ingestion with `xarray`, carrying real QC flags through
- [x] BGC-ARGO parameters in the schema, parser and catalog
- [x] Real Argo GDAC profiles, replacing the synthetic set: 80 floats, 1,099 profiles, 175,364 measurements
- [x] Float and region metadata summarised into pgvector, so semantic retrieval informs SQL generation
- [x] The summaries as a route of their own, answering what the archive holds with the float or region cited and a confidence gate in front of the model
- [x] Named floats looked up literally, after embeddings ranked an unrelated float within 0.002 of the one asked for
- [x] Summaries that name their sensors, after the core QC filter silently discarded every BGC reading
- [x] The index pruned on rebuild, after forty summaries of deleted floats were retrieved as context
- [x] Repeat runs, so a difference can be told from noise
- [x] Trajectory maps, depth profiles, depth-time sections and T-S diagrams
- [x] Conversational follow-ups, by rewriting a follow up into a standalone question
- [x] PostGIS geometry on profiles, with a spatial region lookup and `ST_DWithin` distance queries
- [x] IHO basin polygons loaded and the existing profiles backfilled, moving 128 of 1,099 into a different basin
- [x] A second in-situ platform: 187 drifting buoys, 139,971 fixes, sharing the charts and regions with no code changed
- [x] A conversational thread rather than a single question box, with each turn keeping its own chart, table and SQL
- [x] Relative periods anchored to the archive: the SQL prompt is told where each platform's data starts and ends, read from the tables
- [x] One trace per cast in profile plots, drawn whenever the rows are profiles rather than only when the question says "plot"
- [x] Few-shot window-function examples, measured: they did not move the bucket
- [x] The refusal the drifters reopened, closed by a scope gate in front of the model rather than a fourth rewording
- [x] ERDDAP's numeric fill value dropped at parse time, at the database and in the catalog, after 189 fixes stored `-999999` as a surface current
- [x] The drifting buoy examples shown only to questions that could be about the buoys, so the other 51 do not carry them
- [x] Drifter questions in the gold set, 14 answerable and one not, which found the buoy path is the weakest measured
- [x] Generation scored, not just retrieval: citation accuracy and answer rate deterministically, faithfulness and relevance behind a judge flag
- [x] A repair attempt on a query that failed, given the error it failed with, with refusals never repaired, and a guard after the repair answered a depth question with a temperature
- [x] Cross-platform joins rejected in the validator, including the trivial-subquery form a model found to get round the first version
- [x] A React front end over the API, which is the second client the response contract needed
- [x] The Docker stage, verified: both images build, the production compose comes up, and a request traverses nginx to the API to the database on real data. CI builds both images on every push and publishes them to GHCR on main
- [x] GitHub Actions running the suite and the linter on 3.11 and 3.12
- [x] Both models on the same 67 questions, which retracted an earlier claim: qwen's advantage on the window bucket was noise, and both score 0.167 there
- [x] Scoped to the problem statement: the manual retrieval path, the MCP server, the file export, the combined route and the multilingual translator were removed, with their tests and dependencies
- [x] OpenTelemetry tracing: a span per pipeline step and per model call, with token usage and cold-load time, joined to the interaction log by `trace_id`
- [x] The SQL prompt stopped being cut by the model's context window: a 6,144-token window on every call, the buoy schema gated to buoy questions, and a test that holds the worst case under the window. Re-measured at 0.617
- [x] Fewer waits per question: the model is held for 30 minutes after a call instead of Ollama's 5, both models load in the background at startup, and the router's rules now place all 67 SQL gold questions, 13 of which used to cost a 10 to 20 second routing call

Not done, honestly:

- [ ] Persist conversation history, which currently dies with the process
- [ ] Re-run with the fixed prompt using `--runs 3`, and re-run `qwen2.5-coder:7b`, so the current figure has a spread and the model comparison is on the current prompt. About three hours for llama alone on this machine, and only if nothing else is using its memory
- [ ] Load the data the problem statement's own examples ask for. There are no profiles in the equatorial band in March 2023, and no BGC readings in the Arabian Sea in the last six months of the archive, so both examples return correct, empty queries
- [ ] Rename the Southern Indian Ocean region, which is the fallback for anything outside the named basins and so includes floats north of the equator
- [ ] A judge that is not the model under test, which is the honest limit of the generation scores
- [ ] A larger model on the same 67 questions. Three prompt attempts, a repair loop and a second 7B model have all left the window bucket at 0.167, so the next honest experiment is more capacity rather than better wording
- [ ] A cloud host. Deliberately not: the images are published and the arrangement is proven locally, and putting it on a paid instance is a decision rather than a task
- [ ] A nitrate-carrying BGC float, since none of the sixteen sampled BGC floats has that sensor
- [ ] Coarser summaries, by project or by year, before the fleet grows past what one sentence per float can hold
