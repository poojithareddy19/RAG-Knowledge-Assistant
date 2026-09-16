# Grounded Data Assistant

Natural-language questions answered from two kinds of source, with the evidence shown either way: uploaded documents answered by retrieval with citations, and ARGO ocean float measurements answered by generated SQL that you can read before you believe the numbers.

Built around one constraint: being confidently wrong is worse than saying nothing. Document answers cite the document, page and passage behind every claim and decline when the evidence is thin. Data answers show the exact SQL that produced the table, and that SQL runs as a read-only database user after passing a validator.

> **Build status.** Working end to end: NetCDF ingestion of ARGO profiles with core and biogeochemical parameters and per-parameter QC, a semantic layer that tells the SQL generator what the database actually holds, document RAG with citations and refusal, text-to-SQL with a four-layer safety path, a query router, automatic charts, a Streamlit dashboard, a FastAPI service, and a 52-question text-to-SQL benchmark scoring [0.739 execution accuracy](#results). The largest remaining gaps are that a single question is answered from documents or data but never both, and that no BGC-ARGO float has been loaded yet, so those columns are all NULL. See [Limitations](#limitations) and [Roadmap](#roadmap).

---

## Problem

Two versions of the same problem, which is why they share a codebase.

Organisations keep answers buried in policies, handbooks and SOPs. Keyword search cannot reach them, and a general LLM asked about private data produces a fluent, plausible, wrong answer with no way to check it.

Scientific agencies keep answers buried in numerical archives. ARGO float data is fully open, but reaching it means knowing the schema, the QC conventions and the query language. A researcher, student or policymaker who can describe what they want in a sentence cannot get it.

RAG solves the first. Text-to-SQL solves the second. Both fail the same way, by producing something confident and unverifiable, so both are built here with the evidence surfaced and a refusal path.

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

They live in their own table rather than alongside document chunks, because a question about a policy document must not retrieve a float and a question about a float must not retrieve a policy document.

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
| Charts          | matplotlib, chosen from result shape      | PNG, served to both UIs           |
| API             | FastAPI + Pydantic                        | same service object as Streamlit  |
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
├── scripts/         load_argo_netcdf.py · load_argo.py
│                    build_semantic_index.py
│                    make_sample_data.py · run_ab_test.py
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

Start PostgreSQL with pgvector. The schema, extension and read-only role in `db/` are applied automatically on first boot.

```bash
docker compose up -d db
```

Those files run only when the volume is empty. On a database created before the QC columns existed, apply the migration by hand:

```bash
docker compose exec -T db psql -U gda -d gda < db/004_argo_qc.sql
```

The default provider is Ollama, which needs no API key:

```bash
ollama pull llama3.1
```

### Load ocean data

ARGO is distributed as NetCDF, which is the loader to prefer. It takes files or directories and searches directories recursively:

```bash
python scripts/load_argo_netcdf.py path/to/dac/incois/
```

Re-running is safe: profiles already present are left alone rather than duplicated, so an interrupted load can simply be repeated.

Two ARGO conventions are handled in [`src/ingestion/argo_netcdf.py`](src/ingestion/argo_netcdf.py) rather than left to the caller. Profiles in delayed or adjusted mode are read from their `*_ADJUSTED` variables, because in those modes the raw variables are kept only for provenance and reading them anyway is the most common way to get ARGO wrong. And each parameter's QC flag is stored separately, because a level can have good temperature and bad salinity.

There is also an ERDDAP tabledap CSV loader, kept because a CSV export is often the quickest way to get a region loaded:

```bash
python scripts/load_argo.py path/to/argo_export.csv
```

Or generate synthetic ARGO-shaped data with a planted 0.02 °C/year warming signal, which is useful for checking that the trend tests detect what they should:

```bash
python scripts/make_sample_data.py
```

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

`POST /ask` takes `{"question": "...", "route_override": null}` and returns the answer with citations or with the generated SQL, rows and timings. `GET /health` reports indexed documents, measurement count and whether the LLM is reachable. Interactive docs at `/docs`.

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

## Evaluation

Two harnesses, because the two paths fail differently.

**Retrieval** ([`src/evaluation/evaluator.py`](src/evaluation/evaluator.py)). Recall@K, Precision@K, Hit@K, MRR, nDCG@K against `data/evaluation/questions.csv`. Relevance is anchored at **document and page** level rather than chunk ID, because chunk IDs renumber whenever chunk size changes, which would invalidate the entire gold set every time a chunking parameter is tuned.

**Text-to-SQL** ([`src/evaluation/sql_metrics.py`](src/evaluation/sql_metrics.py)). The metric is **execution accuracy**: run the generated query and a hand-written reference query and compare their result sets. Whether a query parses says nothing about whether it answered the question.

Comparing results rather than SQL text is deliberate, because there are many correct ways to write the same query. In one run the model answered "number of profiles per region" with a join the reference did not have, and was right.

```python
from src.evaluation.sql_metrics import evaluate_file

summary, by_bucket, detail = evaluate_file()
```

### Results

52 questions, `llama3.1:8b` via Ollama, against 40 synthetic floats and 233,600 measurements. Single run.

| Metric | Value |
| ------------------------- | ----- |
| **Execution accuracy**    | **0.739** |
| Validation pass rate      | 1.000 |
| Execution rate            | 0.957 |
| Correct refusal rate      | 0.833 |
| False refusal rate        | 0.000 |
| Median latency            | 26.5 s |
| p95 latency               | 47.3 s |

| Bucket | Questions | Execution accuracy |
| ------------ | --- | ----- |
| easy         | 8   | 1.000 |
| filter       | 8   | 1.000 |
| bgc          | 5   | 1.000 |
| qc           | 5   | 0.800 |
| join         | 8   | 0.625 |
| groupby      | 6   | 0.500 |
| window       | 6   | 0.167 |
| unanswerable | 6   | 5/6 refused |

Accuracy falls as query complexity rises, and the fall is steep. Single-table aggregates and filters are perfect. **Window functions are 1 of 6, the same in three separate runs**, which is the most stable finding here: this model cannot reliably write `lag`, ranking or first/last-value SQL against this schema. That is a model-capability ceiling, not a prompt that needs another sentence.

Validation pass rate is 1.000 while accuracy is 0.739. Every generated query was well formed, safe and ran, and a quarter still answered the wrong question. That gap is the entire argument for measuring results instead of liveness.

**The one refusal that failed is the most serious result here.** Asked for the seafloor depth beneath each float, the model wrote a confident query against a schema that has no bathymetry. Five of six unanswerable questions were refused, including pH and dissolved oxygen before BGC existed, but a single fabricated query matters more than a point of accuracy in a system whose premise is declining rather than guessing.

A previous run of the 47-question set scored 0.707. The two are not comparable: the gold set changed, two unwinnable questions were replaced and a `bgc` bucket was added.

### What these numbers do not tell you

- **Single run per configuration.** Temperature is 0, but any prompt change reshuffles outputs, so individual question flips are weak evidence. Treat differences under about five points as noise until someone runs it three times.
- **Execution accuracy has false positives.** "Number of profiles per year" once scored correct with a query that joined `measurements` and filtered `pressure_dbar < 10`. Every profile happens to have exactly one measurement above 10 decibars in this dataset, so a wrong query collapsed to the right number. A coincidence in the data can mark a wrong query right.
- **Synthetic data flatters some questions and breaks others.** Every float has exactly 730 profiles, so any "which float has the most" question is a 40-way tie with no single right answer. Two such questions had to be replaced. Real ARGO data would not have this shape.
- **The `bgc` bucket scores 1.000 on empty columns.** No BGC-ARGO float has been loaded, so every biogeochemical column is NULL and the reference and generated queries agree by both finding nothing. That result shows the model targets the right columns and stops refusing; it does not show the arithmetic is right.
- **The document retrieval gold set is still a template.** `questions.csv` is five rows referencing documents not committed here. Only the SQL side has real numbers.

## Beyond retrieval

Three analyses that run against the measurements table directly, included because "the model wrote some SQL" is not by itself evidence that the data supports a claim:

- **Trend tests** ([`src/analytics/trend_test.py`](src/analytics/trend_test.py)) — regression and Mann-Kendall on regional surface series, reporting significance rather than a slope with no error bar.
- **Clustering** ([`src/analytics/clustering.py`](src/analytics/clustering.py)) — k-means over profile features into data-driven water-mass groups, with silhouette scoring.
- **Quantile forecasting** ([`src/analytics/forecast.py`](src/analytics/forecast.py)) — LightGBM quantile regression on monthly regional means, so the output is an interval.

There is also a paired A/B harness ([`src/evaluation/ab_test.py`](src/evaluation/ab_test.py)) with a sample-size calculation, for comparing two pipeline configurations without reading noise as improvement.

---

## Design decisions

| Decision                                 | Rationale                                                                                                     |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Router with rules before the model        | Most questions are unambiguous. Paying for a model call on those is latency and cost for nothing               |
| Read-only role for generated SQL          | A validator bug becomes a failed query rather than a data-loss incident                                        |
| Column check against `information_schema` | Read from the live database, not the markdown catalog, so it cannot drift away from the real tables            |
| Summaries in their own table              | A question about a float must not retrieve a policy document, and the reverse                                  |
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

**Data.** The schema carries core CTD and five biogeochemical parameters, but the synthetic dataset has no BGC readings, so every BGC column is currently NULL and the `bgc` eval questions pass by both sides correctly finding nothing. Loading a real BGC-ARGO float is what would exercise them. Real QC flags arrive only through the NetCDF loader; the CSV loader still writes `qc_flag = 1` for every row, so a database built from CSV has a QC column that means nothing. Region is assigned by three latitude/longitude inequalities in [`src/ingestion/regions.py`](src/ingestion/regions.py), which is coarse for a field that nearly every query groups by.

**Architecture.** Semantic retrieval now feeds SQL generation, but a single question still gets a single answer: the router picks documents or data, so a question genuinely needing both sources gets one. Summaries are rebuilt only when the script is run, so they drift from the tables between loads. Summarising is per float and per region; a database with thousands of floats will want coarser grouping than one summary each.

**Retrieval and generation.** No models are trained here. Extraction quality depends on the source PDF; scanned PDFs need OCR, which is not wired in. Confidence is a heuristic over retrieval signals, not a calibrated probability, so the threshold needs tuning against your own gold set. English-only embeddings. Stateless: no conversational memory. No individual document deletion.

**Charts.** Line and bar only, chosen from result shape. No trajectory map and no depth-profile plot, which are the two views this domain actually expects.

**Evaluation.** Text-to-SQL scores 0.739 execution accuracy, and complex queries are much worse than that average suggests: window functions are 1 of 6. One unanswerable question in six still gets a fabricated query rather than a refusal. Every number comes from a single run against synthetic data. The document retrieval side has no published numbers at all.

## Roadmap

- [x] NetCDF ingestion with `xarray`, carrying real QC flags through
- [x] BGC-ARGO parameters in the schema, parser and catalog
- [ ] Load a real BGC float so the bgc questions test more than a shared NULL
- [ ] Real QC flags on the CSV loader too
- [x] Float and region metadata summarised into pgvector, so semantic retrieval informs SQL generation
- [ ] Answer a single question from documents and data together
- [x] Expand the SQL gold set to 47 questions with reference queries, publish the numbers
- [ ] Repeat runs so a difference can be told from noise
- [ ] Close the last refusal gap: a bathymetry question still gets a fabricated query
- [ ] Few-shot window-function examples, or a stronger model, for the bucket scoring 0.167
- [ ] A document retrieval gold set that is not a five-row template
- [ ] Trajectory map and depth-profile plots
- [ ] Proper region assignment instead of lat/lon inequalities
- [ ] LLM-judge / Ragas generation metrics (faithfulness, groundedness, answer relevance)
- [ ] Hybrid search (BM25 + dense)
- [ ] GitHub Actions CI
- [ ] Session-based conversation memory and chat export
- [ ] Multilingual query support
- [ ] OCR path for scanned PDFs
