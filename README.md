# Production RAG Knowledge Assistant

Grounded question-answering over your own documents, with citations, confidence estimation, graceful refusal, evaluation and monitoring built in. Designed as a production-style reference implementation rather than a "chat with PDF" demo.

Every answer is attributed to a specific document, page and passage. When the retrieved evidence does not support an answer, the system declines instead of guessing.

> **Build status.** The retrieval and generation pipeline, Streamlit dashboard, structured logging, monitoring analytics, and the retrieval-metrics evaluation harness are implemented and unit-tested. LLM-judged generation metrics (faithfulness, groundedness, answer relevance) are scaffolded and flagged as the next milestone. See [Roadmap](#roadmap).

---

## Problem

Organisations keep answers buried in policies, handbooks, contracts, SOPs and manuals. Keyword search cannot reach them, and a general LLM asked about private data will produce a fluent, plausible, wrong answer with no way for the reader to check it.

In domains like HR, finance and compliance, being confidently wrong is worse than saying nothing. This system is built around that constraint:

- retrieves only from documents you upload
- cites the exact document, page, chunk and passage behind every answer
- attaches a confidence score derived from retrieval signals, not from asking the model how sure it is
- declines to answer when evidence is insufficient, via two independent guards
- logs every interaction as structured JSON for monitoring and evaluation

## Architecture

```
question
   │
   ▼
┌──────────────┐   ┌───────────────┐   ┌──────────────┐   ┌──────────────┐
│  Embed query │──▶│ FAISS retrieve│──▶│  Confidence  │──▶│  gate < thr? │
└──────────────┘   │   (top-k)     │   │  estimation  │   └──────┬───────┘
                   └───────────────┘   └──────────────┘          │
                                                        decline ◀─┘ (fallback)
                                                            │
                                                        proceed
                                                            ▼
                                              ┌───────────────────────────┐
                                              │  Grounded LLM generation  │
                                              │ (citation-forcing prompt) │
                                              └────────────┬──────────────┘
                                                           ▼
                                    answer + citations + confidence + JSON log
```

### Two independent hallucination guards

1. **Retrieval-side.** A configurable confidence threshold over retrieval signals, evaluated before the LLM is called. Unanswerable questions are refused without spending a token.
2. **Generation-side.** The prompt requires the model to emit `INSUFFICIENT_CONTEXT` when the retrieved passages do not support an answer, checked after generation.

Two guards rather than one because they fail independently. A single guard is a single point of failure, and the two catch different failure modes: the first catches "nothing relevant was retrieved," the second catches "something was retrieved but it does not actually answer the question."

Full reasoning for each choice is in [`docs/design_decisions.md`](docs/design_decisions.md).

---

## Technology stack

| Concern      | Choice                                   | Notes                  |
| ------------ | ---------------------------------------- | ---------------------- |
| Language     | Python 3.12+                             |                        |
| Chunking     | `langchain-text-splitters`               | isolated to one file   |
| Embeddings   | `sentence-transformers` (BAAI/bge-small) | swappable via config   |
| Vector store | FAISS (`IndexFlatIP`, cosine)            | Chroma-ready interface |
| LLM          | OpenAI / Gemini / Ollama                 | selected via `.env`    |
| UI           | Streamlit (7-page dashboard)             |                        |
| Evaluation   | custom retrieval metrics (+ Ragas path)  |                        |
| Logging      | structured JSONL + `logging`             |                        |
| Config       | `config.yaml` + environment variables    |                        |

## Project structure

```
RAG_Assistant/
├── app.py                     # Streamlit dashboard (presentation only)
├── config.yaml                # all tunable behaviour
├── requirements.txt
├── src/
│   ├── ingestion/   loader.py · chunker.py
│   ├── embeddings/  embedding_model.py
│   ├── vectorstore/ vectordb.py            # FAISS + metadata + persistence
│   ├── retrieval/   retriever.py · reranker.py · confidence.py
│   ├── generation/  prompt.py · llm.py · answer_generator.py
│   ├── evaluation/  metrics.py · evaluator.py
│   ├── monitoring/  logger.py · analytics.py
│   └── utils/       config.py · schemas.py · pipeline.py   # service facade
├── data/            raw/ processed/ vector_store/ evaluation/
├── tests/           test_core.py
└── docs/            design_decisions.md
```

Seven decoupled packages communicating through typed dataclass contracts, with a `RAGService` facade that keeps the Streamlit layer free of business logic.

---

## Installation

```bash
git clone https://github.com/poojithareddy19/RAG_Assistant.git
cd RAG_Assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

- **`config.yaml`** controls behaviour: chunk size and overlap, embedding model, top-k, confidence weights and threshold, provider and model defaults.
- **Environment variables** hold secrets and provider selection. Any `SECTION__KEY` variable overrides the matching `config.yaml` value, for example `RETRIEVAL__TOP_K=8`.

Create a `.env` file in the project root:

```bash
GENERATION__PROVIDER=openai      # openai | gemini | ollama
GENERATION__MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

Ollama requires no key. Run a local model and set `GENERATION__PROVIDER=ollama`.

## Running

```bash
streamlit run app.py
```

Dashboard pages: Home, Upload, Knowledge Base, Ask, Evaluation, Monitoring, Settings.

Upload documents (PDF, DOCX, TXT, MD), then ask questions. Each answer shows its citations, a confidence meter, the retrieved chunks with similarity scores, and the exact prompt sent to the model. That last item matters: if an answer looks wrong, you can see immediately whether the retrieval failed or the generation did.

## Testing

```bash
pytest -q
```

Covers the deterministic core: metrics, chunking, confidence estimation and FAISS operations. No API key or GPU required, since the FAISS test uses a fake embedder.

---

## Confidence and refusal

Confidence combines three interpretable retrieval signals, weighted via config and normalised to 0-1:

- **mean similarity** across the top-k retrieved chunks
- **support**, the fraction of top-k chunks above a similarity floor
- **spread**, derived from score variance

Below `confidence.answer_threshold` the system returns the closest passages plus an explanation rather than an answer.

Retrieval signals rather than asking the LLM to self-report, because a model has no view of what is in the corpus. It cannot know that the corpus lacks an answer, only that its context window does. Retrieval statistics measure corpus-to-question fit directly, and cost nothing extra.

**Known flaw in the current formula.** It is additive, so support and spread can together contribute up to 0.40. With the threshold at 0.45, a question with very low mean similarity can still clear the gate on those two components alone. A multiplicative form, or a hard floor on mean similarity, would fix this.

## Evaluation

A gold dataset lives at `data/evaluation/questions.csv` with the schema:

```
question, expected_answer, relevant_documents, relevant_pages, ground_truth_passage
```

Relevance is anchored at **document and page** level rather than chunk ID. Chunk IDs renumber whenever chunk size changes, which would invalidate the entire gold set every time a chunking parameter is tuned. Anchoring to document and page keeps labels durable across re-indexing experiments.

Implemented and unit-tested: Recall@K, Precision@K, Hit@K, MRR, nDCG@K.

```python
from src.utils.pipeline import RAGService
from src.evaluation.evaluator import evaluate_retrieval

svc = RAGService()
print(evaluate_retrieval(svc.retriever, "data/evaluation/questions.csv", k=5)["aggregate"])
```

> **The shipped `questions.csv` is a five-row schema template, not a benchmark.** It references sample documents that are not committed to this repo. To run a meaningful evaluation, index your own corpus and expand the file to 40-100 labelled questions. Results will be published here once run against a real corpus.

---

## Design decisions

| Decision                              | Rationale                                                                                                  |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Recursive splitting over fixed-length | Preserves paragraph and sentence boundaries; fixed slicing cuts mid-word and degrades embedding quality       |
| Chunking isolated per page            | A chunk straddling two pages makes its citation ambiguous. Per-page chunking keeps "page 42" honest.          |
| FAISS `IndexFlatIP` over HNSW         | Exact search. Approximation error would contaminate the retrieval metrics used to tune chunking and embedding |
| Retrieval-based confidence            | LLMs are structurally uncalibrated about corpus coverage; retrieval signals are free and measure fit directly |
| Temperature 0                         | The task is faithful extraction, not creative writing. Also makes evaluation runs comparable.                 |
| Doc/page evaluation anchors           | Chunking parameters can be tuned without destroying the ground truth set                                      |
| Provider abstraction (`BaseLLM`)      | OpenAI, Gemini and Ollama differ in response shape; one interface keeps that out of the pipeline              |
| Guard 1 before the LLM call           | Refusing unanswerable questions at the retrieval stage saves both latency and API cost                        |

## Why RAG rather than fine-tuning

RAG keeps knowledge external and updatable: re-index instead of re-training. It makes answers attributable, so a reader can check the source. And it avoids baking sensitive internal documents into model weights. Fine-tuning changes style and format, not the facts a system can cite.

## Limitations

- No models are trained or fine-tuned here; the embedding model and LLM are both used off the shelf.
- Extraction quality depends on the source PDF. Scanned or image-only PDFs need OCR, which is not yet wired in.
- Confidence is a heuristic over retrieval signals, not a calibrated probability. Tune the threshold against your own gold set.
- The confidence formula is additive; see the flaw noted above.
- `IndexFlatIP` is exact but linear in corpus size. Very large corpora need an ANN index.
- Stateless: no conversational memory between questions.
- No individual document deletion; removing a document requires an index rebuild.
- Enabling the cross-encoder reranker reorders results without updating the confidence inputs, so confidence scores become unreliable when it is on. It is disabled by default for this reason.

## Roadmap

- [ ] LLM-judge / Ragas generation metrics (faithfulness, groundedness, answer relevance, context precision and recall)
- [ ] Published retrieval evaluation results against a real corpus
- [ ] Hybrid search (BM25 + dense) and reranker enabled by default with corrected confidence handling
- [ ] Multiplicative confidence formula with a mean-similarity floor
- [ ] Remaining docs: `architecture.md`, `file_explanations.md`, `evaluation_report.md`, `monitoring.md`
- [ ] Commit a `.env.example` and sample corpus so the repo is runnable on clone
- [ ] FastAPI service layer, Dockerfile, GitHub Actions CI
- [ ] Session-based conversation memory and chat export
- [ ] OCR path for scanned PDFs
