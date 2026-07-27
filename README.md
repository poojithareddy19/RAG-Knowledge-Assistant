# Production RAG Knowledge Assistant

Grounded question-answering over your own documents, with **citations, confidence
estimation, graceful fallback, evaluation and monitoring** built in. Designed as a
production-style reference implementation rather than a "chat with PDF" demo.

> **Status of this build.** The full retrieval → generation pipeline, the
> Streamlit dashboard, structured logging, monitoring analytics, and the
> retrieval-metrics evaluation harness are implemented and tested. The
> LLM-judged *generation* metrics (faithfulness / groundedness / answer
> relevance) and the remaining `docs/` files are scaffolded and flagged as the
> next milestone — see [Roadmap](#roadmap).

---

## Overview

Organizations keep answers buried in policies, handbooks, contracts, SOPs and
manuals. Employees need answers that are **accurate, attributable, and honest
about uncertainty**. This system:

- retrieves only from documents you upload,
- cites the exact **document, page, chunk and passage** behind every answer,
- attaches a **confidence score** derived from retrieval signals,
- **declines to answer** when evidence is insufficient (two independent guards),
- **logs every interaction** as structured JSON for monitoring and evaluation.

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
                                              ┌──────────────────────────┐
                                              │  Grounded LLM generation  │
                                              │  (citation-forcing prompt)│
                                              └────────────┬──────────────┘
                                                           ▼
                                    answer + citations + confidence + JSON log
```

Two independent hallucination guards:
1. **Retrieval-side** — a configurable confidence threshold on retrieval signals.
2. **Generation-side** — the model must emit `INSUFFICIENT_CONTEXT` when the
   passages don't support an answer.

See [`docs/design_decisions.md`](docs/design_decisions.md) for the reasoning
behind each choice.

## Technology stack

| Concern        | Choice                                   | Notes                          |
|----------------|------------------------------------------|--------------------------------|
| Language       | Python 3.12+                             |                                |
| Chunking       | `langchain-text-splitters`               | isolated to one file           |
| Embeddings     | `sentence-transformers` (BAAI/bge-small) | swappable via config           |
| Vector store   | FAISS (`IndexFlatIP`, cosine)            | Chroma-ready interface         |
| LLM            | OpenAI / Gemini / Ollama                 | selected via `.env`            |
| UI             | Streamlit (7-page dashboard)             |                                |
| Evaluation     | custom retrieval metrics (+ Ragas path)  |                                |
| Logging        | structured JSONL + `logging`             |                                |
| Config         | `config.yaml` + `.env`                    |                                |

## Project structure

```
production-rag-assistant/
├── app.py                     # Streamlit dashboard (presentation only)
├── config.yaml                # all tunable behaviour
├── .env.example               # secrets + provider selection
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
└── docs/            design_decisions.md · architecture.md · …
```

## Installation

```bash
git clone <your-repo> && cd production-rag-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit .env
```

## Configuration

- **`config.yaml`** — behaviour: chunk size/overlap, embedding model, top-k,
  confidence weights + threshold, provider/model defaults.
- **`.env`** — secrets and provider selection. Any `SECTION__KEY` env var
  overrides the matching `config.yaml` value (e.g. `RETRIEVAL__TOP_K=8`).

Pick your LLM in `.env`:

```env
GENERATION__PROVIDER=openai      # openai | gemini | ollama
GENERATION__MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

Ollama needs no key — run a local model and set `GENERATION__PROVIDER=ollama`.

## Running

```bash
streamlit run app.py
```

Dashboard pages: **Home · Upload · Knowledge Base · Ask · Evaluation ·
Monitoring · Settings**. Upload documents, then ask questions; answers show
citations, a confidence meter, retrieved chunks with similarity scores, and the
exact prompt sent to the model.

## Evaluation

A gold dataset lives at `data/evaluation/questions.csv`
(`question, expected_answer, relevant_documents, relevant_pages,
ground_truth_passage`). Relevance is matched at **document+page** level so
re-chunking never invalidates labels.

Retrieval metrics implemented and unit-tested: **Recall@K, Precision@K, Hit@K,
MRR, nDCG@K**. Run from the Evaluation page or:

```python
from src.utils.pipeline import RAGService
from src.evaluation.evaluator import evaluate_retrieval
svc = RAGService()
print(evaluate_retrieval(svc.retriever, "data/evaluation/questions.csv", k=5)["aggregate"])
```

The shipped CSV has 5 rows to demonstrate the schema; expand to 50–100 for a
meaningful benchmark.

## Confidence & fallback logic

Confidence combines three interpretable retrieval signals — mean similarity,
support (fraction of top-k above a floor), and spread (score variance) — with
configurable weights, normalized to 0–1. Below `confidence.answer_threshold`
the system returns the closest passages plus an explanation instead of an
answer. Full formula in `src/retrieval/confidence.py`.

## Testing

```bash
pytest -q          # deterministic core: metrics, chunking, confidence, FAISS
```

No API key or GPU needed — the FAISS test uses a fake embedder.

## Roadmap

- [ ] LLM-judge / Ragas generation metrics (faithfulness, groundedness, answer relevance, context precision/recall)
- [ ] Hybrid search (BM25 + dense) and enable the cross-encoder reranker by default
- [ ] Remaining docs: `architecture.md` (sequence diagrams), `file_explanations.md`, `evaluation_report.md`, `monitoring.md`
- [ ] FastAPI service layer + Dockerfile + GitHub Actions CI
- [ ] Session-based conversation memory; chat export

## Why RAG instead of fine-tuning

RAG keeps knowledge **external and updatable** (re-index, don't re-train),
makes answers **attributable** (you can point at the source), and avoids baking
sensitive documents into model weights. Fine-tuning changes *style/format*, not
*facts you can cite*. See `docs/design_decisions.md`.

## Limitations

- Extraction quality depends on the source PDF; scanned/image PDFs need OCR (not yet wired in).
- Confidence is a heuristic over retrieval signals, not a calibrated probability — tune the threshold against your gold set.
- `IndexFlatIP` is exact but linear in corpus size; swap to an ANN index (IVF/HNSW) for very large corpora.
```
