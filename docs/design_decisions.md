# Design Decisions

This document explains *why* the system is built the way it is. In an interview
setting these trade-offs matter more than the code itself.

## RAG instead of fine-tuning

The task is "answer strictly from these documents and cite the source."
Fine-tuning teaches a model *style and format*; it does not give you
**attribution**, **freshness**, or **access control**. RAG keeps the knowledge
external: re-indexing a changed handbook is instant, every claim can be traced
to a passage, and sensitive documents never enter model weights. Fine-tuning
would also have to be redone on every document change and still couldn't produce
a citation. We would only add fine-tuning to shape *how* answers are phrased,
not *what facts* they contain.

## Embedding model: BAAI/bge-small-en-v1.5 (default)

Small (~130MB), fast on CPU, strong on the MTEB retrieval benchmarks for its
size, and Apache-licensed. It expects an instruction prefix on the *query* side
only, which we handle in `embedding_model.py`. The model id is a single config
value; `all-MiniLM-L6-v2` (even smaller, no prefix) or a larger `bge-base` drop
in without code changes. We normalize embeddings so cosine similarity reduces to
an inner product, which lets FAISS use its fast `IndexFlatIP`.

## Vector store: FAISS

`IndexFlatIP` gives **exact** nearest-neighbour search — no recall loss from
approximation — which keeps evaluation honest for a corpus of realistic
document size. FAISS stores only vectors, so chunk metadata and a per-document
manifest are persisted alongside the index. The store exposes a deliberately
small interface (`add / search / save / load / stats / reset`) so a Chroma
backend could implement the same contract without touching retrieval,
generation or the UI. For very large corpora, switching to an ANN index
(IVF/HNSW) is a one-line change at index construction.

## Chunking: per-page recursive, 1000/150 chars

We split **per page** so a chunk never straddles a page boundary — this keeps
citations exact (a chunk is always on one page). Within a page we use a
recursive splitter that prefers natural boundaries (paragraph → line → sentence
→ word). Size 1000 chars (~250 tokens) balances two failure modes: chunks too
small lose context the LLM needs; chunks too large dilute the embedding and
retrieve loosely. 150-char overlap (~15%) prevents an answer that sits on a
boundary from being cut in half. These are the biggest levers on retrieval
quality and are the first thing to tune against the gold set.

## Framework use: minimal and isolated

The brief allows LangChain or LlamaIndex. Heavy framework use tends to obscure
what the system actually does and breaks often across versions. We therefore use
exactly one framework primitive — LangChain's battle-tested
`RecursiveCharacterTextSplitter` — and confine it to `chunker.py`. Everything
else (embedding, indexing, retrieval, prompting, provider calls) is explicit.
This is easier to reason about, cheaper to maintain, and demonstrates
understanding of the pipeline rather than glue skills. Swapping the splitter is
a one-file change.

## Confidence from retrieval signals, not LLM self-report

Asking an LLM "how confident are you?" produces poorly calibrated numbers — the
model has no access to whether the corpus actually contains the answer. Instead
we compute confidence from **retrieval** signals, which directly measure
corpus–question fit:

- **mean similarity** (weight 0.6): how close the top-k passages are to the query;
- **support** (0.3): fraction of top-k above a similarity floor — separates one
  lucky hit from broad corroboration;
- **spread** (0.1): `1 - normalized variance` — a mixed pool of relevant and
  irrelevant passages lowers confidence.

Every displayed number traces back to a statistic, which is what makes the
confidence meter *explainable*. The weights and threshold are config values and
should be **calibrated against the gold dataset** — the default 0.45 threshold
is a starting point, not a tuned constant.

## Two independent hallucination guards

Relying on a single guard is fragile. We use two that fail independently:

1. **Retrieval gate** — if confidence < threshold, decline before ever calling
   the LLM (also saves cost/latency).
2. **Generation self-abstention** — the prompt forces the model to emit
   `INSUFFICIENT_CONTEXT` when the passages don't support an answer, catching
   cases where retrieval *looked* confident but the content is off-topic.

Declining returns the closest passages plus a plain-language reason, so the user
can judge for themselves and refine the question.

## Provider abstraction

OpenAI, Gemini and Ollama sit behind one `BaseLLM` interface returning a
normalized `LLMResponse` (text + token usage). The rest of the system never
branches on provider; adding one is a subclass plus a registry entry. Ollama
support means the whole system can run **fully local** with no API key, which
matters for sensitive internal documents.

## Structured JSONL logging

One JSON object per line is greppable, streamable, and loads straight into
pandas — no database required for a reference implementation. The same log feeds
both the monitoring dashboard and offline evaluation, so operational data and
eval data never drift apart.
