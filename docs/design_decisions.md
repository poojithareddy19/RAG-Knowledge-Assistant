# Design Decisions

This document explains *why* the system is built the way it is. In an interview
setting these trade-offs matter more than the code itself.

## RAG over summaries, SQL over measurements

The archive answers two kinds of question and they fail differently. "What does
the database hold about float 1901393" is a description, and the honest answer
is a sentence with the float named as its source. "What is the average surface
temperature in the Arabian Sea in 2021" is a number, and the honest answer is a
query that a reader can run again.

So there are two paths and one router. The summaries path retrieves one-sentence
descriptions of every float and region, embedded with pgvector, and answers from
them with the float or region cited. The data path generates SQL, validates it,
and runs it as a read-only role. Neither path invents the other's kind of
answer: the summaries prompt forbids prior knowledge and the SQL prompt forbids
copying a number out of a summary.

## Why the summaries are the retrieval corpus

The problem statement asks for a vector store of metadata and summaries, which
is a different corpus from the two Argo manuals this project once indexed. The
manuals answered "what does a QC flag of 4 mean", which is a question about the
format rather than about the data, and it was out of scope. The summaries answer
"what is in here", which is the question a researcher asks first.

They are also a better corpus for a small embedding model. Each summary is one
subject described the way somebody would ask about it, so a hit is a whole
answer rather than a fragment of a page, and the citation is the subject itself
rather than a page number that shifts with every edition.

One consequence is worth stating: a summary carries statistics computed at index
time, such as a mean surface temperature, and the model is told they are
statistics of the tables rather than readings. A question that needs a number
computed under a condition the summaries do not carry belongs to the SQL path,
and the router's wording sends it there.

## Embedding model: BAAI/bge-small-en-v1.5

Small (about 130 MB), fast on CPU, strong on the MTEB retrieval benchmarks for
its size, and Apache-licensed. It expects an instruction prefix on the *query*
side only, which `embedding_model.py` applies. The model id is a single config
value, so a larger `bge-base` drops in without code changes; the summaries are
then rebuilt, because vectors from two models do not compare.

Float summaries are near-identical sentences whose only distinguishing token is
a long number, which is exactly what embeddings represent worst: asking for
float 2900007 scored it 0.703 against 0.701 for an unrelated float. So an
identifier in the question is looked up literally, at similarity 1.0, before the
nearest-neighbour search runs. A question that names a float gets that float.

## Vector store: pgvector, no index on the summaries table

The summaries live in PostgreSQL beside the tables they describe, so one
database holds everything and one rebuild script keeps them in step. The table
is one row per float and per region, thousands at most, and an exact scan
answers in under a millisecond. An approximate index here is actively harmful:
IVFFlat with a hundred lists over a few dozen rows leaves nearly every list
empty, and the single default probe then scans an empty one. The manual-chunk
table this project once had lost most of its recall to exactly that
misconfiguration, and the fix was measured at hit@5 0.160 to 0.960. The lesson
kept is that an index is a decision about the corpus size, not a default.

## Confidence from retrieval signals, not model self-report

Asking a model "how confident are you?" produces poorly calibrated numbers; the
model has no access to whether the corpus contains the answer. Confidence is
computed from **retrieval** signals instead, which measure corpus-to-question
fit directly:

- **mean similarity** (weight 0.6): how close the top-k summaries are to the
  question;
- **support** (0.3): the fraction of the top-k above a similarity floor, which
  separates one lucky hit from broad corroboration;
- **spread** (0.1): one minus the normalised variance, since a pool mixing
  relevant and irrelevant summaries lowers confidence.

The combination is multiplicative: mean similarity is the primary signal, and
the other two are folded into a 0.5 to 1.0 multiplier that can penalise a strong
primary signal but cannot manufacture one. Below a hard floor of 0.28 the score
is zero regardless. An additive version once let a question with very low mean
similarity clear the threshold on support and spread alone, which is the failure
that matters most.

Every displayed number traces back to a statistic, which is what makes the
confidence meter explainable. The weights and threshold are config values and
the default 0.45 threshold is a starting point rather than a tuned constant.

## Two independent hallucination guards

Relying on a single guard is fragile. Two that fail independently:

1. **Retrieval gate.** Below the threshold, decline before calling the model,
   which also saves the latency of a generation.
2. **Generation self-abstention.** The prompt forces the model to emit
   `INSUFFICIENT_CONTEXT` when the summaries do not support an answer, which
   catches the case where retrieval *looked* confident but the content is off
   topic.

Declining returns the closest summaries plus a plain-language reason, so the
user can judge for themselves and refine the question.

## Rules that hold are code, not prose

Three attempts at prompt wording failed to stop the SQL model from aliasing the
deepest pressure a float reached as the depth of the seabed, and from answering
"current speed at 1000 decibars" with a surface velocity. The rule became a
lexical scope gate that runs before the model is called, and a validator rule
that rejects a join between the two platforms. Both are exact, cost nothing,
and do not depend on which model is shipped. The same reasoning put the
generated SQL behind a read-only database role: a validator bug is then a
failed query rather than a data-loss incident.

## Provider abstraction

Ollama sits behind one `BaseLLM` interface returning a normalised `LLMResponse`
with text and token usage. The rest of the system never branches on provider,
so adding one is a subclass plus a registry entry. Running fully local, with no
API key, is what lets the whole system be exercised on a laptop against the
real archive.

## Structured JSONL logging

One JSON object per line is greppable, streamable, and loads straight into
pandas without a database. The same log feeds the monitoring dashboard and
offline evaluation, so operational data and eval data never drift apart.

## OpenTelemetry tracing beside the log, not instead of it

The log has one line per question, which is right for evaluation and the
dashboard and useless for asking why one answer took 70 seconds. One question
can make four model calls (rewrite, route, write SQL, repair it) and
a database query, and only a trace shows them in order with their own timings.

OpenTelemetry rather than a tracing product, because the spans are then
vendor-neutral: the same instrumentation writes to a local file by default and
to Jaeger, Phoenix or Langfuse by setting one environment variable. A
self-hosted Langfuse was the alternative, and it needs its own Postgres,
ClickHouse and web containers, which this machine cannot spare. The default
exporter is a file, synchronous per span, so a killed Streamlit rerun does not
lose the spans still queued.

Prompt text is off by default. Questions can be personal, and the interaction
log already holds the question and answer; the trace's job is timing and
tokens.
