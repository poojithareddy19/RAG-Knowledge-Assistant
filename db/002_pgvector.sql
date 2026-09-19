-- db/002_pgvector.sql
-- PostgreSQL pgvector extension and document chunk storage.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS doc_chunks (
    chunk_id BIGSERIAL PRIMARY KEY,
    document TEXT NOT NULL,
    page INTEGER,
    chunk_index INTEGER,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document
    ON doc_chunks (document);

-- Cosine distance index.
--
-- HNSW and not IVFFlat, and the reason is worth keeping. This index was
-- previously `ivfflat (embedding vector_cosine_ops) WITH (lists = 100)`,
-- sized for a corpus of tens of thousands of rows that never arrived. IVFFlat
-- partitions the vectors into `lists` clusters and, at the default
-- `ivfflat.probes = 1`, searches exactly one of them. With 668 chunks across
-- 100 lists that is about seven candidates per query, so a request for the
-- top 20 and a request for the top 50 returned the same handful of rows and
-- the right page was usually not among them.
--
-- Measured on the 25-question retrieval gold set: hit@5 went from 0.160 with
-- the IVFFlat index to 0.960 after this change, with nothing else altered.
-- That is not tuning, it was a recall bug.
--
-- HNSW has no probes knob to get wrong, keeps its recall as the corpus grows,
-- and on a table this size costs nothing to build.

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON doc_chunks
    USING hnsw (embedding vector_cosine_ops);