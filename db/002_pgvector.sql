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
-- lists=100 is a reasonable starting point for tens of thousands
-- of rows; raise it as the corpus grows.

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON doc_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);