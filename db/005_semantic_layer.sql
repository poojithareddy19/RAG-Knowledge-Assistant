-- db/005_semantic_layer.sql
-- Natural-language summaries of what the measurements tables actually contain.
--
-- The schema catalog tells the model what columns exist. This tells it what is
-- in them: which floats, which regions, which years. Separate from doc_chunks
-- because a question about a policy document must not retrieve a float, and a
-- question about a float must not retrieve a policy document.

CREATE TABLE IF NOT EXISTS data_summaries (
    summary_id BIGSERIAL PRIMARY KEY,
    subject_kind TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    built_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (subject_kind, subject_id)
);

CREATE INDEX IF NOT EXISTS idx_summaries_embedding
    ON data_summaries
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
