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

-- Deliberately no vector index.
--
-- This table holds one row per float and per region, so it is thousands of
-- rows at most and an exact scan answers in under a millisecond. An ivfflat
-- index here is actively harmful: with lists = 100 over a few dozen rows
-- nearly every list is empty, and the default single probe then scans an empty
-- one and returns nothing relevant. Postgres warns about this at creation
-- time. Revisit with HNSW only if this table ever reaches six figures.
