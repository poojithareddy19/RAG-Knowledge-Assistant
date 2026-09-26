-- db/009_drop_doc_chunks.sql
-- Remove the manual-chunk table from databases initialised before the manual
-- retrieval path was taken out. A fresh database never had it; this is a
-- no-op there.

DROP TABLE IF EXISTS doc_chunks;
