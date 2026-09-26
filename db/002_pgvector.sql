-- db/002_pgvector.sql
-- The pgvector extension, which the semantic layer (005) needs for the
-- embedding column and the cosine distance operator.
--
-- This file once also created doc_chunks, the table behind a retrieval path
-- over the Argo manuals. That path was removed when the project was scoped to
-- the FloatChat problem statement, which asks for retrieval over metadata and
-- summaries of the float data rather than over documents. 009 drops the table
-- from databases that were initialised before then.

CREATE EXTENSION IF NOT EXISTS vector;
