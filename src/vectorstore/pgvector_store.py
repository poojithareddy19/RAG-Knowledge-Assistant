"""PostgreSQL + pgvector vector store.

Implements the same public interface as FaissVectorStore so the rest of the
application can switch vector backends without changing ingestion or retrieval.
"""

from __future__ import annotations

import numpy as np
from pgvector.psycopg import register_vector

from src.utils.db import cursor
from src.utils.schemas import Chunk, RetrievedChunk


class PgVectorStore:
    """PostgreSQL-backed vector store using pgvector cosine distance."""

    def __init__(self, table: str = "doc_chunks", dimension: int = 384) -> None:
        self.table = table
        self.dimension = dimension

    # -- write ---------------------------------------------------------------

    def add(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        embedding_model: str,
    ) -> None:
        """Add chunks and their embeddings to PostgreSQL."""

        if len(chunks) == 0:
            return

        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError(
                f"Expected 2D vectors, got shape {vectors.shape}"
            )

        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"{len(chunks)} chunks but {vectors.shape[0]} vectors"
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dim {vectors.shape[1]} != store dimension "
                f"{self.dimension}"
            )

        rows = []

        for chunk, vector in zip(chunks, vectors, strict=True):
            rows.append(
                (
                    chunk.doc_name,
                    chunk.page,
                    self._chunk_index(chunk),
                    chunk.text,
                    vector,
                )
            )

        sql = f"""
            INSERT INTO {self.table}
                (document, page, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s, %s)
        """

        with cursor() as cur:
            register_vector(cur.connection)
            cur.executemany(sql, rows)

    def contains(self, doc_name: str) -> bool:
        """Return True when at least one chunk from the document exists."""

        with cursor(readonly=True) as cur:
            cur.execute(
                f"""
                SELECT EXISTS(
                    SELECT 1
                    FROM {self.table}
                    WHERE document = %s
                )
                """,
                (doc_name,),
            )
            return bool(cur.fetchone()[0])

    # -- read ----------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
    ) -> list[RetrievedChunk]:
        """Search by cosine similarity."""

        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)

        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"Query dim {vector.shape[0]} != store dimension "
                f"{self.dimension}"
            )

        sql = f"""
            SELECT
                chunk_id,
                document,
                page,
                chunk_index,
                content,
                1 - (embedding <=> %s) AS similarity
            FROM {self.table}
            ORDER BY embedding <=> %s
            LIMIT %s
        """

        with cursor(readonly=True) as cur:
            register_vector(cur.connection)
            cur.execute(sql, (vector, vector, k))
            rows = cur.fetchall()

        results: list[RetrievedChunk] = []

        for rank, row in enumerate(rows, start=1):
            chunk = Chunk(
                chunk_id=str(row[0]),
                doc_name=row[1],
                page=int(row[2] or 0),
                text=row[4],
            )

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(row[5]),
                    rank=rank,
                )
            )

        return results

    # -- metadata ------------------------------------------------------------

    def stats(self) -> dict:
        """Return statistics compatible with FaissVectorStore."""

        with cursor(readonly=True) as cur:
            cur.execute(
                f"""
                SELECT
                    COUNT(*) AS chunks,
                    COUNT(DISTINCT document) AS documents
                FROM {self.table}
                """
            )
            chunks, documents = cur.fetchone()

            cur.execute(
                f"""
                SELECT DISTINCT document
                FROM {self.table}
                ORDER BY document
                """
            )
            manifest = [row[0] for row in cur.fetchall()]

        return {
            "documents": int(documents),
            "chunks": int(chunks),
            "dimension": self.dimension,
            "manifest": manifest,
        }

    # -- persistence ---------------------------------------------------------

    def save(self) -> None:
        """No-op because PostgreSQL persists writes immediately."""

    def reset(self) -> None:
        """Remove all indexed document chunks."""

        with cursor() as cur:
            cur.execute(f"TRUNCATE {self.table}")

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _chunk_index(chunk: Chunk) -> int:
        """Extract the chunk index from a standard chunk id when possible."""

        try:
            return int(chunk.chunk_id.rsplit("::c", 1)[1])
        except (IndexError, ValueError):
            return 0