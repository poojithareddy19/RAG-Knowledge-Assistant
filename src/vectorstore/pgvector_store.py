# src/vectorstore/pgvector_store.py

import numpy as np
from pgvector.psycopg import register_vector

from src.utils.db import cursor


class PgVectorStore:
    """Postgres-backed vector store.

    Mirrors the FAISS store's interface.
    """

    def __init__(self, table="doc_chunks", dimension=384):
        self.table = table
        self.dimension = dimension

    def add(self, records):
        """Add document chunks and their embeddings to PostgreSQL.

        records: list of dicts with:
            document, page, chunk_index, content, embedding
        """
        rows = []

        for record in records:
            vec = np.asarray(
                record["embedding"],
                dtype=np.float32,
            )

            if vec.shape[0] != self.dimension:
                raise ValueError(
                    f"expected {self.dimension} dims, got {vec.shape[0]}"
                )

            rows.append(
                (
                    record["document"],
                    record.get("page"),
                    record.get("chunk_index"),
                    record["content"],
                    vec,
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

        return len(rows)

    def search(self, query_vector, k=5, document=None):
        """Search by cosine similarity.

        Returns hits with a similarity score between 0 and 1.
        """
        vec = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        where = ""
        params = [vec]

        if document:
            where = "WHERE document = %s"
            params.append(document)

        params.extend([vec, k])

        sql = f"""
            SELECT
                chunk_id,
                document,
                page,
                chunk_index,
                content,
                1 - (embedding <=> %s) AS similarity
            FROM {self.table}
            {where}
            ORDER BY embedding <=> %s
            LIMIT %s
        """

        with cursor(readonly=True) as cur:
            register_vector(cur.connection)
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            {
                "chunk_id": row[0],
                "document": row[1],
                "page": row[2],
                "chunk_index": row[3],
                "content": row[4],
                "score": float(row[5]),
            }
            for row in rows
        ]

    def count(self):
        """Return the number of stored document chunks."""
        with cursor(readonly=True) as cur:
            cur.execute(
                f"SELECT count(*) FROM {self.table}"
            )
            return cur.fetchone()[0]

    def clear(self):
        """Remove all document chunks."""
        with cursor() as cur:
            cur.execute(
                f"TRUNCATE {self.table}"
            )
    def stats(self) -> dict:
        """Return statistics compatible with the FAISS vector store."""

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