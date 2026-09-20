"""PostgreSQL + pgvector backed vector store.

This is a drop-in replacement for :class:`FaissVectorStore`: same method names,
same argument shapes, same return types.
"""

from __future__ import annotations

import re

import numpy as np

from src.utils.db import cursor
from src.utils.schemas import Chunk, RetrievedChunk

_CHUNK_N = re.compile(r"::c(\d+)$")


class PgVectorStore:
    def __init__(
        self,
        dimension: int = 384,
        table: str = "doc_chunks",
    ) -> None:
        self.dimension = dimension
        self.table = table

    # -- write --------------------------------------------------------------

    def add(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        embedding_model: str = "",
    ) -> None:
        """Insert chunks and their embeddings."""
        if not chunks:
            return

        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dim {vectors.shape[-1]} != table dim {self.dimension}"
            )

        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"{len(chunks)} chunks but {vectors.shape[0]} vectors"
            )

        rows = [
            (
                chunk.doc_name,
                chunk.page,
                self._chunk_index(chunk, fallback=i),
                chunk.text,
                vectors[i],
            )
            for i, chunk in enumerate(chunks)
        ]

        sql = f"""
            INSERT INTO {self.table}
                (document, page, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s, %s)
        """

        with cursor() as cur:
            cur.executemany(sql, rows)

    @staticmethod
    def _chunk_index(chunk: Chunk, fallback: int) -> int:
        match = _CHUNK_N.search(chunk.chunk_id or "")
        return int(match.group(1)) if match else fallback

    def contains(self, doc_name: str) -> bool:
        with cursor(readonly=True) as cur:
            cur.execute(
                f"""
                SELECT 1
                FROM {self.table}
                WHERE document = %s
                LIMIT 1
                """,
                (doc_name,),
            )
            return cur.fetchone() is not None

    # -- read ----------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
    ) -> list[RetrievedChunk]:
        vec = np.asarray(query_vector, dtype=np.float32).reshape(-1)

        if vec.shape[0] != self.dimension:
            raise ValueError(
                f"Query dim {vec.shape[0]} != table dim {self.dimension}"
            )

        sql = f"""
            SELECT
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
            cur.execute(sql, (vec, vec, k))
            rows = cur.fetchall()

        results: list[RetrievedChunk] = []

        for rank, (
            document,
            page,
            chunk_index,
            content,
            similarity,
        ) in enumerate(rows, start=1):
            chunk = Chunk(
                chunk_id=f"{document}::p{page}::c{chunk_index}",
                doc_name=document,
                page=page or 0,
                text=content,
            )

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(similarity),
                    rank=rank,
                )
            )

        return results

    def stats(self) -> dict:
        """Return the same shape as FaissVectorStore.stats()."""
        with cursor(readonly=True) as cur:
            cur.execute(
                f"""
                SELECT
                    document,
                    COALESCE(MAX(page), 0) AS pages,
                    COUNT(*) AS chunks,
                    MAX(created_at) AS indexed_at
                FROM {self.table}
                GROUP BY document
                ORDER BY document
                """
            )
            rows = cur.fetchall()

        manifest = {
            document: {
                "pages": int(pages),
                "chunks": int(chunks),
                "indexed_at": (
                    indexed_at.strftime("%Y-%m-%d %H:%M:%S")
                    if indexed_at
                    else ""
                ),
                "embedding_model": "",
            }
            for document, pages, chunks, indexed_at in rows
        }

        return {
            "documents": len(manifest),
            "chunks": sum(
                item["chunks"] for item in manifest.values()
            ),
            "dimension": self.dimension,
            "manifest": manifest,
        }

    # -- persistence ---------------------------------------------------------

    def save(self) -> None:
        """No-op because PostgreSQL commits rows during add()."""

    def reset(self) -> None:
        with cursor() as cur:
            cur.execute(
                f"TRUNCATE {self.table} RESTART IDENTITY"
            )