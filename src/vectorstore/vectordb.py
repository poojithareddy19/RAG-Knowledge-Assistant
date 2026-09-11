"""Vector store (FAISS backend).

Wraps a FAISS ``IndexFlatIP`` (inner product on normalized vectors == cosine
similarity). FAISS only stores vectors, so we keep a parallel list of ``Chunk``
metadata and a small ``manifest`` describing indexed documents. All three are
persisted together under ``data/vector_store`` so the index survives restarts.

The public surface is deliberately small — ``add``, ``search``, ``save``,
``load``, ``stats`` — so a Chroma backend could implement the same interface
without touching the rest of the app.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

from src.utils.config import get_config
from src.utils.schemas import Chunk, RetrievedChunk

INDEX_FILE = "index.faiss"
META_FILE = "chunks.pkl"
MANIFEST_FILE = "manifest.json"


class FaissVectorStore:
    def __init__(self, dimension: int, store_dir: str | None = None) -> None:
        import faiss

        self.dimension = dimension
        self.store_dir = Path(store_dir or get_config().paths.vector_store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._faiss = faiss
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[Chunk] = []
        # manifest[doc_name] = {pages, chunks, indexed_at, embedding_model}
        self.manifest: dict[str, dict] = {}

    # -- write ---------------------------------------------------------------
    def add(self, chunks: list[Chunk], vectors: np.ndarray, embedding_model: str) -> None:
        if len(chunks) == 0:
            return
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dim {vectors.shape[1]} != index dim {self.dimension}"
            )
        self.index.add(vectors.astype("float32"))
        self.chunks.extend(chunks)

        doc_name = chunks[0].doc_name
        pages = sorted({c.page for c in chunks})
        self.manifest[doc_name] = {
            "pages": max(pages) if pages else 0,
            "chunks": len([c for c in self.chunks if c.doc_name == doc_name]),
            "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedding_model": embedding_model,
        }

    def contains(self, doc_name: str) -> bool:
        return doc_name in self.manifest

    # -- read ----------------------------------------------------------------
    def search(self, query_vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        if self.index.ntotal == 0:
            return []
        q = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(q, k)
        results: list[RetrievedChunk] = []
        for rank, (idx, score) in enumerate(
            zip(ids[0], scores[0], strict=True), 
            start=1
        ):
            if idx < 0:
                continue
            results.append(
                RetrievedChunk(chunk=self.chunks[idx], score=float(score), rank=rank)
            )
        return results

    def stats(self) -> dict:
        return {
            "documents": len(self.manifest),
            "chunks": self.index.ntotal,
            "dimension": self.dimension,
            "manifest": self.manifest,
        }

    # -- persistence ---------------------------------------------------------
    def save(self) -> None:
        self._faiss.write_index(self.index, str(self.store_dir / INDEX_FILE))
        with open(self.store_dir / META_FILE, "wb") as fh:
            pickle.dump(self.chunks, fh)
        with open(self.store_dir / MANIFEST_FILE, "w", encoding="utf-8") as fh:
            json.dump(self.manifest, fh, indent=2)

    @classmethod
    def load(cls, dimension: int, store_dir: str | None = None) -> FaissVectorStore:
        store = cls(dimension=dimension, store_dir=store_dir)
        index_path = store.store_dir / INDEX_FILE
        if index_path.exists():
            store.index = store._faiss.read_index(str(index_path))
            with open(store.store_dir / META_FILE, "rb") as fh:
                store.chunks = pickle.load(fh)
            manifest_path = store.store_dir / MANIFEST_FILE
            if manifest_path.exists():
                with open(manifest_path, encoding="utf-8") as fh:
                    store.manifest = json.load(fh)
        return store

    def reset(self) -> None:
        """Drop all vectors, metadata and on-disk files."""
        self.index = self._faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        self.manifest = {}
        for name in (INDEX_FILE, META_FILE, MANIFEST_FILE):
            p = self.store_dir / name
            if p.exists():
                p.unlink()
