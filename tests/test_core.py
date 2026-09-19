"""Unit tests for deterministic core components.

These run without any ML model or API key (a fake embedder is used for the
FAISS test), so they are fast and CI-friendly.

Run: pytest -q
"""
import hashlib

import numpy as np
import pytest

from src.evaluation import metrics as M
from src.ingestion.chunker import chunk_segments
from src.retrieval.confidence import estimate_confidence
from src.utils.schemas import Chunk, RetrievedChunk
from src.vectorstore.vectordb import FaissVectorStore


# --- retrieval metrics ------------------------------------------------------
def test_recall_precision_hit():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "e"}
    assert M.recall_at_k(retrieved, relevant, 4) == pytest.approx(0.5)
    assert M.precision_at_k(retrieved, relevant, 4) == pytest.approx(0.25)
    assert M.hit_at_k(retrieved, relevant, 4) == 1.0
    assert M.hit_at_k(retrieved, {"z"}, 4) == 0.0


def test_mrr_and_ndcg():
    assert M.mrr(["x", "y", "gold"], {"gold"}) == pytest.approx(1 / 3)
    assert M.mrr(["gold"], {"gold"}) == 1.0
    assert M.mrr(["a", "b"], {"gold"}) == 0.0
    # first-position relevant => perfect nDCG
    assert M.ndcg_at_k(["gold", "x"], {"gold"}, 2) == pytest.approx(1.0)


# --- chunking ---------------------------------------------------------------
def test_chunker_page_attribution_and_ids():
    segments = [(1, "alpha " * 400), (2, "beta " * 400)]
    chunks = chunk_segments("Doc.pdf", segments)
    assert len(chunks) > 2
    assert all(isinstance(c, Chunk) for c in chunks)
    # ids encode document and page; no chunk straddles pages
    assert all(c.chunk_id.startswith("Doc.pdf::p") for c in chunks)
    assert {c.page for c in chunks} == {1, 2}


# --- confidence -------------------------------------------------------------
def _rc(score, i):
    return RetrievedChunk(chunk=Chunk("id", "D", 1, "t"), score=score, rank=i)


def test_confidence_monotonic():
    strong = estimate_confidence([_rc(0.9, 1), _rc(0.85, 2), _rc(0.8, 3)])
    weak = estimate_confidence([_rc(0.2, 1), _rc(0.1, 2), _rc(0.05, 3)])
    assert strong.score > weak.score
    assert 0.0 <= weak.score <= strong.score <= 1.0
    assert estimate_confidence([]).score == 0.0


# --- faiss store ------------------------------------------------------------
def _fake_vec(text, dim=16):
    h = hashlib.sha256(text.encode()).digest()
    v = np.frombuffer((h * 2)[:dim], dtype=np.uint8).astype("float32")
    v -= v.mean()
    n = np.linalg.norm(v)
    return (v / n if n else v).astype("float32")


def test_faiss_add_search_persist(tmp_path):
    dim = 16
    store = FaissVectorStore(dimension=dim, store_dir=str(tmp_path))
    chunks = [Chunk(f"D::p1::c{i}", "D", 1, f"passage number {i}") for i in range(5)]
    vecs = np.vstack([_fake_vec(c.text, dim) for c in chunks])
    store.add(chunks, vecs, embedding_model="fake")
    store.save()

    reloaded = FaissVectorStore.load(dimension=dim, store_dir=str(tmp_path))
    assert reloaded.index.ntotal == 5
    results = reloaded.search(_fake_vec(chunks[2].text, dim), k=3)
    assert results[0].chunk.chunk_id == "D::p1::c2"
    assert results[0].score == pytest.approx(1.0, abs=1e-3)


def test_recall_never_exceeds_one_when_a_page_returns_several_chunks():
    """A page is many chunks, and finding two of them is still one page found."""
    from src.evaluation.metrics import recall_at_k

    retrieved = ["doc.pdf|12", "doc.pdf|12", "doc.pdf|99"]

    assert recall_at_k(retrieved, {"doc.pdf|12"}, 5) == 1.0


def test_recall_counts_distinct_relevant_pages():
    from src.evaluation.metrics import recall_at_k

    retrieved = ["doc.pdf|12", "doc.pdf|12", "doc.pdf|13"]

    assert recall_at_k(retrieved, {"doc.pdf|12", "doc.pdf|13", "doc.pdf|14"}, 5) == 2 / 3
