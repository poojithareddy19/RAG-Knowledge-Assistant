"""Pipeline service facade.

One object that wires the whole system together so callers (Streamlit, tests,
a future FastAPI layer) never touch individual modules. Responsibilities:

  * lazily build/load the FAISS store (sized to the embedding model dimension)
  * ingest files: load -> chunk -> embed -> add -> persist
  * answer questions through the AnswerGenerator and log every interaction
  * expose knowledge-base stats

Kept deliberately thin: it orchestrates, it doesn't implement.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from src.embeddings.embedding_model import get_embedding_model
from src.generation.answer_generator import AnswerGenerator
from src.ingestion.chunker import chunk_segments
from src.ingestion.loader import SUPPORTED_EXTENSIONS, load_document
from src.monitoring.logger import log_interaction
from src.retrieval.retriever import Retriever
from src.utils.config import get_config
from src.utils.schemas import Answer, Chunk
from src.vectorstore.vectordb import FaissVectorStore


class RAGService:
    def __init__(self) -> None:
        self.cfg = get_config()
        self.embedder = get_embedding_model()
        self.store = FaissVectorStore.load(dimension=self.embedder.dimension)
        self.retriever = Retriever(self.store)
        self.generator = AnswerGenerator(self.retriever)

    # -- ingestion -----------------------------------------------------------
    def ingest_file(self, file_path: str | Path, skip_duplicates: bool = True) -> dict:
        file_path = Path(file_path)
        doc_name = file_path.name

        if skip_duplicates and self.store.contains(doc_name):
            return {"doc_name": doc_name, "status": "skipped_duplicate", "chunks": 0}

        segments = load_document(file_path)
        chunks: list[Chunk] = chunk_segments(doc_name, segments)
        if not chunks:
            return {"doc_name": doc_name, "status": "no_text", "chunks": 0}

        vectors = self.embedder.embed_passages([c.text for c in chunks])
        self.store.add(chunks, vectors, embedding_model=self.embedder.model_name)
        self.store.save()

        # Keep a copy of the raw file for provenance.
        raw_dir = Path(self.cfg.paths.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        try:
            if Path(file_path).resolve() != (raw_dir / doc_name).resolve():
                shutil.copy2(file_path, raw_dir / doc_name)
        except Exception:
            pass

        return {
            "doc_name": doc_name,
            "status": "indexed",
            "pages": max((c.page for c in chunks), default=0),
            "chunks": len(chunks),
        }

    # -- querying ------------------------------------------------------------
    def ask(self, question: str) -> Answer:
        answer = self.generator.answer(question)
        try:
            log_interaction(answer)
        except Exception:
            pass  # logging must never break the user path
        return answer

    # -- knowledge base ------------------------------------------------------
    def stats(self) -> dict:
        return self.store.stats()

    def reset(self) -> None:
        self.store.reset()

    @staticmethod
    def supported_extensions() -> tuple:
        return SUPPORTED_EXTENSIONS
