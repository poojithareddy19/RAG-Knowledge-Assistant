"""Embedding generation.

Thin wrapper over ``sentence-transformers`` that (a) applies the correct
instruction prefixes for query vs passage (bge-style models need them,
MiniLM-style models don't), and (b) L2-normalizes so the FAISS inner-product
index yields cosine similarity.

The model is loaded lazily and cached, so importing this module is cheap and the
(heavy) model only loads on first real use.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.utils.config import get_config


class EmbeddingModel:
    """Encapsulates a sentence-transformers model + prefix/normalize policy."""

    def __init__(self) -> None:
        cfg = get_config().embeddings
        self.model_name: str = cfg.model_name
        self.query_prefix: str = cfg.get("query_prefix", "") or ""
        self.passage_prefix: str = cfg.get("passage_prefix", "") or ""
        self.normalize: bool = bool(cfg.get("normalize", True))
        self.batch_size: int = int(cfg.get("batch_size", 32))
        self.device: str = cfg.get("device", "cpu")
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype("float32")


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """Process-wide singleton so the model weights load only once."""
    return EmbeddingModel()
