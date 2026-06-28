"""
Embedding Engine — Generates vector embeddings using sentence-transformers.

Uses the all-MiniLM-L6-v2 model (384 dimensions) which runs locally
with no API key required. Provides both batch and single-query embedding.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded global model to avoid repeated loading
_model = None


def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load the sentence-transformer model (singleton)."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {model_name}")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully")
    return _model


class EmbeddingEngine:
    """Wraps sentence-transformers for embedding generation."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Optional[object] = None

    @property
    def model(self):
        if self._model is None:
            self._model = _get_model(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of embedding vectors."""
        if not texts:
            return []

        logger.info(f"Embedding {len(texts)} texts")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=32,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding[0].tolist()

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality."""
        return self.model.get_sentence_embedding_dimension()
