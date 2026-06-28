"""
BM25 Index — Keyword-based retrieval using the BM25 algorithm.

Complements the semantic (vector) search by capturing exact keyword matches
that embedding models may miss (e.g., acronyms, product names, policy numbers).
The index is serialized to disk for persistence across restarts.
"""

from __future__ import annotations

import logging
import heapq
import os
import pickle
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from app.models import DocumentChunk

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25-based keyword search index for document chunks."""

    def __init__(self, index_path: str = "./bm25_index.pkl"):
        self.index_path = index_path
        self._chunks: list[DocumentChunk] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: Optional[BM25Okapi] = None

        # Try loading from disk
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[DocumentChunk]):
        """Add chunks to the BM25 index and persist."""
        if not chunks:
            return

        self._chunks.extend(chunks)
        # Re-tokenize the entire corpus and rebuild the BM25 model
        self._tokenized_corpus = [self._tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._save()

        logger.info(f"BM25 index updated — total {len(self._chunks)} chunks")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Search the BM25 index. Returns list of dicts with keys:
        chunk, score
        """
        if self._bm25 is None or not self._chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices without sorting the full corpus.
        top_indices = heapq.nlargest(top_k, range(len(scores)), key=lambda i: scores[i])

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    {
                        "chunk": self._chunks[idx],
                        "score": float(scores[idx]),
                    }
                )

        return results

    def remove_document(self, document_name: str) -> int:
        """Remove all chunks associated with a specific document and rebuild index."""
        if not self._chunks:
            return 0
            
        original_count = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.document_name != document_name]
        removed_count = original_count - len(self._chunks)
        
        if removed_count > 0:
            if self._chunks:
                self._tokenized_corpus = [self._tokenize(c.text) for c in self._chunks]
                self._bm25 = BM25Okapi(self._tokenized_corpus)
            else:
                self._tokenized_corpus = []
                self._bm25 = None
            self._save()
            logger.info(f"Removed {removed_count} chunks for {document_name} from BM25 index")
            
        return removed_count

    def clear(self):
        """Clear the index and remove persisted file."""
        self._chunks = []
        self._tokenized_corpus = []
        self._bm25 = None
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        logger.info("BM25 index cleared")

    @property
    def count(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing."""
        text = text.lower()
        # Remove special characters but keep alphanumeric and hyphens
        tokens = re.findall(r"\b[a-z0-9][\w\-]*\b", text)
        return tokens

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        """Persist the index to disk."""
        try:
            data = {
                "chunks": self._chunks,
                "tokenized_corpus": self._tokenized_corpus,
            }
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"BM25 index saved to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")

    def _load(self):
        """Load the index from disk if available."""
        if not os.path.exists(self.index_path):
            return

        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            self._chunks = data["chunks"]
            self._tokenized_corpus = data["tokenized_corpus"]
            if self._tokenized_corpus:
                self._bm25 = BM25Okapi(self._tokenized_corpus)
            logger.info(f"BM25 index loaded from disk — {len(self._chunks)} chunks")
        except Exception as e:
            logger.warning(f"Failed to load BM25 index: {e}")
            self._chunks = []
            self._tokenized_corpus = []
            self._bm25 = None
