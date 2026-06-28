"""
Hybrid Retriever — Combines semantic search (ChromaDB) with keyword search (BM25),
merges results via Reciprocal Rank Fusion (RRF), and re-ranks with a cross-encoder.

This multi-stage retrieval pipeline ensures both semantic understanding and
exact keyword matching, producing high-quality ranked chunks for the LLM.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.bm25_index import BM25Index
from app.embedding_engine import EmbeddingEngine
from app.models import DocumentChunk, RankedChunk
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Lazy-loaded cross-encoder
_cross_encoder = None


def _get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Lazy-load the cross-encoder model (singleton)."""
    global _cross_encoder
    if _cross_encoder is None:
        logger.info(f"Loading cross-encoder model: {model_name}")
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(model_name)
        logger.info("Cross-encoder model loaded")
    return _cross_encoder


class HybridRetriever:
    """
    Multi-stage retrieval pipeline:
    1. Semantic search via ChromaDB
    2. Keyword search via BM25
    3. Reciprocal Rank Fusion (RRF) to merge results
    4. Cross-encoder re-ranking for final precision
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedding_engine: EmbeddingEngine,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 10,
        top_n: int = 5,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_engine = embedding_engine
        self.cross_encoder_model = cross_encoder_model
        self.top_k = top_k   # candidates from each retriever
        self.top_n = top_n   # final results after re-ranking
        self.rrf_k = rrf_k   # RRF constant

    def retrieve(self, query: str) -> list[RankedChunk]:
        """
        Full retrieval pipeline:
        Semantic + BM25 → RRF Merge → Cross-Encoder Re-rank → Top-N
        """
        logger.info(f"Retrieving for query: {query[:80]}...")

        # Stage 1: Semantic search
        query_embedding = self.embedding_engine.embed_query(query)
        semantic_results = self.vector_store.search(query_embedding, top_k=self.top_k)

        # Stage 2: BM25 keyword search
        bm25_results = self.bm25_index.search(query, top_k=self.top_k)

        # Stage 3: Reciprocal Rank Fusion
        merged = self._reciprocal_rank_fusion(semantic_results, bm25_results)

        if not merged:
            logger.warning("No results from retrieval")
            return []

        # Stage 4: Cross-encoder re-ranking
        reranked = self._cross_encoder_rerank(query, merged)

        logger.info(f"Retrieval complete — {len(reranked)} results")
        return reranked[:self.top_n]

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[dict],
        bm25_results: list[dict],
    ) -> list[RankedChunk]:
        """
        Merge semantic and BM25 results using Reciprocal Rank Fusion.
        RRF score = Σ 1 / (k + rank_i) for each ranking the document appears in.
        """
        chunk_map: dict[str, RankedChunk] = {}

        # Process semantic results
        for rank, hit in enumerate(semantic_results):
            chunk_id = hit["chunk_id"]
            if chunk_id not in chunk_map:
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=hit["text"],
                    document_name=hit["metadata"].get("document_name", ""),
                    page_number=hit["metadata"].get("page_number", 1),
                    chunk_index=hit["metadata"].get("chunk_index", 0),
                    file_type=hit["metadata"].get("file_type", ""),
                )
                chunk_map[chunk_id] = RankedChunk(
                    chunk=chunk,
                    retrieval_score=hit.get("score", 0.0),
                )
            chunk_map[chunk_id].combined_score += 1.0 / (self.rrf_k + rank + 1)

        # Process BM25 results
        for rank, hit in enumerate(bm25_results):
            chunk: DocumentChunk = hit["chunk"]
            chunk_id = chunk.chunk_id
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = RankedChunk(
                    chunk=chunk,
                    retrieval_score=0.0,
                )
            # Keep the higher retrieval score
            bm25_norm = min(hit.get("score", 0.0) / 20.0, 1.0)  # normalize BM25 scores
            chunk_map[chunk_id].retrieval_score = max(
                chunk_map[chunk_id].retrieval_score, bm25_norm
            )
            chunk_map[chunk_id].combined_score += 1.0 / (self.rrf_k + rank + 1)

        # Sort by RRF score
        merged = sorted(chunk_map.values(), key=lambda x: x.combined_score, reverse=True)

        logger.info(
            f"RRF merge: {len(semantic_results)} semantic + {len(bm25_results)} BM25 → {len(merged)} unique"
        )
        return merged

    # ------------------------------------------------------------------
    # Cross-Encoder Re-ranking
    # ------------------------------------------------------------------

    def _cross_encoder_rerank(self, query: str, candidates: list[RankedChunk]) -> list[RankedChunk]:
        """Re-rank candidates using a cross-encoder for maximum precision."""
        if not candidates:
            return []

        try:
            cross_encoder = _get_cross_encoder(self.cross_encoder_model)

            # Prepare query-document pairs
            pairs = [(query, c.chunk.text) for c in candidates]

            # Score all pairs
            scores = cross_encoder.predict(pairs)

            # Convert raw logits to probabilities using sigmoid
            import math
            for i, candidate in enumerate(candidates):
                raw_score = float(scores[i])
                sigmoid_score = 1 / (1 + math.exp(-raw_score))
                candidate.reranker_score = sigmoid_score
                candidate.combined_score = sigmoid_score

            # Sort by reranker score
            candidates.sort(key=lambda x: x.reranker_score, reverse=True)
            
            top_score = candidates[0].reranker_score if candidates else 0.0
            
            # Remove chunks whose rerank score is less than 20% of the top rerank score
            cutoff = 0.20 * top_score
            filtered = [c for c in candidates if c.reranker_score >= cutoff]
            
            # Keep top 5
            filtered = filtered[:self.top_n]
            
            logger.info(
                f"Cross-encoder re-ranked {len(candidates)} candidates. "
                f"Filtered {len(candidates) - len(filtered)} noisy chunks. Top score: {top_score:.3f}"
            )
            
            # Debug Logging for Top 5
            for i, c in enumerate(filtered, 1):
                raw_s = float(scores[candidates.index(c)])
                logger.info(
                    f"Chunk {i} | Source: {c.chunk.document_name} (Page {c.chunk.page_number}) | "
                    f"Raw Cross Encoder Score: {raw_s:.3f} | Sigmoid Score: {c.reranker_score:.3f}"
                )
            
            return filtered

        except Exception as e:
            logger.error(f"Cross-encoder re-ranking failed: {e}. Using RRF scores.")

        return candidates
