"""
Hybrid Retriever — combines semantic search (ChromaDB) with BM25 keyword search,
merges results via Reciprocal Rank Fusion, and re-ranks with a cross-encoder.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.bm25_index import BM25Index
from app.embedding_engine import EmbeddingEngine
from app.models import DocumentChunk, RankedChunk
from app.profiling import log_latency_breakdown, timed_step
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)

_cross_encoder = None


def _get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Lazy-load the cross-encoder model as a singleton."""
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
    3. Reciprocal Rank Fusion (RRF)
    4. Cross-encoder re-ranking
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
        self.top_k = top_k
        self.top_n = top_n
        self.rrf_k = rrf_k

    def retrieve(self, query: str) -> list[RankedChunk]:
        """Run semantic + BM25 retrieval, RRF merge, and cross-encoder re-rank."""
        logger.info(f"Retrieving for query: {query[:80]}...")
        timings: dict[str, float] = {}

        # BM25 is independent of semantic embedding/Chroma, so overlap the work.
        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(self.bm25_index.search, query, self.top_k)

            with timed_step(timings, "semantic_embedding"):
                query_embedding = self.embedding_engine.embed_query(query)
            with timed_step(timings, "chroma_query"):
                semantic_results = self.vector_store.search(query_embedding, top_k=self.top_k)
            with timed_step(timings, "bm25"):
                bm25_results = bm25_future.result()

        with timed_step(timings, "rrf_merge"):
            merged = self._reciprocal_rank_fusion(semantic_results, bm25_results)

        if not merged:
            logger.warning("No results from retrieval")
            log_latency_breakdown("retrieval", timings)
            return []

        with timed_step(timings, "cross_encoder"):
            reranked = self._cross_encoder_rerank(query, merged)

        logger.info(f"Retrieval complete — {len(reranked)} results")
        log_latency_breakdown("retrieval", timings)
        return reranked[: self.top_n]

    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[dict],
        bm25_results: list[dict],
    ) -> list[RankedChunk]:
        """Merge semantic and BM25 results using Reciprocal Rank Fusion."""
        chunk_map: dict[str, RankedChunk] = {}

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

        for rank, hit in enumerate(bm25_results):
            chunk: DocumentChunk = hit["chunk"]
            chunk_id = chunk.chunk_id
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = RankedChunk(chunk=chunk, retrieval_score=0.0)

            bm25_norm = min(hit.get("score", 0.0) / 20.0, 1.0)
            chunk_map[chunk_id].retrieval_score = max(
                chunk_map[chunk_id].retrieval_score,
                bm25_norm,
            )
            chunk_map[chunk_id].combined_score += 1.0 / (self.rrf_k + rank + 1)

        merged = sorted(chunk_map.values(), key=lambda x: x.combined_score, reverse=True)
        logger.info(
            "RRF merge: %s semantic + %s BM25 -> %s unique",
            len(semantic_results),
            len(bm25_results),
            len(merged),
        )
        return merged

    def _cross_encoder_rerank(self, query: str, candidates: list[RankedChunk]) -> list[RankedChunk]:
        """Re-rank candidates using the cross-encoder."""
        if not candidates:
            return []

        try:
            cross_encoder = _get_cross_encoder(self.cross_encoder_model)
            pairs = [(query, c.chunk.text) for c in candidates]
            scores = cross_encoder.predict(pairs, show_progress_bar=False)

            import math

            raw_score_by_id: dict[str, float] = {}
            for i, candidate in enumerate(candidates):
                raw_score = float(scores[i])
                raw_score_by_id[candidate.chunk.chunk_id] = raw_score
                sigmoid_score = 1 / (1 + math.exp(-raw_score))
                candidate.reranker_score = sigmoid_score
                candidate.combined_score = sigmoid_score

            candidates.sort(key=lambda x: x.reranker_score, reverse=True)

            top_score = candidates[0].reranker_score if candidates else 0.0
            cutoff = 0.20 * top_score
            filtered = [c for c in candidates if c.reranker_score >= cutoff]
            filtered = filtered[: self.top_n]

            logger.info(
                "Cross-encoder re-ranked %s candidates. Filtered %s noisy chunks. Top score: %.3f",
                len(candidates),
                len(candidates) - len(filtered),
                top_score,
            )

            for i, c in enumerate(filtered, 1):
                raw_s = raw_score_by_id.get(c.chunk.chunk_id, 0.0)
                logger.info(
                    "Chunk %s | Source: %s (Page %s) | Raw Cross Encoder Score: %.3f | Sigmoid Score: %.3f",
                    i,
                    c.chunk.document_name,
                    c.chunk.page_number,
                    raw_s,
                    c.reranker_score,
                )

            return filtered

        except Exception as e:
            logger.error(f"Cross-encoder re-ranking failed: {e}. Using RRF scores.", exc_info=True)
            return candidates
