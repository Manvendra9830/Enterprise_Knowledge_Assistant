"""
Vector Store — ChromaDB wrapper for persistent vector storage and retrieval.

Stores document chunk embeddings with metadata. Supports semantic search
with optional document-level filtering.
"""

from __future__ import annotations

import logging
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.models import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Persistent ChromaDB vector store for document chunks."""

    COLLECTION_NAME = "enterprise_knowledge"

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        logger.info(f"Initializing ChromaDB at: {persist_directory}")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB collection '{self.COLLECTION_NAME}' ready — {self.collection.count()} vectors")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> int:
        """Add document chunks with their embeddings to the store."""
        if not chunks or not embeddings:
            return 0

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "document_name": c.document_name,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "file_type": c.file_type,
            }
            for c in chunks
        ]

        # Upsert to handle re-ingestion gracefully
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(f"Upserted {len(chunks)} chunks into ChromaDB")
        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_document: Optional[str] = None,
    ) -> list[dict]:
        """
        Search for similar chunks. Returns list of dicts with keys:
        chunk_id, text, metadata, distance, score
        """
        where_filter = None
        if filter_document:
            where_filter = {"document_name": filter_document}

        count = self.collection.count()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count) if count > 0 else top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        hits = []
        for i, chunk_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i] if results["distances"] else 0.0
            # ChromaDB cosine distance → similarity score (1 - distance)
            score = max(0.0, 1.0 - distance)
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": distance,
                    "score": score,
                }
            )

        return hits

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def get_all_documents(self) -> list[dict]:
        """Return metadata about all unique documents in the store."""
        if self.collection.count() == 0:
            return []

        # Get all metadatas
        all_data = self.collection.get(include=["metadatas"])
        doc_map: dict[str, dict] = {}

        for meta in all_data["metadatas"]:
            name = meta.get("document_name", "unknown")
            if name not in doc_map:
                doc_map[name] = {
                    "name": name,
                    "file_type": meta.get("file_type", "unknown"),
                    "chunk_count": 0,
                }
            doc_map[name]["chunk_count"] += 1

        return list(doc_map.values())

    def delete_document(self, document_name: str) -> int:
        """Delete all chunks for a specific document."""
        # Get IDs for this document
        all_data = self.collection.get(
            where={"document_name": document_name},
            include=[],
        )
        if all_data["ids"]:
            self.collection.delete(ids=all_data["ids"])
            logger.info(f"Deleted {len(all_data['ids'])} chunks for {document_name}")
            return len(all_data["ids"])
        return 0

    def clear(self):
        """Delete the entire collection and recreate it."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection cleared")

    @property
    def count(self) -> int:
        return self.collection.count()
